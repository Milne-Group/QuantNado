from __future__ import annotations

import sys
import math
from pathlib import Path
from collections.abc import Iterable, Mapping

from loguru import logger

NETWORK_FS = {
    "ceph", "nfs", "nfs4", "lustre",
    "glusterfs", "cifs", "smbfs",
    "afs", "davfs"
}

def setup_logging(log_path: Path, verbose: bool):
    logger.remove()
    log_format = "{time:YYYY-MM-DD HH:mm:ss} [{level: <8}] {message}"
    logger.add(log_path, level="DEBUG", format=log_format, mode="w", colorize=False)
    
    # Only colorize stderr if it's a TTY (interactive terminal)
    colorize_stderr = sys.stderr.isatty()
    logger.add(
        sys.stderr,
        level="DEBUG" if verbose else "INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> [<magenta>{level}</magenta>] <level>{message}</level>",
        colorize=colorize_stderr,
    )

def get_filesystem_type(path: str | Path) -> str:
    path = Path(path).resolve()
    best_match = None
    best_len = -1

    with open("/proc/self/mountinfo") as f:
        for line in f:
            parts = line.strip().split()
            # mountinfo format:
            # [..] mount_point [..] - fs_type source super_opts
            sep = parts.index("-")
            mount_point = parts[4]
            fs_type = parts[sep + 1]

            if path.as_posix().startswith(mount_point) and len(mount_point) > best_len:
                best_match = fs_type
                best_len = len(mount_point)

    return best_match or "unknown"


def is_network_fs(path):
    fs = get_filesystem_type(path)
    return fs in NETWORK_FS

def parse_genomic_region(region: str) -> tuple[str, int | None, int | None]:
    """
    Parse a genomic region string into components.
    
    Supports formats:
    - "chr9:77418764-78339335"
    - "chr9:77,418,764-78,339,335" (commas are removed)
    - "chr9" (returns start=None, end=None for whole chromosome)
    
    Parameters
    ----------
    region : str
        Genomic region string to parse.
    
    Returns
    -------
    tuple[str, int | None, int | None]
        (chromosome, start, end) where start/end are 0-based.
        If only chromosome provided, returns (chrom, None, None).
        
    Raises
    ------
    ValueError
        If the region string format is invalid.
        
    Examples
    --------
    >>> parse_genomic_region("chr9:77,418,764-78,339,335")
    ('chr9', 77418764, 78339335)
    >>> parse_genomic_region("chr9")
    ('chr9', None, None)
    """
    # Remove all commas
    region = region.replace(",", "")
    
    # Check if region contains colon (has coordinates)
    if ":" not in region:
        # Just chromosome name
        return region, None, None
    
    # Split into chromosome and coordinates
    try:
        chrom, coords = region.split(":")
        
        # Check for range separator
        if "-" not in coords:
            raise ValueError(f"Invalid region format: {region}. Expected 'chr:start-end' or 'chr'.")
        
        # Split on last dash to handle negative coordinates properly
        # e.g., "chr1:-5-10" -> start="-5", end="10"
        dash_idx = coords.rfind("-")
        if dash_idx == 0:
            # Only one dash at the start, invalid
            raise ValueError(f"Invalid region format: {region}. Expected 'chr:start-end' or 'chr'.")
        
        start_str = coords[:dash_idx]
        end_str = coords[dash_idx + 1:]
        
        start = int(start_str)
        end = int(end_str)
        
        if start < 0 or end < 0:
            raise ValueError(f"Coordinates must be non-negative: {region}")
        if end <= start:
            raise ValueError(f"End coordinate must be greater than start: {region}")
        
        return chrom, start, end
        
    except (ValueError, AttributeError) as e:
        if "non-negative" in str(e) or "greater than start" in str(e):
            # Re-raise our custom validation errors
            raise
        raise ValueError(f"Invalid region format: {region}. Expected 'chr:start-end' or 'chr'. Error: {e}")


def estimate_chunk_len(
    contig_lengths: Iterable[int] | None = None,
    total_positions: int | None = None,
    dtype_bytes: int = 2,
    fs_is_network: bool | None = None,
    *,
    # tuning knobs (sane defaults)
    local_target_bytes: int = 4 * 1024**2,      # 4 MB per chunk on local SSD
    network_target_bytes: int = 128 * 1024**2,   # 128 MB per chunk on network/Ceph
    min_chunk_bytes: int = 64 * 1024,           # don't go smaller than 64 KB chunk
    round_to: int = 1024,                       # round chunk_len to multiple of this
    max_chunks_local: int = 100_000,
    max_chunks_network: int = 10_000,
) -> dict[str, int]:
    """
    Estimate chunk length (positions) for flattened position axis.

    Provide either contig_lengths (iterable of ints) OR total_positions.
    dtype_bytes: bytes per value (uint16=2, uint32=4, etc.)
    fs_is_network: if None, caller should determine and pass True/False; defaults to False.

    Returns a dict:
      {
        "chunk_len": int,           # positions per chunk
        "chunk_bytes": int,         # bytes per chunk (approx)
        "num_chunks": int,          # estimated number of chunks for dataset
        "total_positions": int,
        "fs_is_network": bool
      }
    """
    if contig_lengths is None and total_positions is None:
        raise ValueError("Provide contig_lengths or total_positions")

    if contig_lengths is not None:
        if isinstance(contig_lengths, Mapping):
            contig_list = list(contig_lengths.values())
        else:
            contig_list = list(contig_lengths)
        total = int(sum(contig_list))
        max_contig = max(contig_list) if contig_list else 0
    else:
        total = int(total_positions)
        max_contig = total

    if fs_is_network is None:
        # default conservative assumption: local unless caller overrides
        fs_is_network = False

    target_bytes = network_target_bytes if fs_is_network else local_target_bytes
    target_bytes = max(target_bytes, min_chunk_bytes)

    # initial chunk_len in positions
    chunk_len = max(1, int(target_bytes // max(1, dtype_bytes)))

    # round chunk_len to a convenient boundary
    if round_to > 1:
        chunk_len = max(1, int(round(chunk_len / round_to) * round_to))

    # avoid tiny chunk length
    if chunk_len * dtype_bytes < min_chunk_bytes:
        chunk_len = max(1, int(math.ceil(min_chunk_bytes / dtype_bytes)))
        if round_to > 1:
            chunk_len = int(round(chunk_len / round_to) * round_to) or round_to

    # compute estimated number of chunks and grow chunk_len if there would be too many files
    max_chunks = max_chunks_network if fs_is_network else max_chunks_local
    num_chunks = int(math.ceil(total / chunk_len)) if chunk_len > 0 else int(1e12)

    # double chunk_len until num_chunks <= max_chunks (but don't exceed largest contig too absurdly)
    while num_chunks > max_chunks:
        chunk_len *= 2
        if chunk_len > max_contig and max_contig > 0:
            # cap at max_contig (makes each contig trivially <= 1 chunk)
            chunk_len = max_contig
            break
        num_chunks = int(math.ceil(total / chunk_len))

    # final safety: never exceed total positions
    chunk_len = min(chunk_len, total) if total > 0 else chunk_len
    chunk_bytes = chunk_len * dtype_bytes
    num_chunks = int(math.ceil(total / chunk_len)) if chunk_len > 0 else 0

    return {
        "chunk_len": int(chunk_len),
        "chunk_bytes": int(chunk_bytes),
        "num_chunks": int(num_chunks),
        "total_positions": int(total),
        "fs_is_network": bool(fs_is_network),
    }

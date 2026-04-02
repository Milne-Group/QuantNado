"""Pure utility functions for BAM processing — no Zarr or BamStore dependencies."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Any
import hashlib

import bamnado
import pysam
import numpy as np
import pandas as pd
from loguru import logger


def _copy_read_filter(rf: "bamnado.ReadFilter") -> "bamnado.ReadFilter":
    import bamnado

    if hasattr(rf, "copy"):
        return rf.copy()
    new_rf = bamnado.ReadFilter()
    for attr in (
        "min_mapq",
        "proper_pair",
        "min_length",
        "max_length",
        "strand",
        "min_fragment_length",
        "max_fragment_length",
        "blacklist_bed",
        "whitelisted_barcodes",
        "read_group",
        "filter_tag",
        "filter_tag_value",
    ):
        setattr(new_rf, attr, getattr(rf, attr))
    return new_rf


def _to_str_list(items: Iterable[Any]) -> list[str]:
    return [str(i) if not pd.isna(i) else "" for i in items]


def _compute_sample_hash(sample_names: list[str]) -> str:
    canonical = "|".join(sample_names)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _compute_bam_hash(bam_path: Path | str) -> str:
    h = hashlib.md5()
    try:
        with open(bam_path, "rb") as f:
            h.update(f.read(16384))
    except (FileNotFoundError, PermissionError) as e:
        logger.warning(f"Could not compute hash for {bam_path}: {e}")
        return ""
    return h.hexdigest()


def _collect_bam_stats(bam_file: str) -> tuple[str, int, float]:
    """Return (bam_hash, total_reads, mean_read_length) for a BAM file."""
    bam_hash = _compute_bam_hash(bam_file)
    total_reads = 0
    mean_read_length = float("nan")
    try:
        with pysam.AlignmentFile(bam_file, "rb") as bam:
            total_reads = bam.mapped
            lengths = []
            for read in bam.fetch():
                if not read.is_unmapped and read.query_length:
                    lengths.append(read.query_length)
                    if len(lengths) >= 10_000:
                        break
            if lengths:
                mean_read_length = float(np.mean(lengths))
    except Exception as e:
        logger.warning(f"Could not compute BAM stats for {bam_file}: {e}")
    return bam_hash, total_reads, mean_read_length


def _get_chromsizes_from_bam(bam_path: Path | str) -> dict[str, int]:
    with pysam.AlignmentFile(str(bam_path), "rb") as sam:
        return {ref: length for ref, length in zip(sam.references, sam.lengths)}


def _parse_chromsizes(
    chromsizes: str | Path | dict[str, int],
    *,
    filter_chromosomes: bool = True,
    test: bool = False,
) -> dict[str, int]:
    if isinstance(chromsizes, dict):
        chromsizes_dict = chromsizes
    else:
        path = Path(chromsizes)
        if not path.exists():
            raise FileNotFoundError(f"Chromsizes file not found: {path}")
        df = pd.read_csv(path, sep="\t", header=None, names=["chrom", "size"])
        chromsizes_dict = df.set_index("chrom")["size"].to_dict()

    if filter_chromosomes:
        chromsizes_dict = {
            k: v for k, v in chromsizes_dict.items() if k.startswith("chr") and "_" not in k
        }

    if test:
        desired = ["chr21", "chr22", "chrY"]
        chromsizes_dict = {k: v for k, v in chromsizes_dict.items() if k in desired}
        logger.info(f"Test mode enabled: keeping chromosomes {list(chromsizes_dict.keys())}")

    return chromsizes_dict

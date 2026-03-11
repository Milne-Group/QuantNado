from __future__ import annotations

from pathlib import Path
from typing import Iterable, Any
import os
import shutil
import hashlib
import json
import tempfile
import uuid
import pysam

import bamnado
import numpy as np
import pandas as pd
import zarr
from zarr.storage import LocalStore
from zarr.codecs import BloscCodec
from loguru import logger
import xarray as xr
import dask.array as da

from .metadata import extract_metadata
from quantnado.utils import estimate_chunk_len, is_network_fs

DEFAULT_CHUNK_LEN = 65536
BIN_SIZE = 1
CONSTRUCTION_ARRAY_DTYPE = np.uint32
DEFAULT_CONSTRUCTION_COMPRESSION = "default"

STRANDED_LIBRARY_TYPES = {"R", "F", "U"}


def _normalize_strandedness(
    stranded: "str | list[str] | dict[str, str] | None",
    sample_names: list[str],
) -> dict[str, str | None]:
    """Normalize stranded input to a per-sample dict.

    Parameters
    ----------
    stranded : str | list[str] | dict[str, str] | None
        - ``None`` — no stranded coverage for any sample.
        - ``str`` — apply this library type to all samples.
        - ``list[str]`` — sample names to process with ``"U"``.
        - ``dict[str, str]`` — mapping of sample name → library type.
    sample_names : list[str]
        All sample names in the store.

    Returns
    -------
    dict mapping each sample name to its library type (or None).
    """
    if stranded is None:
        return {s: None for s in sample_names}

    if isinstance(stranded, str):
        if stranded not in STRANDED_LIBRARY_TYPES:
            raise ValueError(
                f"stranded must be one of {STRANDED_LIBRARY_TYPES} or None, got {stranded!r}"
            )
        return {s: stranded for s in sample_names}

    if isinstance(stranded, list):
        unknown = set(stranded) - set(sample_names)
        if unknown:
            raise ValueError(
                f"Unknown sample names in stranded list: {sorted(unknown)}"
            )
        sample_set = set(stranded)
        return {s: ("U" if s in sample_set else None) for s in sample_names}

    if isinstance(stranded, dict):
        unknown = set(stranded) - set(sample_names)
        if unknown:
            raise ValueError(
                f"Unknown sample names in stranded dict: {sorted(unknown)}"
            )
        for sname, lt in stranded.items():
            if lt is not None and lt not in STRANDED_LIBRARY_TYPES:
                raise ValueError(
                    f"stranded[{sname!r}] must be one of {STRANDED_LIBRARY_TYPES}, got {lt!r}"
                )
        return {s: stranded.get(s) for s in sample_names}

    raise TypeError(
        f"stranded must be str, list, dict, or None, got {type(stranded).__name__}"
    )


def _get_stranded_signal(
    bam_path: str | Path,
    chrom: str,
    chrom_size: int,
    library_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract per-base coverage split by strand using pysam.

    Parameters
    ----------
    library_type : str
        ``"U"`` — split reads by raw alignment strand (``+`` → fwd, ``-`` → rev).
        ``"R"`` — ISR / dUTP / TruSeq Stranded (read1 from reverse strand).
        ``"F"`` — ISF / ligation / Directional (read1 from forward strand).

    Returns
    -------
    (fwd, rev) : tuple of np.ndarray uint32
        Per-base coverage for the forward (+) and reverse (−) strands,
        each of length *chrom_size*.
    """
    if library_type not in STRANDED_LIBRARY_TYPES:
        raise ValueError(
            f"library_type must be one of {STRANDED_LIBRARY_TYPES}, got {library_type!r}"
        )

    def _valid(read) -> bool:
        return not read.is_unmapped and not read.is_secondary and not read.is_supplementary

    if library_type == "U":
        def _fwd_cb(read):
            return _valid(read) and not read.is_reverse

        def _rev_cb(read):
            return _valid(read) and read.is_reverse
    else:
        def _is_sense(read) -> bool:
            """True if this read represents the + (sense) strand of the transcript."""
            if read.is_paired:
                if library_type == "R":
                    return (read.is_read1 and read.is_reverse) or (
                        read.is_read2 and not read.is_reverse
                    )
                else:  # F
                    return (read.is_read1 and not read.is_reverse) or (
                        read.is_read2 and read.is_reverse
                    )
            else:  # single-end
                return read.is_reverse if library_type == "R" else not read.is_reverse

        def _fwd_cb(read):
            return _valid(read) and _is_sense(read)

        def _rev_cb(read):
            return _valid(read) and not _is_sense(read)

    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        fwd_raw = bam.count_coverage(chrom, 0, chrom_size, quality_threshold=0, read_callback=_fwd_cb)
        rev_raw = bam.count_coverage(chrom, 0, chrom_size, quality_threshold=0, read_callback=_rev_cb)

    fwd = sum(np.asarray(a, dtype=np.uint32) for a in fwd_raw)
    rev = sum(np.asarray(a, dtype=np.uint32) for a in rev_raw)

    # align to expected length (mirrors bamnado length correction)
    def _align(arr: np.ndarray, size: int) -> np.ndarray:
        if arr.shape[0] > size:
            return arr[:size]
        if arr.shape[0] < size:
            return np.pad(arr, (0, size - arr.shape[0]))
        return arr

    return _align(fwd, chrom_size), _align(rev, chrom_size)


def _to_str_list(values: Iterable[Any]) -> list[str]:
    arr_obj = np.asarray(list(values), dtype=object)
    return ["" if (pd.isna(v) or v is None) else str(v) for v in arr_obj]


def _compute_sample_hash(sample_names: list[str]) -> str:
    """Compute a deterministic hash of the sample names to ensure alignment."""
    canonical = "|".join(sample_names)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _compute_bam_hash(bam_path: Path | str) -> str:
    """Compute a hash of the BAM header/partial start to identify the physical sample."""
    h = hashlib.md5()
    try:
        with open(bam_path, "rb") as f:
            # Read first 16KB which covers most headers and is fast
            h.update(f.read(16384))
    except (FileNotFoundError, PermissionError) as e:
        logger.warning(f"Could not compute hash for {bam_path}: {e}")
        return ""
    return h.hexdigest()


def _resolve_chunk_len(
    chromsizes: dict[str, int],
    store_path: Path,
    chunk_len: int | None,
) -> int:
    if chunk_len is not None:
        resolved = int(chunk_len)
        if resolved <= 0:
            raise ValueError("chunk_len must be a positive integer")
        return resolved

    fs_probe_path = store_path if store_path.exists() else store_path.parent
    fs_is_network = is_network_fs(fs_probe_path)
    estimate = estimate_chunk_len(
        contig_lengths=chromsizes,
        dtype_bytes=np.dtype(CONSTRUCTION_ARRAY_DTYPE).itemsize,
        fs_is_network=fs_is_network,
    )
    resolved = int(estimate["chunk_len"])
    fs_label = "network" if fs_is_network else "local"
    logger.info(
        "Resolved chunk_len={} for {} filesystem at {} ({} estimated chunks)",
        resolved,
        fs_label,
        fs_probe_path,
        estimate["num_chunks"],
    )
    return resolved


def _normalize_construction_compression(profile: str | None) -> str:
    normalized = (profile or DEFAULT_CONSTRUCTION_COMPRESSION).strip().lower()
    aliases = {
        "uncompressed": "none",
        "off": "none",
    }
    normalized = aliases.get(normalized, normalized)
    valid_profiles = {"default", "fast", "none"}
    if normalized not in valid_profiles:
        raise ValueError(
            f"construction_compression must be one of {sorted(valid_profiles)}, got {profile!r}"
        )
    return normalized


def _resolve_construction_compressors(
    profile: str | None,
) -> tuple[str, list[BloscCodec]]:
    normalized = _normalize_construction_compression(profile)
    if normalized == "none":
        return normalized, []
    if normalized == "fast":
        return normalized, [BloscCodec(cname="zstd", clevel=1, shuffle="shuffle")]
    return normalized, [BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")]


def _resolve_staging_root(staging_dir: Path | str | None) -> Path:
    if staging_dir is not None:
        return Path(staging_dir)

    return Path(os.environ.get("TMPDIR") or tempfile.gettempdir())


def _build_staging_store_path(
    final_store_path: Path,
    staging_dir: Path | str | None,
) -> Path:
    staging_root = _resolve_staging_root(staging_dir)
    staging_root.mkdir(parents=True, exist_ok=True)
    return staging_root / f".{final_store_path.stem}.staging-{uuid.uuid4().hex}.zarr"


def _delete_store_path(store_path: Path) -> None:
    if not store_path.exists():
        return

    if store_path.is_dir():
        shutil.rmtree(store_path)
    else:
        store_path.unlink()


def _publish_staged_store(staged_store_path: Path, final_store_path: Path) -> None:
    final_store_path.parent.mkdir(parents=True, exist_ok=True)
    publish_tmp_path = final_store_path.parent / (
        f".{final_store_path.name}.publishing-{uuid.uuid4().hex}"
    )

    try:
        shutil.copytree(staged_store_path, publish_tmp_path)
        _delete_store_path(final_store_path)
        publish_tmp_path.rename(final_store_path)
    except Exception:
        _delete_store_path(publish_tmp_path)
        raise
    finally:
        _delete_store_path(staged_store_path)


def _get_chromsizes_from_bam(bam_path: Path | str) -> dict[str, int]:
    """Extract chromosome names and sizes from a BAM file header."""
    with pysam.AlignmentFile(str(bam_path), "rb") as sam:
        return {ref: length for ref, length in zip(sam.references, sam.lengths)}


def _parse_chromsizes(
    chromsizes: str | Path | dict[str, int],
    *,
    filter_chromosomes: bool = True,
    test: bool = False,
) -> dict[str, int]:
    
    if isinstance(chromsizes, dict) and not filter_chromosomes and not test:
        return chromsizes
    elif isinstance(chromsizes, dict) and (filter_chromosomes) and not test:
        chromsizes = {k: v for k, v in chromsizes.items() if k.startswith("chr") and "_" not in k}
        return chromsizes
    elif isinstance(chromsizes, dict) and test:
        desired = ["chr21", "chr22", "chrY"]
        chromsizes = {k: v for k, v in chromsizes.items() if k in desired}
        return chromsizes

    chromsizes_dict: dict[str, int] = {}
    with open(chromsizes) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                logger.warning(
                    f"Skipping invalid line {line_num} in chromsizes file: {line}"
                )
                continue
            chrom, size_str = parts[0], parts[1]
            try:
                size = int(size_str)
            except ValueError:
                logger.warning(
                    f"Skipping line {line_num} - invalid size '{size_str}': {line}"
                )
                continue
            if filter_chromosomes:
                if chrom.startswith("chr") and "_" not in chrom:
                    chromsizes_dict[chrom] = size
            else:
                chromsizes_dict[chrom] = size

    if test:
        desired = ["chr21", "chr22", "chrY"]
        chromsizes_dict = {
            c: chromsizes_dict[c] for c in desired if c in chromsizes_dict
        }
        logger.info(
            f"Test mode enabled: keeping chromosomes {list(chromsizes_dict.keys())}"
        )

    logger.info(f"Loaded {len(chromsizes_dict)} chromosomes from {chromsizes}")
    return chromsizes_dict


class BamStore:
    """
    Zarr-backed BAM signal store for per-chromosome, per-sample data and metadata.

    Use `from_bam_files` to create a new store, or `open` to load an existing one.

    By default, `open` attaches in read-only mode to prevent accidental data corruption.
    To modify the store, pass `read_only=False`.

    Example:
        # Read-only (default)
        store = BamStore.open("/path/to/store.zarr")
        print(store.sample_names)

        # Writable
        store = BamStore.open("/path/to/store.zarr", read_only=False)
        store.add_metadata_column(...)
    """

    def __init__(
        self,
        store_path: Path | str,
        chromsizes: dict[str, int] | Path | str,
        sample_names: list[str],
        *,
        chunk_len: int | None = None,
        construction_compression: str = DEFAULT_CONSTRUCTION_COMPRESSION,
        overwrite: bool = True,
        resume: bool = False,
        read_only: bool = False,
        stranded: "str | list[str] | dict[str, str] | None" = None,
    ) -> None:
        self.store_path = self._normalize_path(store_path)
        self.chromsizes = _parse_chromsizes(chromsizes)
        self.chromosomes = list(self.chromsizes.keys())
        self.sample_names = [str(s) for s in sample_names]
        self.n_samples = len(self.sample_names)
        self.sample_hash = _compute_sample_hash(self.sample_names)
        self.construction_compression, self.compressors = _resolve_construction_compressors(
            construction_compression
        )
        self.read_only = read_only
        self._strandedness_map: dict[str, str | None] = _normalize_strandedness(
            stranded, self.sample_names
        )
        # Expose a summary attribute: the original value (for attrs storage / repr)
        self.stranded = stranded

        if self.n_samples == 0:
            raise ValueError("sample_names must not be empty")

        if self.store_path.exists():
            if overwrite:
                if read_only:
                    raise ValueError("Cannot overwrite store in read-only mode.")
                logger.warning(f"Deleting existing store at: {self.store_path}")
                if self.store_path.is_dir():
                    shutil.rmtree(self.store_path)
                else:
                    self.store_path.unlink()
                self.chunk_len = _resolve_chunk_len(
                    self.chromsizes,
                    self.store_path,
                    chunk_len,
                )
                self._init_store()
            elif resume:
                self._load_existing()
                self.chunk_len = int(
                    self.root.attrs.get(
                        "chunk_len",
                        _resolve_chunk_len(self.chromsizes, self.store_path, chunk_len),
                    )
                )
                self._validate_sample_names()
            else:
                raise FileExistsError(
                    f"Store already exists at {self.store_path}; set overwrite=True or resume=True"
                )
        else:
            if read_only:
                raise FileNotFoundError(f"Store does not exist at {self.store_path} (read_only=True)")
            self.chunk_len = _resolve_chunk_len(
                self.chromsizes,
                self.store_path,
                chunk_len,
            )
            self._init_store()
    @classmethod
    def open(cls, store_path: str | Path, read_only: bool = True) -> "BamStore":
        """
        Open an existing BAM Zarr store for reading (default) or writing.

        Args:
            store_path: Path to the Zarr store directory.
            read_only: If True (default), disables all write operations.

        Returns:
            BamStore instance attached to the on-disk store.

        Raises:
            FileNotFoundError: If the store or required attributes are missing.
            ValueError: If the store is missing required metadata.
        """
        store_path = cls._normalize_path(store_path)
        if not store_path.exists():
            raise FileNotFoundError(f"Store does not exist at {store_path}")
        # Open Zarr group in appropriate mode
        mode = "r" if read_only else "r+"
        group = zarr.open_group(str(store_path), mode=mode)
        # Read required root attributes
        try:
            sample_names = list(group.attrs["sample_names"])
            chromsizes = dict(group.attrs["chromsizes"])
            chunk_len = int(group.attrs["chunk_len"])
        except KeyError as e:
            raise ValueError(f"Missing required attribute in store: {e}")
        # stranded may be stored as a dict (new format) or string (old format)
        raw_strand = group.attrs.get("stranded")
        if isinstance(raw_strand, dict):
            # Convert back to a per-sample dict, dropping empty strings → None
            stranded: "str | dict[str, str] | None" = {
                s: (lt if lt else None) for s, lt in raw_strand.items()
            }
            # If all values are None, simplify to None
            if not any(stranded.values()):
                stranded = None
        else:
            stranded = raw_strand or None
        # Return BamStore instance
        return cls(
            store_path=store_path,
            chromsizes=chromsizes,
            sample_names=sample_names,
            chunk_len=chunk_len,
            overwrite=False,
            resume=True,
            read_only=read_only,
            stranded=stranded,
        )
    def _check_writable(self):
        if getattr(self, "read_only", False):
            raise RuntimeError("Store is in read-only mode. Reopen with read_only=False to allow modifications.")

    @staticmethod
    def _normalize_path(path: Path | str) -> Path:
        path = Path(path)
        if not str(path).endswith(".zarr"):
            path = path.with_suffix(".zarr")
        return path

    def _init_store(self) -> None:
        store = LocalStore(str(self.store_path))
        self.root = zarr.group(store=store, overwrite=True, zarr_format=3)
        self.meta = self.root.create_group("metadata")

        for chrom, size in self.chromsizes.items():
            self.root.create_array(
                name=chrom,
                shape=(self.n_samples, size),
                chunks=(1, self.chunk_len),
                dtype=CONSTRUCTION_ARRAY_DTYPE,
                compressors=self.compressors,
                fill_value=0,
                overwrite=True,
            )

        self.meta.create_array(
            name="completed",
            shape=(self.n_samples,),
            dtype=bool,
            fill_value=False,
            overwrite=True,
        )
        self.meta.create_array(
            name="sparsity",
            shape=(self.n_samples,),
            dtype=np.float32,
            fill_value=np.nan,
            overwrite=True,
        )
        self.meta.create_array(
            name="sample_hashes",
            shape=(self.n_samples, 16),
            dtype=np.uint8,
            fill_value=0,
            overwrite=True,
        )
        self.meta.create_array(
            name="total_reads",
            shape=(self.n_samples,),
            dtype=np.int64,
            fill_value=0,
            overwrite=True,
        )

        if any(v for v in self._strandedness_map.values() if v):
            for chrom, size in self.chromsizes.items():
                for suffix in ("_fwd", "_rev"):
                    self.root.create_array(
                        name=f"{chrom}{suffix}",
                        shape=(self.n_samples, size),
                        chunks=(1, self.chunk_len),
                        dtype=np.uint32,
                        compressors=self.compressors,
                        fill_value=0,
                        overwrite=True,
                    )

        # Persist stranded as a per-sample dict so it round-trips correctly
        strandedness_for_attrs = {
            s: (lt or "") for s, lt in self._strandedness_map.items()
        }
        self.root.attrs.update(
            {
                "chromosomes": self.chromosomes,
                "chromsizes": self.chromsizes,
                "n_samples": self.n_samples,
                "chunk_len": self.chunk_len,
                "construction_compression": self.construction_compression,
                "structure": "per-chromosome (sample x position)",
                "bin_size": BIN_SIZE,
                "sample_names": self.sample_names,
                "sample_names_hash": self.sample_hash,
                "stranded": strandedness_for_attrs,
            }
        )
        logger.info(f"Initialized Zarr store at {self.store_path}")

    def _load_existing(self) -> None:
        store = LocalStore(str(self.store_path))
        self.root = zarr.open_group(store=store, mode="a")
        self.meta = self.root["metadata"]
        logger.info(f"Resuming existing store at {self.store_path}")

    def _validate_sample_names(self) -> None:
        stored_names = self.root.attrs.get("sample_names")
        stored_hash = self.root.attrs.get("sample_names_hash")

        if stored_names is None:
            raise ValueError(
                "Existing store missing sample_names attribute; cannot validate"
            )

        stored_names = [str(s) for s in stored_names]
        if stored_names != self.sample_names:
            raise ValueError(
                "Provided sample_names do not match stored sample_names; refusing to resume to prevent corruption"
            )

        if stored_hash and stored_hash != self.sample_hash:
            raise ValueError(
                f"Sample names hash mismatch (stored={stored_hash}, current={self.sample_hash}); "
                "the sample list has changed even if names match (unlikely but possible with manual edits)."
            )

    @property
    def completed_mask(self) -> np.ndarray:
        return self.meta["completed"][:].astype(bool)

    @property
    def sample_hashes(self) -> np.ndarray:
        """Return the calculated MD5 hashes for each sample as hex strings."""
        arr = self.meta["sample_hashes"][:]
        return ["".join(f"{b:02x}" for b in row) for row in arr]

    @property
    def library_sizes(self) -> pd.Series | None:
        """
        Total mapped reads per sample as a ``pd.Series`` indexed by sample name.

        Returns ``None`` for stores built before library-size tracking was added.
        Use :func:`quantnado.get_library_sizes` to get a helpful error in that case.
        """
        if "total_reads" not in self.meta:
            return None
        reads = self.meta["total_reads"][:].astype(float)
        reads[~self.completed_mask] = np.nan
        return pd.Series(reads, index=self.sample_names, name="library_size")

    def _process_chromosome(
        self, bam_file: str, contig: str, contig_size: int, library_type: str | None = None
    ) -> tuple[str, np.ndarray, float, np.ndarray | None, np.ndarray | None]:
        signal = bamnado.get_signal_for_chromosome(
            bam_path=bam_file,
            chromosome_name=contig,
            bin_size=BIN_SIZE,
            scale_factor=1.0,
            use_fragment=False,
            ignore_scaffold_chromosomes=False,
        )

        actual_len = signal.shape[0]
        if actual_len != contig_size:
            logger.warning(
                f"Signal length for {contig} differs from chromsizes ({actual_len} vs {contig_size}); aligning"
            )
            if actual_len > contig_size:
                signal = signal[:contig_size]
            else:
                signal = np.pad(signal, (0, contig_size - actual_len), mode="constant")

        max_val = signal.max()
        dtype = np.uint16 if max_val <= np.iinfo(np.uint16).max else np.uint32
        data = signal.astype(dtype, copy=False)
        sparsity = float((np.sum(data == 0) / data.size) * 100)

        fwd_data = rev_data = None
        if library_type:
            fwd_raw, rev_raw = _get_stranded_signal(
                bam_file, contig, contig_size, library_type
            )
            fwd_data = fwd_raw.astype(np.uint32)
            rev_data = rev_raw.astype(np.uint32)

        return contig, data, sparsity, fwd_data, rev_data

    def _process_and_write_single_sample(
        self,
        sample_idx: int,
        bam_file: str,
        sample_name: str,
        chromsizes_dict: dict[str, int],
        max_workers: int = 1,
    ) -> None:
        """Process a single sample by streaming chromosomes directly to disk.

        Unlike the legacy ``_process_single_sample`` this method writes each
        chromosome to the zarr store immediately after extraction, so only one
        chromosome's worth of data is ever held in memory at a time.  This
        reduces peak memory from O(genome_size) to O(max_chromosome_size).
        """
        logger.info(
            f"Processing sample {sample_idx + 1}/{self.n_samples}: {sample_name}"
        )

        bam_hash = _compute_bam_hash(bam_file)
        library_type = self._strandedness_map.get(sample_name)
        sparsity_values: list[float] = []
        contigs = list(chromsizes_dict.keys())

        if max_workers <= 1 or len(contigs) <= 1:
            # Sequential: process + write one chromosome at a time
            for contig, size in chromsizes_dict.items():
                _, data, sparsity, fwd, rev = self._process_chromosome(
                    bam_file, contig, size, library_type
                )
                self.root[contig][sample_idx, : data.shape[0]] = data
                del data
                if fwd is not None:
                    self.root[f"{contig}_fwd"][sample_idx, : fwd.shape[0]] = fwd
                    self.root[f"{contig}_rev"][sample_idx, : rev.shape[0]] = rev
                    del fwd, rev
                sparsity_values.append(sparsity)
        else:
            # Parallel chromosome processing with streaming writes.
            # Process chromosomes in a thread pool (bamnado releases the GIL).
            # Write results as they complete to bound memory.
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(
                        self._process_chromosome,
                        bam_file,
                        contig,
                        chromsizes_dict[contig],
                        library_type,
                    ): contig
                    for contig in contigs
                }
                for future in as_completed(futures):
                    contig, data, sparsity, fwd, rev = future.result()
                    self.root[contig][sample_idx, : data.shape[0]] = data
                    del data
                    if fwd is not None:
                        self.root[f"{contig}_fwd"][sample_idx, : fwd.shape[0]] = fwd
                        self.root[f"{contig}_rev"][sample_idx, : rev.shape[0]] = rev
                        del fwd, rev
                    sparsity_values.append(sparsity)

        # Write sample-level metadata
        self.meta["sparsity"][sample_idx] = (
            float(np.mean(sparsity_values)) if sparsity_values else np.nan
        )
        self.meta["sample_hashes"][sample_idx, :] = 0
        if bam_hash:
            hash_bytes = bytes.fromhex(bam_hash)
            self.meta["sample_hashes"][sample_idx, : len(hash_bytes)] = np.frombuffer(
                hash_bytes, dtype=np.uint8,
            )

        total_reads: int | None = None
        try:
            with pysam.AlignmentFile(bam_file, "rb") as bam:
                total_reads = bam.mapped
        except Exception as e:
            logger.warning(f"Could not read total mapped reads for {bam_file}: {e}")

        if total_reads is not None and "total_reads" in self.meta:
            self.meta["total_reads"][sample_idx] = total_reads

        self.meta["completed"][sample_idx] = True

    def process_samples(
        self,
        bam_files: list[str],
        max_workers: int = 1,
    ) -> None:
        """Process BAM files and write signal data to the zarr store.

        Samples are processed **sequentially** to minimise peak memory.  Each
        sample's chromosomes are streamed directly to disk so only a single
        chromosome's array is ever resident in memory at a time.

        Within each sample, chromosome processing is parallelised using
        ``max_workers`` threads (bamnado releases the GIL).
        """
        if len(bam_files) != self.n_samples:
            raise ValueError("bam_files length must match number of sample_names")

        chromsizes_dict = self.chromsizes
        completed = self.completed_mask

        for sample_idx, (bam_file, sample_name) in enumerate(
            zip(bam_files, self.sample_names)
        ):
            if completed[sample_idx]:
                logger.info(
                    f"Skipping completed sample '{sample_name}' (index {sample_idx})"
                )
                continue

            self._process_and_write_single_sample(
                sample_idx,
                bam_file,
                sample_name,
                chromsizes_dict,
                max_workers=max_workers,
            )

        all_sparsity = self.meta["sparsity"][:]
        if np.isfinite(all_sparsity).any():
            self.root.attrs["average_sparsity"] = float(np.nanmean(all_sparsity))
    
    @classmethod
    def from_bam_files(
        cls,
        bam_files: list[str],
        chromsizes: str | Path | dict[str, int] | None = None,
        store_path: Path | str | None = None,
        metadata: pd.DataFrame | Path | str | list[Path | str] | None = None,
        bam_sample_names: list[str] | None = None,
        *,
        filter_chromosomes: bool = True,
        overwrite: bool = True,
        resume: bool = False,
        sample_column: str = "sample_id",
        chunk_len: int | None = None,
        construction_compression: str = DEFAULT_CONSTRUCTION_COMPRESSION,
        local_staging: bool = False,
        staging_dir: Path | str | None = None,
        max_workers: int = 1,
        log_file: Path | None = None,
        test: bool = False,
        stranded: "str | list[str] | dict[str, str] | None" = None,
    ) -> "BamStore":
        """
        Create BamStore from list of BAM files and optionally attach metadata.

        If chunk_len is omitted, a filesystem-aware value is derived from
        quantnado.utils.estimate_chunk_len using the destination store path.

        If local_staging is enabled or staging_dir is provided, construction is
        performed under scratch storage and then published to the final store path.

        construction_compression controls build-time compression only and does
        not affect reader compatibility.

        max_workers controls chromosome-level parallelism within each sample.
        Samples are processed sequentially to keep memory usage low.
        """

        if log_file is not None:
            from quantnado.utils import setup_logging

            setup_logging(Path(log_file), verbose=False)

        if chromsizes is None:
            if not bam_files:
                raise ValueError(
                    "bam_files list is empty; cannot extract chromsizes from BAM."
                )
            logger.info(f"Extracting chromsizes from {bam_files[0]}")
            chromsizes_raw = _get_chromsizes_from_bam(bam_files[0])
        else:
            chromsizes_raw = chromsizes

        chromsizes_dict = _parse_chromsizes(
            chromsizes_raw, filter_chromosomes=filter_chromosomes, test=test
        )

        if bam_sample_names is not None:
            if len(bam_sample_names) != len(bam_files):
                raise ValueError(
                    f"bam_sample_names length ({len(bam_sample_names)}) != bam_files length ({len(bam_files)})"
                )
            sample_names = bam_sample_names
        else:
            sample_names = [Path(f).stem for f in bam_files]

        if store_path is None:
            raise ValueError("store_path must be provided.")

        final_store_path = cls._normalize_path(store_path)
        staging_enabled = local_staging or staging_dir is not None

        if resume and staging_enabled:
            raise ValueError(
                "resume=True is not supported with local staging; resume the final store directly"
            )

        if staging_enabled and final_store_path.exists() and not overwrite:
            raise FileExistsError(
                f"Store already exists at {final_store_path}; set overwrite=True to publish staged output"
            )

        build_store_path = (
            _build_staging_store_path(final_store_path, staging_dir)
            if staging_enabled
            else final_store_path
        )
        resolved_chunk_len = _resolve_chunk_len(
            chromsizes_dict,
            final_store_path,
            chunk_len,
        )

        if staging_enabled:
            logger.info(
                f"Building dataset under staging path {build_store_path} before publishing to {final_store_path}"
            )

        try:
            store = cls(
                store_path=build_store_path,
                chromsizes=chromsizes_dict,
                sample_names=sample_names,
                chunk_len=resolved_chunk_len,
                construction_compression=construction_compression,
                overwrite=True if staging_enabled else overwrite,
                resume=False if staging_enabled else resume,
                stranded=stranded,
            )
            store.process_samples(bam_files, max_workers=max_workers)

            if metadata is not None:
                if isinstance(metadata, list):
                    metadata_df = cls._combine_metadata_files(metadata)
                elif isinstance(metadata, (str, Path)):
                    metadata_df = pd.read_csv(metadata)
                else:
                    metadata_df = metadata

                store.set_metadata(metadata_df, sample_column=sample_column)

            if staging_enabled:
                _publish_staged_store(build_store_path, final_store_path)
                logger.info(f"Published staged dataset to {final_store_path}")
                return cls.open(final_store_path, read_only=False)

            return store
        except Exception:
            if staging_enabled:
                _delete_store_path(build_store_path)
            raise

    @staticmethod
    def _combine_metadata_files(metadata_files: list[Path | str]) -> pd.DataFrame:
        """Merge multiple CSV metadata files into one DataFrame."""
        if not metadata_files:
            raise ValueError("No metadata files provided")

        frames = []
        for file_path in metadata_files:
            file_path = Path(file_path)
            if not file_path.exists():
                logger.warning(f"Metadata file not found: {file_path}, skipping")
                continue
            frames.append(pd.read_csv(file_path))

        if not frames:
            raise ValueError("No valid metadata files found")

        # Basic union of columns, avoiding R1/R2 paths which might be large/unique
        all_columns = set()
        for df in frames:
            for col in df.columns:
                if "r1" not in col.lower() and "r2" not in col.lower():
                    all_columns.add(col)

        # Standardize columns across all frames
        for i in range(len(frames)):
            for col in all_columns:
                if col not in frames[i].columns:
                    frames[i][col] = ""
            frames[i] = frames[i][list(all_columns)]

        return pd.concat(frames, ignore_index=True)

    @property
    def dataset(self):
        """Expose the root Zarr group."""
        return self.root

    @property
    def n_completed(self) -> int:
        """Return number of completed samples."""
        return int(self.completed_mask.sum())

    def get_metadata(self) -> pd.DataFrame:
        """Retrieve all metadata columns as a DataFrame."""
        return extract_metadata(self.root)

    def list_metadata_columns(self) -> list[str]:
        """List current metadata column names."""
        return [
            k.replace("metadata_", "")
            for k in self.root.attrs.keys()
            if k.startswith("metadata_")
        ]

    def remove_metadata_columns(self, columns: list[str]) -> None:
        """Remove specified metadata columns from the store."""
        self._check_writable()
        for col in columns:
            # Special handling: 'sample_hash' is backed by the metadata group array
            if col == "sample_hash" and "sample_hashes" in self.meta:
                self.meta["sample_hashes"][:] = 0
                logger.info("Cleared sample_hash values in metadata group")
                continue

            key = f"metadata_{col}"
            if key in self.root.attrs:
                del self.root.attrs[key]
                logger.info(f"Removed metadata column: {col}")

    def set_metadata(
        self,
        metadata: pd.DataFrame,
        sample_column: str = "sample_id",
        merge: bool = True,
    ) -> None:
        """
        Store metadata columns from a DataFrame. Subsets of samples are allowed;
        missing samples will have empty strings for the metadata.
        """
        if sample_column not in metadata.columns:
            raise ValueError(
                f"Sample column '{sample_column}' not found in metadata DataFrame"
            )

        meta_subset = metadata.copy()
        meta_subset[sample_column] = meta_subset[sample_column].astype(str)

        if not merge:
            for col in self.list_metadata_columns():
                del self.root.attrs[f"metadata_{col}"]

        # Reindex to match store order, filling gaps with empty strings or existing values
        meta_subset = meta_subset.set_index(sample_column)

        # Optional: Validate hashes if provided in metadata
        if "sample_hash" in meta_subset.columns:
            incoming_hashes = meta_subset["sample_hash"].reindex(
                self.sample_names, fill_value=""
            )
            stored_hashes = self.sample_hashes
            mismatches = []
            for i, (inc, sto) in enumerate(zip(incoming_hashes, stored_hashes)):
                if inc and sto and inc != sto:
                    mismatches.append(
                        f"{self.sample_names[i]}: meta={inc}, store={sto}"
                    )
            if mismatches:
                raise ValueError(
                    f"Sample hash mismatch for: {', '.join(mismatches)}. "
                    "The metadata provided does not seem to match the BAM files used to create this dataset."
                )

        for col in meta_subset.columns:
            target_col = col if col != "assay" else "assay"
            key = f"metadata_{target_col}"

            # If merging and column exists, start with existing values
            if merge and key in self.root.attrs:
                current_values = list(self.root.attrs[key])
                # Update only provided samples
                for i, sample in enumerate(self.sample_names):
                    if sample in meta_subset.index:
                        current_values[i] = str(meta_subset.loc[sample, col])
                values = _to_str_list(current_values)
            else:
                # Full overwrite or new column: reindex filling with ""
                values = _to_str_list(
                    meta_subset[col].reindex(self.sample_names, fill_value="").tolist()
                )

            self.root.attrs[key] = values
            logger.info(f"Updated metadata column: {target_col}")

    def update_metadata(self, updates: dict[str, list[Any] | dict[str, Any]]) -> None:
        """
        Update metadata columns using a dictionary.

        Parameters
        ----------
        updates : dict
            Mapping of column names to either:
            - A list of values (must align exactly with self.sample_names)
            - A dict of {sample_name: value} (subsets allowed, others preserved or filled with "")
        """
        for col, values in updates.items():
            target_col = col if col != "assay" else "assay"
            key = f"metadata_{target_col}"

            if isinstance(values, dict):
                # Start with existing values if available
                if key in self.root.attrs:
                    final_values = list(self.root.attrs[key])
                    for i, sample in enumerate(self.sample_names):
                        if sample in values:
                            final_values[i] = str(values[sample])
                else:
                    final_values = [str(values.get(s, "")) for s in self.sample_names]
            elif isinstance(values, (list, np.ndarray)):
                if len(values) != self.n_samples:
                    raise ValueError(
                        f"Update for {col} has {len(values)} items but store has {self.n_samples}"
                    )
                final_values = _to_str_list(values)
            else:
                raise TypeError(f"Values for {col} must be list or dict")

            self.root.attrs[key] = _to_str_list(final_values)
            logger.info(f"Updated metadata column: {target_col}")

    @classmethod
    def metadata_from_csv(cls, path: Path | str, **kwargs) -> pd.DataFrame:
        """Helper to load metadata from CSV."""
        return pd.read_csv(path, **kwargs)

    def metadata_to_csv(self, path: Path | str) -> None:
        """Export current metadata to CSV."""
        self.get_metadata().to_csv(path)

    @classmethod
    def metadata_from_json(cls, path: Path | str) -> pd.DataFrame:
        """Helper to load metadata from JSON."""
        with open(path) as f:
            data = json.load(f)
        return pd.DataFrame(data)

    def metadata_to_json(self, path: Path | str) -> None:
        """Export current metadata to JSON."""
        self.get_metadata().reset_index().to_json(path, orient="records", indent=2)
    def to_xarray(
        self,
        chromosomes: list[str] | None = None,
        chunks: str | dict | None = None,
    ) -> dict[str, xr.DataArray]:
        """
        Extract the BAM store as a dictionary of per-chromosome Xarray DataArrays.

        Each DataArray uses lazy dask arrays for efficient memory usage. All samples
        must be marked complete; incomplete samples will raise an error.

        Parameters
        ----------
        chromosomes : list[str], optional
            Specific chromosomes to extract. If None, extracts all chromosomes.
        chunks : str or dict, optional
            Dask chunking strategy. Default matches store's chunk_len (per-sample chunks).
            Can be "auto" for automatic optimization or a dict like {"sample": 1, "position": chunk_len}.

        Returns
        -------
        dict[str, xr.DataArray]
            Dictionary mapping chromosome name to DataArray.
            Each DataArray has:
            - Dimensions: (sample, position)
            - Coordinates: sample (names), position (0 to chrom_size), plus all metadata columns
            - Attributes: sample_hashes only

        Raises
        ------
        RuntimeError
            If any sample is marked incomplete.
        ValueError
            If requested chromosomes are not in the store.

        Example
        -------
        >>> store = BamStore.open("/path/to/store.zarr")
        >>> xr_dict = store.to_xarray(chromosomes=["chr1", "chr22"])
        >>> chr1_data = xr_dict["chr1"]
        >>> # Access metadata as coordinates
        >>> chr1_data.coords["cell_type"]
        >>> # Filter by metadata
        >>> chr1_data.sel(cell_type="A549")
        """
        # Strict check: all samples must be complete
        if not self.completed_mask.all():
            incomplete_indices = np.where(~self.completed_mask)[0]
            incomplete_names = [self.sample_names[i] for i in incomplete_indices]
            raise RuntimeError(
                f"Cannot extract Xarray: {len(incomplete_names)} sample(s) incomplete: {incomplete_names}"
            )

        # Default to all chromosomes; validate if subset requested
        chroms_to_extract = chromosomes if chromosomes is not None else self.chromosomes
        invalid_chroms = set(chroms_to_extract) - set(self.chromosomes)
        if invalid_chroms:
            raise ValueError(
                f"Requested chromosomes not in store: {invalid_chroms}. Available: {self.chromosomes}"
            )

        # Default chunking: match store's chunk_len (per-sample chunks)
        if chunks is None:
            chunks = {"sample": 1, "position": self.chunk_len}

        # Extract metadata
        metadata_df = self.get_metadata()
        
        result = {}
        for chrom in chroms_to_extract:
            chrom_size = self.chromsizes[chrom]
            # Load zarr array as dask array with lazy evaluation
            zarr_array = self.root[chrom]
            dask_arr = da.from_array(zarr_array, chunks=chunks)
            # Re-chunk with auto strategy if requested
            if chunks == "auto":
                dask_arr = dask_arr.rechunk("auto")
            elif isinstance(chunks, dict):
                # Convert named dimensions to axis indices for dask
                chunks_by_axis = {}
                dim_names = ("sample", "position")
                for dim_name, chunk_size in chunks.items():
                    if dim_name in dim_names:
                        axis_idx = dim_names.index(dim_name)
                        chunks_by_axis[axis_idx] = chunk_size
                dask_arr = dask_arr.rechunk(chunks_by_axis)

            # Build coordinates dict with sample metadata
            coords = {
                "sample": self.sample_names,
                "position": np.arange(chrom_size),
            }
            # Add all metadata columns as coordinates
            for col in metadata_df.columns:
                if col != "sample_id":  # sample_id is already in "sample" coordinate
                    coords[col] = ("sample", metadata_df[col].values)

            # Create DataArray with coordinates
            da_xr = xr.DataArray(
                dask_arr,
                dims=("sample", "position"),
                coords=coords,
                attrs={
                    "sample_hashes": self.sample_hashes,
                },
            )
            result[chrom] = da_xr

        logger.info(f"Extracted {len(result)} chromosomes as Xarray DataArrays with {len(metadata_df.columns)} metadata columns")
        return result

    def extract_region(
        self,
        region: str | None = None,
        chrom: str | None = None,
        start: int | None = None,
        end: int | None = None,
        samples: list[str] | list[int] | None = None,
        as_xarray: bool = True,
        strand: str | None = None,
    ) -> xr.DataArray | np.ndarray:
        """
        Extract signal data for a specific genomic region.
        
        Parameters
        ----------
        region : str, optional
            Genomic region in format "chr:start-end" or "chr:start,start-end,end".
            Commas in coordinates are removed automatically.
            Alternative to specifying chrom/start/end separately.
        chrom : str, optional
            Chromosome name (alternative to region string).
        start : int, optional
            Start position (0-based, inclusive). If None, defaults to 0.
        end : int, optional
            End position (0-based, exclusive). If None, defaults to chromosome end.
        samples : list of str or int, optional
            Sample names or indices to extract. If None, extracts all completed samples.
        as_xarray : bool, default True
            If True, return xr.DataArray with coordinates (lazy dask array).
            If False, return computed np.ndarray.
        strand : {"+" or "-"}, optional
            Return strand-specific coverage. Requires the store to have been built
            with ``stranded`` set. ``"+"`` returns sense-strand coverage from
            the ``{chrom}_fwd`` array; ``"-"`` returns antisense coverage from
            ``{chrom}_rev``. If None (default), returns total coverage.
        
        Returns
        -------
        xr.DataArray or np.ndarray
            Extracted region data with dimensions (sample, position).
            
        Raises
        ------
        ValueError
            If region format is invalid, chromosome not found, or both region and chrom specified.
        RuntimeError
            If any requested sample is incomplete.
            
        Examples
        --------
        >>> # String format
        >>> data = store.extract_region("chr9:77,418,764-78,339,335")
        >>> 
        >>> # Separate parameters
        >>> data = store.extract_region(chrom="chr9", start=77418764, end=78339335)
        >>> 
        >>> # Whole chromosome
        >>> data = store.extract_region(chrom="chr1")
        >>> 
        >>> # Subset samples
        >>> data = store.extract_region("chr1:1000-2000", samples=["s1", "s2"])
        >>> 
        >>> # Get numpy array instead of xarray
        >>> arr = store.extract_region("chr1:1000-2000", as_xarray=False)
        """
        from ..utils import parse_genomic_region
        
        # Parse region or use separate parameters
        if region is not None and chrom is not None:
            raise ValueError("Specify either 'region' or 'chrom', not both")
        
        if region is not None:
            chrom, parsed_start, parsed_end = parse_genomic_region(region)
            # If start/end were in region string, use them (override None defaults)
            if parsed_start is not None:
                start = parsed_start
            if parsed_end is not None:
                end = parsed_end
        
        if chrom is None:
            raise ValueError("Must specify either 'region' or 'chrom'")
        
        # Validate chromosome exists
        if chrom not in self.chromsizes:
            raise ValueError(
                f"Chromosome '{chrom}' not in store. Available: {list(self.chromsizes.keys())}"
            )
        
        # Default start/end to whole chromosome
        chrom_size = self.chromsizes[chrom]
        if start is None:
            start = 0
        if end is None:
            end = chrom_size
        
        # Validate coordinates
        if start < 0:
            raise ValueError(f"Start position must be >= 0, got {start}")
        if end > chrom_size:
            raise ValueError(
                f"End position {end} exceeds chromosome size {chrom_size} for {chrom}"
            )
        if end <= start:
            raise ValueError(f"End position {end} must be greater than start {start}")
        
        # Resolve sample indices
        if samples is None:
            # Use all samples
            sample_indices = np.arange(len(self.sample_names))
            sample_names = self.sample_names
        else:
            # Parse sample names or indices
            sample_indices = []
            sample_names = []
            for s in samples:
                if isinstance(s, str):
                    if s not in self.sample_names:
                        raise ValueError(f"Sample '{s}' not found in store")
                    idx = self.sample_names.index(s)
                    sample_indices.append(idx)
                    sample_names.append(s)
                elif isinstance(s, int):
                    if s < 0 or s >= len(self.sample_names):
                        raise ValueError(f"Sample index {s} out of range [0, {len(self.sample_names)})")
                    sample_indices.append(s)
                    sample_names.append(self.sample_names[s])
                else:
                    raise TypeError(f"Samples must be strings or integers, got {type(s)}")
            sample_indices = np.array(sample_indices)
        
        # Check all requested samples are complete
        completed = self.meta["completed"][:]
        incomplete_samples = [sample_names[i] for i, idx in enumerate(sample_indices) if not completed[idx]]
        if incomplete_samples:
            raise RuntimeError(
                f"Cannot extract region: {len(incomplete_samples)} sample(s) incomplete: {incomplete_samples}"
            )
        
        # Extract data from zarr store
        if strand is not None:
            if strand not in ("+", "-"):
                raise ValueError(f"strand must be '+', '-', or None, got {strand!r}")
            has_any_stranded = any(v for v in self._strandedness_map.values() if v)
            if not has_any_stranded:
                raise RuntimeError(
                    "This store was not built with stranded; no strand-specific arrays available."
                )
            array_key = f"{chrom}_fwd" if strand == "+" else f"{chrom}_rev"
            zarr_array = self.root[array_key]
        else:
            zarr_array = self.root[chrom]
        # Slice: [sample_indices, start:end]
        data_slice = zarr_array[sample_indices.tolist(), start:end]
        
        if not as_xarray:
            # Return computed numpy array
            return np.array(data_slice)
        
        # Wrap in xarray DataArray with lazy dask array
        dask_arr = da.from_array(data_slice, chunks=(1, -1))  # chunk by sample
        
        # Build coordinates with metadata
        metadata_df = self.get_metadata()
        metadata_subset = metadata_df.iloc[sample_indices]
        
        coords = {
            "sample": sample_names,
            "position": np.arange(start, end),
        }
        
        # Add metadata columns as coordinates
        for col in metadata_subset.columns:
            if col != "sample_id":  # sample_id redundant with "sample" coordinate
                coords[col] = ("sample", np.asarray(metadata_subset[col]))
        
        # Create DataArray
        da_xr = xr.DataArray(
            dask_arr,
            dims=("sample", "position"),
            coords=coords,
            attrs={
                "chromosome": chrom,
                "start": start,
                "end": end,
                "sample_hashes": [self.sample_hashes[i] for i in sample_indices],
            },
        )
        
        logger.info(f"Extracted region {chrom}:{start}-{end} ({end-start} bp) for {len(sample_names)} sample(s)")
        return da_xr
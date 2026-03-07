from __future__ import annotations

from pathlib import Path
from typing import Iterable, Any
import shutil
import hashlib
import json
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

DEFAULT_CHUNK_LEN = 65536
BIN_SIZE = 1

STRANDED_LIBRARY_TYPES = {"fr-firststrand", "fr-secondstrand"}


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
        RNA-seq library type:
        - ``"fr-firststrand"`` — dUTP / RF (most common; e.g. TruSeq stranded)
        - ``"fr-secondstrand"`` — ligation / FR (e.g. Directional)

    Returns
    -------
    (fwd, rev) : tuple of np.ndarray uint32
        Per-base coverage for the sense (+) and antisense (−) strands,
        each of length *chrom_size*.
    """
    if library_type not in STRANDED_LIBRARY_TYPES:
        raise ValueError(
            f"library_type must be one of {STRANDED_LIBRARY_TYPES}, got {library_type!r}"
        )

    def _is_sense(read) -> bool:
        """True if this read represents the + (sense) strand of the transcript."""
        if read.is_paired:
            if library_type == "fr-firststrand":
                return (read.is_read1 and read.is_reverse) or (
                    read.is_read2 and not read.is_reverse
                )
            else:  # fr-secondstrand
                return (read.is_read1 and not read.is_reverse) or (
                    read.is_read2 and read.is_reverse
                )
        else:  # single-end
            return read.is_reverse if library_type == "fr-firststrand" else not read.is_reverse

    def _valid(read) -> bool:
        return not read.is_unmapped and not read.is_secondary and not read.is_supplementary

    def _sense_cb(read):
        return _valid(read) and _is_sense(read)

    def _antisense_cb(read):
        return _valid(read) and not _is_sense(read)

    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        fwd_raw = bam.count_coverage(chrom, 0, chrom_size, quality_threshold=0, read_callback=_sense_cb)
        rev_raw = bam.count_coverage(chrom, 0, chrom_size, quality_threshold=0, read_callback=_antisense_cb)

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
        chunk_len: int = DEFAULT_CHUNK_LEN,
        overwrite: bool = True,
        resume: bool = False,
        read_only: bool = False,
        strandedness: str | None = None,
    ) -> None:
        self.store_path = self._normalize_path(store_path)
        self.chromsizes = _parse_chromsizes(chromsizes)
        self.chromosomes = list(self.chromsizes.keys())
        self.chunk_len = int(chunk_len)
        self.sample_names = [str(s) for s in sample_names]
        self.n_samples = len(self.sample_names)
        self.sample_hash = _compute_sample_hash(self.sample_names)
        self.compressor = BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")
        self.read_only = read_only
        if strandedness is not None and strandedness not in STRANDED_LIBRARY_TYPES:
            raise ValueError(
                f"strandedness must be one of {STRANDED_LIBRARY_TYPES} or None, got {strandedness!r}"
            )
        self.strandedness = strandedness

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
                self._init_store()
            elif resume:
                self._load_existing()
                self._validate_sample_names()
            else:
                raise FileExistsError(
                    f"Store already exists at {self.store_path}; set overwrite=True or resume=True"
                )
        else:
            if read_only:
                raise FileNotFoundError(f"Store does not exist at {self.store_path} (read_only=True)")
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
        strandedness = group.attrs.get("strandedness") or None
        # Return BamStore instance
        return cls(
            store_path=store_path,
            chromsizes=chromsizes,
            sample_names=sample_names,
            chunk_len=chunk_len,
            overwrite=False,
            resume=True,
            read_only=read_only,
            strandedness=strandedness,
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
                dtype=np.uint32,
                compressors=[self.compressor],
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

        if self.strandedness:
            for chrom, size in self.chromsizes.items():
                for suffix in ("_fwd", "_rev"):
                    self.root.create_array(
                        name=f"{chrom}{suffix}",
                        shape=(self.n_samples, size),
                        chunks=(1, self.chunk_len),
                        dtype=np.uint32,
                        compressors=[self.compressor],
                        fill_value=0,
                        overwrite=True,
                    )

        self.root.attrs.update(
            {
                "chromosomes": self.chromosomes,
                "chromsizes": self.chromsizes,
                "n_samples": self.n_samples,
                "chunk_len": self.chunk_len,
                "structure": "per-chromosome (sample x position)",
                "bin_size": BIN_SIZE,
                "sample_names": self.sample_names,
                "sample_names_hash": self.sample_hash,
                "strandedness": self.strandedness or "",
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

    def _process_chromosome(
        self, bam_file: str, contig: str, contig_size: int
    ) -> tuple[str, np.ndarray, float]:
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
        return contig, data, sparsity

    def _process_bam_file(
        self,
        bam_file: str,
        chromsizes_dict: dict[str, int],
        max_workers: int,
    ) -> tuple[dict[str, np.ndarray], list[float]]:
        """Process chromosomes sequentially without Python threading.

        Returns a dict of contig->data and list of sparsity values, preserving
        the original API for tests that monkeypatch this method.
        """
        sample_data: dict[str, np.ndarray] = {}
        sparsity_values: list[float] = []

        for contig, size in chromsizes_dict.items():
            c, data, sparsity = self._process_chromosome(bam_file, contig, size)
            sample_data[c] = data
            sparsity_values.append(sparsity)

        return sample_data, sparsity_values

    # _write_sample removed; writes occur inline in process_samples

    def process_samples(
        self,
        bam_files: list[str],
        max_workers: int = 1,
    ) -> None:
        if len(bam_files) != self.n_samples:
            raise ValueError("bam_files length must match number of sample_names")

        chromsizes_dict = self.chromsizes
        completed = self.completed_mask

        # Helper function to process a single sample
        def _process_single_sample(sample_idx: int, bam_file: str, sample_name: str) -> tuple[int, dict]:
            """Process a single BAM file and return results."""
            if completed[sample_idx]:
                logger.info(
                    f"Skipping completed sample '{sample_name}' (index {sample_idx})"
                )
                return sample_idx, {}

            logger.info(
                f"Processing sample {sample_idx + 1}/{self.n_samples}: {sample_name}"
            )

            bam_hash = _compute_bam_hash(bam_file)
            sparsity_values: list[float] = []
            chr_data = {}
            
            for contig, size in chromsizes_dict.items():
                _c, data, sparsity = self._process_chromosome(bam_file, contig, size)
                chr_data[contig] = data
                sparsity_values.append(sparsity)

            return sample_idx, {
                "sparsity": float(np.mean(sparsity_values)) if sparsity_values else np.nan,
                "hash": bam_hash,
                "chr_data": chr_data,
            }

        # Process samples in parallel or sequentially
        if max_workers > 1:
            from concurrent.futures import ThreadPoolExecutor
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(_process_single_sample, idx, bam_file, sample_name)
                    for idx, (bam_file, sample_name) in enumerate(zip(bam_files, self.sample_names))
                ]
                
                for future in futures:
                    sample_idx, results = future.result()
                    if results:  # Skip if no results (already completed)
                        # Write chromosome data
                        for contig, data in results["chr_data"].items():
                            self.root[contig][sample_idx, : data.shape[0]] = data
                        
                        # Write metadata
                        self.meta["sparsity"][sample_idx] = results["sparsity"]
                        hash_bytes = bytes.fromhex(results["hash"])
                        self.meta["sample_hashes"][sample_idx, : len(hash_bytes)] = np.frombuffer(hash_bytes, dtype=np.uint8)
                        self.meta["completed"][sample_idx] = True
        else:
            # Sequential processing (original behavior)
            for sample_idx, (bam_file, sample_name) in enumerate(
                zip(bam_files, self.sample_names)
            ):
                sample_idx_result, results = _process_single_sample(sample_idx, bam_file, sample_name)
                if results:
                    for contig, data in results["chr_data"].items():
                        self.root[contig][sample_idx, : data.shape[0]] = data
                    
                    self.meta["sparsity"][sample_idx] = results["sparsity"]
                    hash_bytes = bytes.fromhex(results["hash"])
                    self.meta["sample_hashes"][sample_idx, : len(hash_bytes)] = np.frombuffer(hash_bytes, dtype=np.uint8)
                    self.meta["completed"][sample_idx] = True

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
        *,
        filter_chromosomes: bool = True,
        overwrite: bool = True,
        resume: bool = False,
        sample_column: str = "sample_id",
        chunk_len: int = DEFAULT_CHUNK_LEN,
        max_workers: int = 1,
        log_file: Path | None = None,
        test: bool = False,
    ) -> "BamStore":
        """
        Create BamStore from list of BAM files and optionally attach metadata.
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
        sample_names = [Path(f).stem for f in bam_files]

        if store_path is None:
            raise ValueError("store_path must be provided.")

        store = cls(
            store_path=store_path,
            chromsizes=chromsizes_dict,
            sample_names=sample_names,
            chunk_len=chunk_len,
            overwrite=overwrite,
            resume=resume,
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

        return store

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
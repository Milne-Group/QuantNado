"""QuantNadoDataset — unified read-only xarray view over per-sample zarr stores.

Supports two layouts:

1. **Directory of per-sample zarrs** — ``QuantNadoDataset("dataset/")``
   Each ``.zarr`` has ``root.attrs["assay"]`` and ``root.attrs["sample"]``.

2. **Combined zarr** — ``QuantNadoDataset("dataset/combined.zarr")``
   Written by :meth:`QuantNadoDataset.combine`.

Both expose the same API.  Auto-detected on open.
"""

from __future__ import annotations

from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr
import zarr
from loguru import logger


# Keys that are zarr infrastructure — excluded when listing chromosomes / assays
_META_KEYS = frozenset({"metadata"})


# ---------------------------------------------------------------------------
# Layout detection helpers
# ---------------------------------------------------------------------------


def _is_combined_zarr(root: zarr.Group) -> bool:
    """True if this is a combined multi-sample store (has assays attr)."""
    return "assays" in root.attrs or "sample_names" in root.attrs


def _is_per_sample_zarr(root: zarr.Group) -> bool:
    """True if this is a single-sample store (has assay + sample attrs)."""
    return "assay" in root.attrs and "sample" in root.attrs


def _chrom_keys(root: zarr.Group) -> list[str]:
    """Chromosome group keys — excludes metadata and non-chrom entries."""
    return sorted(
        k for k in root.keys()
        if k not in _META_KEYS and isinstance(root[k], zarr.Group)
    )


def _assay_keys(chrom_group: zarr.Group) -> list[str]:
    """Array keys inside a chromosome group."""
    return [k for k in chrom_group.keys() if isinstance(chrom_group[k], zarr.Array)]


# ---------------------------------------------------------------------------
# Internal store descriptors
# ---------------------------------------------------------------------------


class _PerSampleStore:
    """Wraps a single opened per-sample zarr root."""

    def __init__(self, path: Path, root: zarr.Group) -> None:
        self.path = path
        self.root = root
        attrs = dict(root.attrs)
        self.assay = attrs.get("assay", "")
        self.sample = attrs.get("sample", path.stem)
        self.ip = attrs.get("ip", "")
        self.chromsizes: dict[str, int] = {
            str(k): int(v) for k, v in attrs.get("chromsizes", {}).items()
        }
        self.chromosomes = _chrom_keys(root)
        self.chunk_len: int = int(attrs.get("chunk_len", 65536))
        self.viewpoints: list[str] = attrs.get("viewpoints", [])
        meta = root.get("metadata")
        self.completed = bool(meta["completed"][0]) if meta is not None and "completed" in meta else False
        self.total_reads: int = int(meta["total_reads"][0]) if meta is not None and "total_reads" in meta else 0

    def array_keys(self) -> list[str]:
        if not self.chromosomes:
            return []
        return _assay_keys(self.root[self.chromosomes[0]])

    def get_array(self, chrom: str, key: str) -> zarr.Array:
        return self.root[chrom][key]


# ---------------------------------------------------------------------------
# QuantNadoDataset
# ---------------------------------------------------------------------------


class QuantNadoDataset:
    """Unified read-only xarray view over QuantNado zarr stores.

    Parameters
    ----------
    path:
        Path to either:
        - a directory containing per-sample ``.zarr`` stores, or
        - a single combined ``.zarr`` store written by :meth:`combine`.

    Examples
    --------
    >>> qn = QuantNadoDataset("dataset/")
    >>> region = qn.sel(chrom="chr1", start=1_000_000, end=1_001_000)
    >>> region["atac"].sel(sample="ATAC-SEM-1")
    >>> tree = qn.to_datatree()
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._stores: list[_PerSampleStore] = []
        self._combined_root: zarr.Group | None = None
        self._combined = False

        if self.path.is_dir() and not str(self.path).endswith(".zarr"):
            # Directory of per-sample zarrs
            self._load_directory(self.path)
        elif self.path.exists():
            root = zarr.open_group(str(self.path), mode="r")
            if _is_combined_zarr(root):
                self._combined_root = root
                self._combined = True
            elif _is_per_sample_zarr(root):
                # Single per-sample zarr opened directly
                self._stores.append(_PerSampleStore(self.path, root))
            else:
                raise ValueError(f"Cannot determine store layout for {self.path}")
        else:
            raise FileNotFoundError(f"Path does not exist: {self.path}")

    def _load_directory(self, directory: Path) -> None:
        zarr_paths = sorted(directory.glob("*.zarr"))
        if not zarr_paths:
            raise FileNotFoundError(f"No .zarr stores found in {directory}")
        for zp in zarr_paths:
            try:
                root = zarr.open_group(str(zp), mode="r")
                if _is_per_sample_zarr(root):
                    store = _PerSampleStore(zp, root)
                    if not store.completed:
                        logger.warning(f"Skipping incomplete store: {zp.name}")
                        continue
                    self._stores.append(store)
                else:
                    logger.debug(f"Skipping {zp.name}: not a per-sample store")
            except Exception as e:
                logger.warning(f"Could not open {zp}: {e}")
        if not self._stores:
            raise ValueError(f"No completed per-sample stores found in {directory}")
        logger.info(f"Opened {len(self._stores)} stores from {directory}")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def sample_names(self) -> list[str]:
        if self._combined:
            names = self._combined_root.attrs.get("sample_names", [])
            return [str(s) for s in names]
        # Per-sample: all sample names across all stores (MCC may contribute multiple)
        names = []
        for store in self._stores:
            if store.viewpoints:
                names.extend(f"{store.sample}_{vp}" for vp in store.viewpoints)
            else:
                names.append(store.sample)
        return names

    @property
    def assays(self) -> list[str]:
        """All array keys present in the store(s)."""
        if self._combined:
            return list(self._combined_root.attrs.get("assays", []))
        keys: set[str] = set()
        for store in self._stores:
            keys.update(store.array_keys())
        return sorted(keys)

    @property
    def chromosomes(self) -> list[str]:
        if self._combined:
            return _chrom_keys(self._combined_root)
        if not self._stores:
            return []
        return self._stores[0].chromosomes

    @property
    def chromsizes(self) -> dict[str, int]:
        if self._combined:
            stored = self._combined_root.attrs.get("chromsizes", {})
            return {str(k): int(v) for k, v in stored.items()}
        if not self._stores:
            return {}
        return self._stores[0].chromsizes

    @property
    def completed_mask(self) -> np.ndarray:
        if self._combined:
            meta = self._combined_root.get("metadata")
            if meta is not None and "completed" in meta:
                return meta["completed"][:].astype(bool)
            return np.ones(len(self.sample_names), dtype=bool)
        return np.array([s.completed for s in self._stores], dtype=bool)

    # ------------------------------------------------------------------
    # Region selection
    # ------------------------------------------------------------------

    def sel(
        self,
        chrom: str,
        start: int | None = None,
        end: int | None = None,
    ) -> xr.Dataset:
        """Extract a genomic region as an xr.Dataset.

        Parameters
        ----------
        chrom:
            Chromosome name (e.g. ``"chr1"``).
        start:
            1-based start position (inclusive). Defaults to 1.
        end:
            1-based end position (inclusive). Defaults to chromosome length.

        Returns
        -------
        xr.Dataset
            dims: sample × position
            coords: position (1-based), sample
            data_vars: one per assay key (atac, chip_h3k27ac, rna_fwd, …)
        """
        chrom_len = self.chromsizes.get(chrom)
        if chrom_len is None:
            raise ValueError(f"Chromosome '{chrom}' not in store. Available: {self.chromosomes}")

        start_1 = start if start is not None else 1
        end_1 = end if end is not None else chrom_len

        if start_1 < 1:
            raise ValueError(f"start must be >= 1 (1-based), got {start_1}")
        if end_1 > chrom_len:
            raise ValueError(f"end {end_1} exceeds chromosome length {chrom_len}")
        if end_1 < start_1:
            raise ValueError(f"end {end_1} < start {start_1}")

        # 0-based slice for array indexing
        s0 = start_1 - 1
        e0 = end_1

        position_coords = np.arange(start_1, end_1 + 1, dtype=np.int64)

        if self._combined:
            return self._sel_combined(chrom, s0, e0, position_coords)
        return self._sel_per_sample(chrom, s0, e0, position_coords)

    def _sel_per_sample(
        self, chrom: str, s0: int, e0: int, position_coords: np.ndarray
    ) -> xr.Dataset:
        """Build Dataset by grouping per-sample stores by assay key."""
        # Group stores by assay key they contribute
        key_to_stores: dict[str, list[_PerSampleStore]] = {}
        for store in self._stores:
            if chrom not in store.chromosomes:
                continue
            for key in store.array_keys():
                key_to_stores.setdefault(key, []).append(store)

        data_vars: dict[str, xr.DataArray] = {}
        for key, stores in sorted(key_to_stores.items()):
            chunks: list[da.Array] = []
            sample_labels: list[str] = []
            for store in stores:
                arr = store.get_array(chrom, key)  # (1, chrom_len)
                chunk = da.from_zarr(arr)[:, s0:e0]  # (1, region_len)
                chunks.append(chunk)
                sample_labels.append(store.sample)
            stacked = da.concatenate(chunks, axis=0)  # (n_samples, region_len)
            data_vars[key] = xr.DataArray(
                stacked,
                dims=("sample", "position"),
                coords={"sample": sample_labels, "position": position_coords},
            )

        return xr.Dataset(data_vars)

    def _sel_combined(
        self, chrom: str, s0: int, e0: int, position_coords: np.ndarray
    ) -> xr.Dataset:
        """Build Dataset from a combined zarr."""
        root = self._combined_root
        chrom_grp = root[chrom]
        all_assays = _assay_keys(chrom_grp)
        key_to_samples: dict[str, list[str]] = dict(root.attrs.get("key_to_samples", {}))

        data_vars: dict[str, xr.DataArray] = {}
        for key in all_assays:
            arr = chrom_grp[key]  # (n_samples_for_assay, chrom_len)
            dask_arr = da.from_zarr(arr)[:, s0:e0]
            sample_labels = key_to_samples.get(key, [f"{key}_{i}" for i in range(dask_arr.shape[0])])
            data_vars[key] = xr.DataArray(
                dask_arr,
                dims=("sample", "position"),
                coords={"sample": sample_labels, "position": position_coords},
            )

        return xr.Dataset(data_vars)

    # ------------------------------------------------------------------
    # DataTree
    # ------------------------------------------------------------------

    def to_datatree(self, chromosomes: list[str] | None = None, lazy: bool = True) -> xr.DataTree:
        """Return the full dataset as an xr.DataTree.

        Each chromosome is a child node containing an xr.Dataset with
        1-based position coordinates and assay data variables.

        Parameters
        ----------
        chromosomes:
            Subset of chromosomes. Defaults to all.
        lazy:
            If True (default), use lazy dask arrays without materializing
            position coordinate arrays. This is much faster for large datasets.
            If False, materializes the full position coordinate for each chromosome.
        """
        chroms = chromosomes if chromosomes is not None else self.chromosomes
        nodes: dict[str, xr.Dataset] = {
            "/": xr.Dataset(attrs={"chromsizes": self.chromsizes})
        }
        for chrom in chroms:
            chrom_len = self.chromsizes[chrom]
            ds = self.sel(chrom)

            if not lazy:
                # Materialize position coords (slow for large chromosomes)
                position_coords = np.arange(1, chrom_len + 1, dtype=np.int64)
                ds = ds.assign_coords(position=position_coords)

            nodes[chrom] = ds
        return xr.DataTree.from_dict(nodes)

    def extract(
        self,
        feature_type: str = "promoter",
        GTF_FILE: str | None = None,
        fixed_width: int | None = None,
        anchor: str = "start",
        bin_size: int = 50,
        assay: str = "atac",
        samples: list[str] | None = None,
    ) -> xr.DataArray:
        """Extract signal into fixed-width bins around genomic features.

        Parameters
        ----------
        feature_type : str
            Feature type: "promoter", "gene", "transcript", or "exon".
        GTF_FILE : str
            Path to GTF file.
        fixed_width : int, optional
            If provided, expands features to this width around anchor point.
            If None, uses feature length.
        anchor : str
            Anchor point: "start", "end", or "midpoint".
        bin_size : int
            Width of each bin in bp (default: 50).
        assay : str
            Signal array key (default: "atac").
        samples : list of str, optional
            Sample names to extract. If None, uses all samples.

        Returns
        -------
        xr.DataArray
            Shape (interval, bin, sample) with binned signal.
        """
        from .features import load_gtf, extract_feature_ranges, extract_promoters
        from .ranges import extract_signal_into_bins
        import pandas as pd

        if GTF_FILE is None:
            raise ValueError("GTF_FILE is required")

        if samples is None:
            samples = self.sample_names

        # Load and extract features
        gtf = load_gtf(GTF_FILE)

        if feature_type == "promoter":
            features_pr = extract_promoters(gtf, anchor_feature="gene")
        else:
            features_pr = extract_feature_ranges(gtf, feature_type=feature_type)

        features_df = pd.DataFrame(features_pr)

        # Rename PyRanges columns to standard names
        if "Chromosome" in features_df.columns:
            features_df = features_df.rename(columns={"Chromosome": "chrom"})
        if "Start" in features_df.columns:
            features_df = features_df.rename(columns={"Start": "start"})
        if "End" in features_df.columns:
            features_df = features_df.rename(columns={"End": "end"})

        # Apply fixed width if specified
        if fixed_width is not None:
            if anchor == "start":
                center = features_df["start"].values
            elif anchor == "end":
                center = features_df["end"].values
            elif anchor == "midpoint":
                center = (features_df["start"].values + features_df["end"].values) // 2
            else:
                raise ValueError(f"Unknown anchor: {anchor}")

            half_width = fixed_width // 2
            features_df["start"] = center - half_width
            features_df["end"] = center + half_width

        # Convert to 1-based intervals
        intervals = [
            (row["chrom"], int(row["start"]) + 1, int(row["end"]))
            for _, row in features_df.iterrows()
        ]

        # Extract signal into bins
        signal_array = extract_signal_into_bins(
            intervals, self, assay, bin_size, samples
        )

        # Create DataArray
        n_intervals, n_bins, n_samples = signal_array.shape
        interval_ids = np.arange(n_intervals)
        bin_ids = np.arange(n_bins)

        da = xr.DataArray(
            signal_array,
            dims=("interval", "bin", "sample"),
            coords={
                "interval": interval_ids,
                "bin": bin_ids,
                "sample": samples,
            },
        )

        return da

    # ------------------------------------------------------------------
    # Combine
    # ------------------------------------------------------------------


    @classmethod
    def combine(
        cls,
        src: Path | str,
        output: Path | str,
        overwrite: bool = True,
    ) -> "QuantNadoDataset":
        """Combine a directory of per-sample zarrs into a single multi-sample zarr.

        Only ``completed`` stores are included.  Same-assay arrays are stacked
        along axis 0: ``(1, chrom_len) × N → (N, chrom_len)``.

        Parameters
        ----------
        src:
            Directory containing per-sample ``.zarr`` stores.
        output:
            Path for the combined ``.zarr`` output.
        overwrite:
            Delete ``output`` if it already exists.
        """
        from zarr.storage import LocalStore

        src_ds = cls(src)
        if src_ds._combined:
            raise ValueError("src is already a combined store")

        output_path = Path(output)
        if overwrite and output_path.exists():
            import shutil
            shutil.rmtree(output_path) if output_path.is_dir() else output_path.unlink()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        out_root = zarr.group(store=LocalStore(str(output_path)), overwrite=True, zarr_format=3)

        # Collect metadata across all stores
        all_samples: list[str] = src_ds.sample_names
        all_assays: list[str] = src_ds.assays
        all_assay_labels = [
            s.assay for s in src_ds._stores for _ in (s.viewpoints if s.viewpoints else [None])
        ]

        # Group stores by assay key; also build key→sample-names mapping for sel()
        key_to_stores: dict[str, list[_PerSampleStore]] = {}
        for store in src_ds._stores:
            for key in store.array_keys():
                key_to_stores.setdefault(key, []).append(store)

        key_to_samples: dict[str, list[str]] = {}
        for key, stores in key_to_stores.items():
            names: list[str] = []
            for store in stores:
                if store.viewpoints and key.startswith("mcc_"):
                    vp = key[len("mcc_"):]
                    names.append(f"{store.sample}_{vp}")
                else:
                    names.append(store.sample)
            key_to_samples[key] = names

        chromsizes = src_ds.chromsizes
        chunk_len = src_ds._stores[0].chunk_len if src_ds._stores else 65536

        for chrom, chrom_len in chromsizes.items():
            grp = out_root.require_group(chrom)
            for key, stores in key_to_stores.items():
                # Stack: concatenate along sample axis
                chunks_list = []
                for store in stores:
                    if chrom in store.chromosomes:
                        arr = store.get_array(chrom, key)
                        chunks_list.append(da.from_zarr(arr))  # (1, chrom_len)
                if not chunks_list:
                    continue
                stacked = da.concatenate(chunks_list, axis=0)  # (N, chrom_len)
                n = stacked.shape[0]
                # Determine dtype from first array
                first_dtype = chunks_list[0].dtype
                fill = np.nan if np.issubdtype(first_dtype, np.floating) else 0
                out_arr = grp.require_array(
                    key,
                    shape=(n, chrom_len),
                    chunks=(1, chunk_len),
                    dtype=first_dtype,
                    fill_value=fill,
                    overwrite=True,
                )
                da.store(stacked, out_arr)

        # Write combined metadata
        meta_grp = out_root.require_group("metadata")
        # Use VariableLengthUTF8 for string arrays in zarr v3
        from zarr.core.dtype import VariableLengthUTF8
        sn_arr = meta_grp.require_array(
            "sample_names", shape=(len(all_samples),), dtype=VariableLengthUTF8(), overwrite=True
        )
        sn_arr[:] = all_samples
        # Expand per-store metadata to per-sample (MCC stores contribute N viewpoints each)
        completed_list: list[bool] = []
        total_reads_list: list[int] = []
        for s in src_ds._stores:
            n = len(s.viewpoints) if s.viewpoints else 1
            completed_list.extend([s.completed] * n)
            total_reads_list.extend([s.total_reads] * n)
        completed = np.array(completed_list, dtype=bool)
        total_reads_arr = np.array(total_reads_list, dtype=np.int64)
        meta_grp.require_array("completed", shape=(len(all_samples),), dtype=bool, overwrite=True)
        meta_grp["completed"][:] = completed
        meta_grp.require_array("total_reads", shape=(len(all_samples),), dtype=np.int64, overwrite=True)
        meta_grp["total_reads"][:] = total_reads_arr

        out_root.attrs.update({
            "assays": all_assays,
            "sample_names": all_samples,
            "key_to_samples": key_to_samples,
            "chromsizes": chromsizes,
            "chunk_len": chunk_len,
        })

        zarr.consolidate_metadata(str(output_path))
        logger.info(f"Combined {len(src_ds._stores)} stores → {output_path}")
        return cls(output_path)

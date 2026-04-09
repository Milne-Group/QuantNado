from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import zarr
import xarray as xr
import dask.array as da
from loguru import logger


def extract_metadata(ds) -> pd.DataFrame:
    """Extract sample-level metadata from the Zarr-based layout."""

    attrs = getattr(ds, "attrs", {})
    sample_labels = None

    if "sample_names" in attrs:
        sample_labels = attrs.get("sample_names")
    elif hasattr(ds, "meta") and "sample_names" in ds.meta:
        stored = ds.meta["sample_names"][:]
        sample_labels = [
            s.decode() if isinstance(s, (bytes, bytearray)) else str(s) for s in stored
        ]

    if sample_labels is None:
        if hasattr(ds, "sample"):
            sample_labels = getattr(ds, "sample").values.astype(str)
        else:
            root = getattr(ds, "root", None)
            if root is not None:
                sample_labels = root.attrs.get("sample_names")

    if sample_labels is None:
        raise ValueError("Unable to determine sample labels from dataset")

    metadata_df = pd.DataFrame({"sample_id": sample_labels})

    metadata_cols = {
        k.replace("metadata_", ""): v
        for k, v in attrs.items()
        if k.startswith("metadata_") and hasattr(v, "__len__") and len(v) == len(sample_labels)
    }

    for col, values in metadata_cols.items():
        cleaned_values = [v if v != "" else pd.NA for v in values]
        metadata_df[col] = cleaned_values

    meta_group = None
    if hasattr(ds, "get"):
        maybe_meta = ds.get("metadata")
        if isinstance(maybe_meta, zarr.Group):
            meta_group = maybe_meta

        if meta_group is not None:
            if "sample_hashes" in meta_group:
                arr = meta_group["sample_hashes"][:]
                hashes = []
                for row in arr:
                    if (row == 0).all():
                        hashes.append(pd.NA)
                    else:
                        hashes.append("".join(f"{int(b):02x}" for b in row))
                metadata_df["sample_hash"] = hashes
            if "completed" in meta_group:
                metadata_df["completed"] = meta_group["completed"][:].astype(bool)
            if "sparsity" in meta_group:
                metadata_df["sparsity"] = meta_group["sparsity"][:]

    front_cols = ["sample_id"]
    if "assay" in metadata_df.columns:
        front_cols.append("assay")

    remaining = [c for c in metadata_df.columns if c not in front_cols]
    metadata_df = metadata_df[front_cols + remaining]

    return metadata_df.set_index("sample_id")


class BaseStore:
    """Base class for all QuantNado data stores, providing shared read functionality.

    Arrays are stored under a ``coverage/`` group with shape ``(position, n_samples)``.
    Stranded arrays live under ``coverage_fwd/`` and ``coverage_rev/``.
    """

    def __init__(self, path: Path | str, **_: Any) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"The specified path does not exist: {self.path}")

        if str(self.path).endswith(".zarr.zip"):
            store = zarr.storage.ZipStore(str(self.path), mode="r")
            self.root = zarr.open_group(store=store, mode="r")
        else:
            self.root = zarr.open_group(str(self.path), mode="r")
        self._init_common_attributes()

    def _init_common_attributes(self, sample_names: list[str] | None = None) -> None:
        self.meta = self.root.get("metadata")

        stored_names = None
        if self.meta is not None:
            if "sample_names" in self.meta:
                stored_names = self.meta["sample_names"][:]
        if stored_names is None:
            stored_names = self.root.attrs.get("sample_names")

        if stored_names is None:
            raise ValueError("missing sample_names in metadata or attributes")

        self.sample_names = [
            s.decode() if isinstance(s, (bytes, bytearray)) else str(s) for s in stored_names
        ]

        if sample_names is not None:
            provided = [str(s) for s in sample_names]
            if provided != self.sample_names:
                raise ValueError(
                    f"Sample names mismatch. Store has {self.sample_names}, but {provided} was provided."
                )

        self._setup_sample_lookup()

        if self.meta is not None and "completed" in self.meta:
            self.completed_mask_raw = self.meta["completed"][:].astype(bool)
        else:
            self.completed_mask_raw = np.ones(len(self.sample_names), dtype=bool)

        self.n_samples = len(self.sample_names)
        self._chromosomes = None
        self._chromsizes = None
        self._metadata_cache = None

    @property
    def chromosomes(self) -> list[str]:
        """List of chromosome names."""
        if self._chromosomes is None:
            if "coverage" in self.root:
                cov = self.root["coverage"]
                if isinstance(cov, zarr.Group):
                    # Old group layout: keys are chrom names
                    self._chromosomes = sorted(cov.keys())
                else:
                    # Flat array layout: chroms from contig_offsets attrs
                    offsets = self.root.attrs.get("contig_offsets", {})
                    self._chromosomes = sorted(offsets.keys()) if offsets else list(
                        self.root.attrs.get("chromosomes", [])
                    )
            else:
                # Fallback: old layout or methyl/variant stores (keys excluding metadata)
                self._chromosomes = sorted(
                    [
                        k
                        for k in self.root.keys()
                        if k not in ("metadata", "coverage_fwd", "coverage_rev")
                    ]
                )
        return self._chromosomes

    @property
    def chromsizes(self) -> dict[str, int]:
        if self._chromsizes is None:
            stored = self.root.attrs.get("chromsizes")
            if stored is not None:
                self._chromsizes = {str(k): int(v) for k, v in stored.items()}
            elif "coverage" in self.root and isinstance(self.root["coverage"], zarr.Array):
                # Flat layout: derive from contig_offsets
                offsets = self.root.attrs.get("contig_offsets", {})
                self._chromsizes = {c: v[1] - v[0] for c, v in offsets.items()}
            elif "coverage" in self.root and isinstance(self.root["coverage"], zarr.Group):
                self._chromsizes = {c: self.root["coverage"][c].shape[0] for c in self.chromosomes}
            else:
                self._chromsizes = {c: self.root[c].shape[0] for c in self.chromosomes}
        return self._chromsizes

    def _setup_sample_lookup(self) -> None:
        self._sample_name_to_idx = {name: idx for idx, name in enumerate(self.sample_names)}

    @property
    def metadata(self) -> pd.DataFrame:
        if self._metadata_cache is None:
            self._metadata_cache = extract_metadata(self.root)
        return self._metadata_cache

    def _clear_metadata_cache(self) -> None:
        self._metadata_cache = None

    def list_metadata_columns(self) -> list[str]:
        return [
            k.replace("metadata_", "") for k in self.root.attrs.keys() if k.startswith("metadata_")
        ]

    def _check_writable(self):
        if getattr(self, "read_only", False):
            raise RuntimeError(
                "Store is in read-only mode. Reopen with read_only=False to allow modifications."
            )

    @staticmethod
    def _normalize_path(path: "Path | str") -> "Path":
        path = Path(path)
        if str(path).endswith(".zarr.zip") or str(path).endswith(".zarr"):
            return path
        return path.with_suffix(".zarr")

    def _load_existing(self) -> None:
        from zarr.storage import LocalStore
        import zarr as _zarr

        store = LocalStore(str(self.store_path))
        self.root = _zarr.open_group(store=store, mode="a")
        self.meta = self.root.get("metadata")
        logger.info(f"Resuming existing store at {self.store_path}")

    def _validate_sample_names(self) -> None:
        stored = self.root.attrs.get("sample_names")
        if stored is None:
            raise ValueError("Existing store missing sample_names attribute; cannot validate")
        if [str(s) for s in stored] != self.sample_names:
            raise ValueError("sample_names mismatch; refusing to resume to prevent corruption")

    def _contig_row_range(self, chrom: str) -> "tuple[int, int]":
        """Return (start_row, end_row) for a chromosome in the flat arrays."""
        offsets = self.root.attrs.get("contig_offsets", {})
        if chrom not in offsets:
            raise ValueError(f"Chromosome '{chrom}' not in store. Available: {self.chromosomes}")
        start, end = offsets[chrom]
        return int(start), int(end)

    def remove_metadata_columns(self, columns: list[str]) -> None:
        self._check_writable()
        for col in columns:
            key = f"metadata_{col}"
            if key in self.root.attrs:
                del self.root.attrs[key]
        self._clear_metadata_cache()

    def set_metadata(
        self,
        metadata: pd.DataFrame,
        sample_column: str = "sample_id",
        merge: bool = True,
    ) -> None:
        self._check_writable()
        if sample_column not in metadata.columns:
            raise ValueError(f"Sample column '{sample_column}' not found in metadata DataFrame")

        meta_subset = metadata.copy()
        meta_subset[sample_column] = meta_subset[sample_column].astype(str)

        if not merge:
            for col in self.list_metadata_columns():
                del self.root.attrs[f"metadata_{col}"]

        meta_subset = meta_subset.set_index(sample_column)

        if "sample_hash" in meta_subset.columns and hasattr(self, "sample_hashes"):
            incoming_hashes = meta_subset["sample_hash"].reindex(self.sample_names, fill_value="")
            stored_hashes = self.sample_hashes
            mismatches = []
            for i, (inc, sto) in enumerate(zip(incoming_hashes, stored_hashes)):
                if inc and sto and inc != sto:
                    mismatches.append(f"{self.sample_names[i]}: meta={inc}, store={sto}")
            if mismatches:
                raise ValueError(f"Sample hash mismatch for: {', '.join(mismatches)}.")

        for col in meta_subset.columns:
            target_col = str(col)
            key = f"metadata_{target_col}"

            if merge and key in self.root.attrs:
                current_values = list(self.root.attrs[key])
                for i, sample in enumerate(self.sample_names):
                    if sample in meta_subset.index:
                        current_values[i] = str(meta_subset.loc[sample, col])
                values = self._to_str_list(current_values)
            else:
                values = self._to_str_list(
                    meta_subset[col].reindex(self.sample_names, fill_value="").tolist()
                )

            self.root.attrs[key] = values

        self._clear_metadata_cache()

    def update_metadata(self, updates: dict[str, list[Any] | dict[str, Any]]) -> None:
        self._check_writable()
        for col, values in updates.items():
            key = f"metadata_{col}"
            if isinstance(values, dict):
                if key in self.root.attrs:
                    final_values = list(self.root.attrs[key])
                    for i, sample in enumerate(self.sample_names):
                        if sample in values:
                            final_values[i] = str(values[sample])
                else:
                    final_values = [str(values.get(s, "")) for s in self.sample_names]
            elif isinstance(values, (list, np.ndarray)):
                if len(values) != len(self.sample_names):
                    raise ValueError(
                        f"Update for {col} has {len(values)} items but store has {len(self.sample_names)}"
                    )
                final_values = [str(v) for v in values]
            else:
                raise TypeError(f"Values for {col} must be list or dict")
            self.root.attrs[key] = self._to_str_list(final_values)
        self._clear_metadata_cache()

    def _to_str_list(self, items: Iterable[Any]) -> list[str]:
        return [str(i) if not pd.isna(i) else "" for i in items]

    def metadata_to_csv(self, path: Path | str) -> None:
        self.metadata.to_csv(path)

    def metadata_to_json(self, path: Path | str) -> None:
        self.metadata.reset_index().to_json(path, orient="records", indent=2)

    @property
    def completed_mask(self) -> np.ndarray:
        return self.completed_mask_raw

    def _get_coverage_array_and_offset(
        self, chrom: str, suffix: str = "coverage"
    ) -> tuple[zarr.Array, int]:
        """Return ``(zarr_array, row_offset)`` for chromosome-level coverage access.

        For group layout: returns ``(root[suffix][chrom], 0)`` — array is chrom-sized.
        For flat layout: returns ``(root[suffix], chrom_start_row)`` — array is genome-wide.
        """
        arr = self.root.get(suffix)
        if arr is None:
            raise KeyError(f"'{suffix}' not found in store")
        if isinstance(arr, zarr.Group):
            return arr[chrom], 0
        else:
            start, _ = self._contig_row_range(chrom)
            return arr, start

    def get_chrom(self, chrom: str) -> zarr.Array:
        """Return the coverage zarr array for a chromosome: shape (chrom_len, n_samples).

        For flat-layout stores returns the whole-genome array; use
        ``_get_coverage_array_and_offset`` for correct positional access.
        """
        if "coverage" in self.root:
            return self._get_coverage_array_and_offset(chrom)[0]
        return self.root[chrom]

    def valid_sample_indices(self) -> np.ndarray:
        return np.nonzero(self.completed_mask)[0]

    def to_xarray(
        self,
        chromosomes: list[str] | None = None,
        chunks: str | dict | None = None,
        sparse: bool = False,
        strand: str | None = None,
    ) -> dict[str, xr.DataArray]:
        """Extract the dataset as a dict of per-chromosome lazy Xarray DataArrays.

        Each DataArray has dimensions ``(sample, position)``.

        Parameters
        ----------
        sparse:
            If True, each dask chunk is backed by a ``sparse.COO`` array instead of a
            dense NumPy array.  Useful for RNA-seq / ChIP-seq data where most positions
            have zero coverage, significantly reducing memory usage for whole-genome
            access.  Requires the ``sparse`` package (already a project dependency).
        strand:
            ``'+'`` or ``'-'`` to select the forward or reverse strand arrays
            (``coverage_fwd`` / ``coverage_rev``).  Only valid for stranded stores
            (e.g. RNA-seq).  ``None`` uses the unstranded coverage array.
        """
        if strand is not None and strand not in ("+", "-"):
            raise ValueError(f"strand must be '+', '-', or None, got {strand!r}")

        if not self.completed_mask.all():
            incomplete_indices = np.where(~self.completed_mask)[0]
            incomplete_names = [self.sample_names[i] for i in incomplete_indices]
            raise RuntimeError(
                f"Cannot extract Xarray: {len(incomplete_names)} sample(s) incomplete: {incomplete_names}"
            )

        chroms_to_extract = chromosomes if chromosomes is not None else self.chromosomes
        invalid_chroms = set(chroms_to_extract) - set(self.chromosomes)
        if invalid_chroms:
            raise ValueError(
                f"Requested chromosomes not in store: {invalid_chroms}. Available: {self.chromosomes}"
            )

        if chunks is None:
            first_chrom = chroms_to_extract[0]
            zarr_arr, _ = self._get_coverage_array_and_offset(first_chrom)
            chunk_len = self.root.attrs.get("chunk_len") or zarr_arr.chunks[0]
            chunks = {"position": chunk_len, "sample": self.n_samples}

        metadata_df = self.metadata

        # For flat layout, build the genome-wide dask array once and slice per chrom.
        _suffix = ("coverage_fwd" if strand == "+" else "coverage_rev") if strand else "coverage"
        _cov_arr = self.root.get(_suffix)
        _flat = isinstance(_cov_arr, zarr.Array)
        _full_dask = da.from_zarr(_cov_arr) if _flat else None

        result = {}
        for chrom in chroms_to_extract:
            chrom_size = self.chromsizes[chrom]
            if strand is not None:
                suffix = "coverage_fwd" if strand == "+" else "coverage_rev"
                if suffix not in self.root:
                    raise RuntimeError(
                        f"Strand-specific array '{suffix}' not found. "
                        "Store may not be stranded."
                    )
                cov = self.root[suffix]
                if isinstance(cov, zarr.Group) and chrom not in cov:
                    raise RuntimeError(
                        f"Strand-specific array '{suffix}/{chrom}' not found. "
                        "Store may not be stranded."
                    )

            if _flat:
                start_row, end_row = self._contig_row_range(chrom)
                dask_arr = _full_dask[start_row:end_row, :]
                if isinstance(chunks, dict):
                    pos_chunk = chunks.get("position", "auto")
                    smp_chunk = chunks.get("sample", "auto")
                    dask_arr = dask_arr.rechunk({0: pos_chunk, 1: smp_chunk})
                elif chunks == "auto":
                    dask_arr = dask_arr.rechunk("auto")
            else:
                zarr_array, _ = self._get_coverage_array_and_offset(
                    chrom, suffix if strand else "coverage"
                )
                dask_arr = da.from_zarr(zarr_array, chunks=chunks)
                if chunks == "auto":
                    dask_arr = dask_arr.rechunk("auto")
                elif isinstance(chunks, dict):
                    chunks_by_axis = {}
                    dim_names = ("position", "sample")
                    for dim_name, chunk_size in chunks.items():
                        if dim_name in dim_names:
                            chunks_by_axis[dim_names.index(dim_name)] = chunk_size
                    dask_arr = dask_arr.rechunk(chunks_by_axis)

            # Array is (position, sample); transpose to (sample, position) for xarray
            dask_arr = dask_arr.T

            if sparse:
                import sparse as sp

                dask_arr = dask_arr.map_blocks(sp.COO, dtype=dask_arr.dtype)

            coords: dict = {
                "sample": self.sample_names,
                "position": np.arange(chrom_size),
            }
            for col in metadata_df.columns:
                if col != "sample_id":
                    coords[col] = ("sample", metadata_df[col].to_numpy(dtype=object, na_value=None))

            da_xr = xr.DataArray(
                dask_arr,
                dims=("sample", "position"),
                coords=coords,
                attrs={
                    "sample_hashes": metadata_df["sample_hash"].values
                    if "sample_hash" in metadata_df.columns
                    else [],
                },
            )
            result[chrom] = da_xr

        return result

    def to_datatree(
        self,
        chromosomes: list[str] | None = None,
    ) -> xr.DataTree:
        """Return coverage data as a hierarchical :class:`xr.DataTree`.

        Each chromosome is a child node under its coverage group, keeping the
        ``position`` dimension chromosome-local (no size conflicts between contigs).

        Tree structure::

            /
            ├── coverage/
            │   ├── chr1    coverage (position, sample) uint32
            │   └── ...
            ├── coverage_fwd/   (stranded stores only)
            ├── coverage_rev/   (stranded stores only)
            └── metadata/       per-sample metadata variables

        Parameters
        ----------
        chromosomes : list[str], optional
            Chromosomes to include. Defaults to all chromosomes.
        """
        chroms = chromosomes if chromosomes is not None else self.chromosomes

        nodes: dict[str, xr.Dataset] = {
            "/": xr.Dataset(
                attrs={"sample_names": self.sample_names, "chromsizes": self.chromsizes}
            )
        }

        def _add_group(group: zarr.Group, prefix: str) -> None:
            for chrom in chroms:
                if chrom not in group:
                    continue
                zarr_arr = group[chrom]
                chrom_size = self.chromsizes.get(chrom, zarr_arr.shape[0])
                arr = xr.DataArray(
                    da.from_zarr(zarr_arr),
                    dims=("position", "sample"),
                    coords={
                        "position": np.arange(chrom_size),
                        "sample": self.sample_names,
                    },
                    name="coverage",
                )
                nodes[f"{prefix}/{chrom}"] = xr.Dataset({"coverage": arr})

        def _add_flat(flat_arr: zarr.Array, prefix: str) -> None:
            full_dask = da.from_zarr(flat_arr)
            for chrom in chroms:
                start_row, end_row = self._contig_row_range(chrom)
                chrom_size = end_row - start_row
                arr = xr.DataArray(
                    full_dask[start_row:end_row, :],
                    dims=("position", "sample"),
                    coords={
                        "position": np.arange(chrom_size),
                        "sample": self.sample_names,
                    },
                    name="coverage",
                )
                nodes[f"{prefix}/{chrom}"] = xr.Dataset({"coverage": arr})

        if "coverage" in self.root:
            cov = self.root["coverage"]
            if isinstance(cov, zarr.Group):
                _add_group(cov, "coverage")
            else:
                _add_flat(cov, "coverage")
        else:
            # Old layout: arrays at root
            _add_group(self.root, "coverage")

        for suffix in ("coverage_fwd", "coverage_rev"):
            if suffix in self.root:
                arr = self.root[suffix]
                if isinstance(arr, zarr.Group):
                    _add_group(arr, suffix)
                else:
                    _add_flat(arr, suffix)

        meta_df = self.metadata
        meta_vars = {
            col: xr.DataArray(
                meta_df[col].values,
                dims=("sample",),
                coords={"sample": self.sample_names},
            )
            for col in meta_df.columns
            if col != "sample_id"
        }
        if meta_vars:
            nodes["metadata"] = xr.Dataset(meta_vars)

        return xr.DataTree.from_dict(nodes)

    def extract_region(
        self,
        region: str | None = None,
        chrom: str | None = None,
        start: int | None = None,
        end: int | None = None,
        samples: list[str] | list[int] | None = None,
        as_xarray: bool = True,
        strand: str | None = None,
        normalise: str | None = None,
        normalize: str | None = None,
        library_sizes: pd.Series | dict | None = None,
    ) -> xr.DataArray | np.ndarray:
        """Extract signal data for a specific genomic region.

        Returns an array with dimensions ``(sample, position)``.
        """
        from ..utils import parse_genomic_region

        if normalise is not None and normalize is not None and normalise != normalize:
            raise ValueError("Specify only one normalisation method: 'normalise' or 'normalize'")
        normalise = normalise if normalise is not None else normalize

        if region is not None and chrom is not None:
            raise ValueError("Specify either 'region' or 'chrom', not both")

        if region is not None:
            chrom, parsed_start, parsed_end = parse_genomic_region(region)
            if parsed_start is not None:
                start = parsed_start
            if parsed_end is not None:
                end = parsed_end

        if chrom is None:
            raise ValueError("Must specify either 'region' or 'chrom'")

        if chrom not in self.chromosomes:
            has_fwd = "coverage_fwd" in self.root and (
                isinstance(self.root["coverage_fwd"], zarr.Array)
                or chrom in self.root["coverage_fwd"]
            )
            if not has_fwd:
                raise ValueError(
                    f"Chromosome '{chrom}' not in store. Available: {self.chromosomes}"
                )

        chrom_size = self.chromsizes.get(chrom)
        if chrom_size is None:
            fwd = self.root.get("coverage_fwd")
            if fwd is not None:
                if isinstance(fwd, zarr.Array):
                    s, e = self._contig_row_range(chrom)
                    chrom_size = e - s
                elif chrom in fwd:
                    chrom_size = fwd[chrom].shape[0]
            if chrom_size is None:
                raise ValueError(f"Chromosome '{chrom}' size not found")

        if start is None:
            start = 0
        if end is None:
            end = chrom_size

        if start < 0:
            raise ValueError(f"Start position must be >= 0, got {start}")
        if end > chrom_size:
            raise ValueError(f"End position {end} exceeds chromosome size {chrom_size} for {chrom}")
        if end <= start:
            raise ValueError(f"End position {end} must be greater than start {start}")

        if samples is None:
            sample_indices = np.arange(len(self.sample_names))
            sample_names = self.sample_names
        else:
            sample_indices = []
            sample_names = []
            for s in samples:
                if isinstance(s, str):
                    if s not in self._sample_name_to_idx:
                        raise ValueError(f"Sample '{s}' not found in store")
                    idx = self._sample_name_to_idx[s]
                    sample_indices.append(idx)
                    sample_names.append(s)
                elif isinstance(s, int):
                    if s < 0 or s >= len(self.sample_names):
                        raise ValueError(f"Sample index {s} out of range")
                    sample_indices.append(s)
                    sample_names.append(self.sample_names[s])
                else:
                    raise TypeError(f"Samples must be strings or integers, got {type(s)}")
            sample_indices = np.array(sample_indices)

        incomplete_samples = [
            sample_names[i] for i, idx in enumerate(sample_indices) if not self.completed_mask[idx]
        ]
        if incomplete_samples:
            raise RuntimeError(
                f"Cannot extract region: {len(incomplete_samples)} sample(s) incomplete: {incomplete_samples}"
            )

        # Select the right zarr array (and its genome offset for flat stores)
        if strand is not None:
            if strand not in ("+", "-"):
                raise ValueError(f"strand must be '+', '-', or None, got {strand!r}")
            suffix = "coverage_fwd" if strand == "+" else "coverage_rev"
            if suffix not in self.root:
                raise RuntimeError(f"Strand-specific array '{suffix}' not found in store.")
            cov = self.root[suffix]
            if isinstance(cov, zarr.Group) and chrom not in cov:
                raise RuntimeError(f"Strand-specific array '{suffix}/{chrom}' not found in store.")
            zarr_array, chrom_offset = self._get_coverage_array_and_offset(chrom, suffix)
        else:
            zarr_array, chrom_offset = self._get_coverage_array_and_offset(chrom)

        abs_start = chrom_offset + start
        abs_end = chrom_offset + end

        if not as_xarray:
            # Array is (position, sample); slice positions, select samples, transpose to (sample, position)
            result_np = zarr_array[abs_start:abs_end, sample_indices.tolist()].T
            if normalise is None:
                return result_np
            from ..analysis.normalise import normalise as _normalise

            result_xr = xr.DataArray(
                result_np,
                dims=("sample", "position"),
                coords={"sample": sample_names, "position": np.arange(start, end)},
            )
            return _normalise(result_xr, self, method=normalise, library_sizes=library_sizes).values

        chunk_len = self.root.attrs.get("chunk_len") or zarr_array.chunks[0]
        region_len = end - start
        n_sel = len(sample_indices)

        if region_len < 10 * chunk_len:
            data = zarr_array[abs_start:abs_end, sample_indices.tolist()].T
            dask_arr = da.from_array(data, chunks={0: n_sel, 1: min(chunk_len, region_len)})
        else:
            dask_arr = da.from_zarr(zarr_array, chunks={0: chunk_len, 1: n_sel})
            dask_arr = dask_arr[abs_start:abs_end, sample_indices.tolist()].T

        metadata_df = self.metadata
        metadata_subset = metadata_df.iloc[sample_indices]

        coords = {
            "sample": sample_names,
            "position": np.arange(start, end),
        }
        for col in metadata_subset.columns:
            if col != "sample_id":
                coords[col] = ("sample", metadata_subset[col].to_numpy(dtype=object, na_value=None))

        da_xr = xr.DataArray(
            dask_arr,
            dims=("sample", "position"),
            coords=coords,
            attrs={
                "chromosome": chrom,
                "start": start,
                "end": end,
                "sample_hashes": metadata_subset["sample_hash"].values
                if "sample_hash" in metadata_subset.columns
                else [],
            },
        )

        if normalise is None:
            return da_xr

        from ..analysis.normalise import normalise as _normalise

        return _normalise(da_xr, self, method=normalise, library_sizes=library_sizes)


QuantNadoDataset = BaseStore

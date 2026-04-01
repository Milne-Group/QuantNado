from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import zarr
import xarray as xr
import dask.array as da

from .metadata import extract_metadata


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

        self.sample_names = [s.decode() if isinstance(s, (bytes, bytearray)) else str(s) for s in stored_names]

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
        """List of chromosome names (from coverage/ group)."""
        if self._chromosomes is None:
            if "coverage" in self.root:
                self._chromosomes = sorted(self.root["coverage"].keys())
            else:
                # Fallback: old layout or methyl/variant stores (keys excluding metadata)
                self._chromosomes = sorted([k for k in self.root.keys() if k not in ("metadata", "coverage_fwd", "coverage_rev")])
        return self._chromosomes

    @property
    def chromsizes(self) -> dict[str, int]:
        if self._chromsizes is None:
            stored = self.root.attrs.get("chromsizes")
            if stored is not None:
                self._chromsizes = {str(k): int(v) for k, v in stored.items()}
            elif "coverage" in self.root:
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

    def clear_metadata_cache(self) -> None:
        self._metadata_cache = None

    def get_metadata(self) -> pd.DataFrame:
        return self.metadata

    def list_metadata_columns(self) -> list[str]:
        return [
            k.replace("metadata_", "")
            for k in self.root.attrs.keys()
            if k.startswith("metadata_")
        ]

    def _check_writable(self):
        if self.root.mode == "r":
            raise RuntimeError(
                "Store is in read-only mode. Reopen with read_only=False to allow modifications."
            )

    def remove_metadata_columns(self, columns: list[str]) -> None:
        self._check_writable()
        for col in columns:
            key = f"metadata_{col}"
            if key in self.root.attrs:
                del self.root.attrs[key]
        self.clear_metadata_cache()

    def set_metadata(
        self,
        metadata: pd.DataFrame,
        sample_column: str = "sample_id",
        merge: bool = True,
    ) -> None:
        self._check_writable()
        if sample_column not in metadata.columns:
            raise ValueError(
                f"Sample column '{sample_column}' not found in metadata DataFrame"
            )

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
                raise ValueError(
                    f"Sample hash mismatch for: {', '.join(mismatches)}."
                )

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

        self.clear_metadata_cache()

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
        self.clear_metadata_cache()

    def _to_str_list(self, items: Iterable[Any]) -> list[str]:
        return [str(i) if not pd.isna(i) else "" for i in items]

    def metadata_to_csv(self, path: Path | str) -> None:
        self.get_metadata().to_csv(path)

    def metadata_to_json(self, path: Path | str) -> None:
        self.get_metadata().reset_index().to_json(path, orient="records", indent=2)

    @property
    def completed_mask(self) -> np.ndarray:
        return self.completed_mask_raw

    def get_chrom(self, chrom: str) -> zarr.Array:
        """Return the coverage array for a chromosome: shape (chrom_len, n_samples)."""
        if "coverage" in self.root:
            return self.root["coverage"][chrom]
        return self.root[chrom]

    def valid_sample_indices(self) -> np.ndarray:
        return np.nonzero(self.completed_mask)[0]

    def to_xarray(
        self,
        chromosomes: list[str] | None = None,
        chunks: str | dict | None = None,
    ) -> dict[str, xr.DataArray]:
        """Extract the dataset as a dict of per-chromosome lazy Xarray DataArrays.

        Each DataArray has dimensions ``(sample, position)``.
        """
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
            zarr_arr = self.get_chrom(first_chrom)
            chunk_len = self.root.attrs.get("chunk_len") or zarr_arr.chunks[0]
            chunks = {"position": chunk_len, "sample": self.n_samples}

        metadata_df = self.metadata

        result = {}
        for chrom in chroms_to_extract:
            chrom_size = self.chromsizes[chrom]
            zarr_array = self.get_chrom(chrom)
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
                    "sample_hashes": metadata_df["sample_hash"].values if "sample_hash" in metadata_df.columns else [],
                },
            )
            result[chrom] = da_xr

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
            has_fwd = "coverage_fwd" in self.root and chrom in self.root["coverage_fwd"]
            if not has_fwd:
                raise ValueError(
                    f"Chromosome '{chrom}' not in store. Available: {self.chromosomes}"
                )

        chrom_size = self.chromsizes.get(chrom)
        if chrom_size is None:
            if "coverage_fwd" in self.root and chrom in self.root["coverage_fwd"]:
                chrom_size = self.root["coverage_fwd"][chrom].shape[0]
            else:
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

        incomplete_samples = [sample_names[i] for i, idx in enumerate(sample_indices) if not self.completed_mask[idx]]
        if incomplete_samples:
            raise RuntimeError(
                f"Cannot extract region: {len(incomplete_samples)} sample(s) incomplete: {incomplete_samples}"
            )

        # Select the right zarr array
        if strand is not None:
            if strand not in ("+", "-"):
                raise ValueError(f"strand must be '+', '-', or None, got {strand!r}")
            suffix = "coverage_fwd" if strand == "+" else "coverage_rev"
            if suffix not in self.root or chrom not in self.root[suffix]:
                raise RuntimeError(f"Strand-specific array '{suffix}/{chrom}' not found in store.")
            zarr_array = self.root[suffix][chrom]
        else:
            zarr_array = self.get_chrom(chrom)

        if not as_xarray:
            # Array is (position, sample); slice positions, select samples, transpose to (sample, position)
            result_np = zarr_array[start:end, sample_indices.tolist()].T
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
            data = zarr_array[start:end, sample_indices.tolist()].T
            dask_arr = da.from_array(data, chunks={0: n_sel, 1: min(chunk_len, region_len)})
        else:
            dask_arr = da.from_zarr(zarr_array, chunks={0: chunk_len, 1: n_sel})
            dask_arr = dask_arr[start:end, sample_indices.tolist()].T

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
                "sample_hashes": metadata_subset["sample_hash"].values if "sample_hash" in metadata_subset.columns else [],
            },
        )

        if normalise is None:
            return da_xr

        from ..analysis.normalise import normalise as _normalise
        return _normalise(da_xr, self, method=normalise, library_sizes=library_sizes)


QuantNadoDataset = BaseStore

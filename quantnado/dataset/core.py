from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import zarr
import xarray as xr
import dask.array as da

from .constants import DEFAULT_CHUNK_LEN
from .metadata import extract_metadata



class BaseStore:
    """Base class for all QuantNado data stores, providing shared read functionality."""

    def __init__(self, path: Path | str, **_: Any) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"The specified path does not exist: {self.path}")

        self.root = zarr.open_group(str(self.path), mode="r")
        self._init_common_attributes()

    def _init_common_attributes(self, sample_names: list[str] | None = None) -> None:
        """Initialize attributes from existing store or provided sample names."""
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

        # Validation on resume/existing store
        if sample_names is not None:
            provided = [str(s) for s in sample_names]
            if provided != self.sample_names:
                 raise ValueError(
                    f"Sample names mismatch. These names do not match the existing store. "
                    f"Store has {self.sample_names}, but {provided} was provided."
                )
        
        self._setup_sample_lookup()
        
        if self.meta is not None and "completed" in self.meta:
            self.completed_mask_raw = self.meta["completed"][:].astype(bool)
        else:
            self.completed_mask_raw = np.ones(len(self.sample_names), dtype=bool)

        self._chromosomes = None
        self._chromsizes = None
        self._metadata_cache = None

    @property
    def chromosomes(self) -> list[str]:
        """List of chromosome names in the store."""
        if self._chromosomes is None:
            self._chromosomes = sorted([k for k in self.root.keys() if k != "metadata"])
        return self._chromosomes

    @property
    def chromsizes(self) -> dict[str, int]:
        """Dictionary mapping chromosome name to size."""
        if self._chromsizes is None:
            self._chromsizes = self.root.attrs.get(
                "chromsizes", {c: self.root[c].shape[1] for c in self.chromosomes}
            )
        return self._chromsizes

    def _setup_sample_lookup(self) -> None:
        """Initialize O(1) sample index lookup."""
        self._sample_name_to_idx = {name: idx for idx, name in enumerate(self.sample_names)}

    @property
    def metadata(self) -> pd.DataFrame:
        """Retrieve all metadata columns as a DataFrame (cached)."""
        if self._metadata_cache is None:
            self._metadata_cache = extract_metadata(self.root)
        return self._metadata_cache

    def clear_metadata_cache(self) -> None:
        """Invalidate the internal metadata cache."""
        self._metadata_cache = None

    def get_metadata(self) -> pd.DataFrame:
        """Retrieve all metadata columns as a DataFrame."""
        return self.metadata

    def list_metadata_columns(self) -> list[str]:
        """List current metadata column names."""
        return [
            k.replace("metadata_", "")
            for k in self.root.attrs.keys()
            if k.startswith("metadata_")
        ]

    def _check_writable(self):
        """Check if the store is opened in a writable mode."""
        if self.root.mode == "r":
            raise RuntimeError(
                "Store is in read-only mode. Reopen with read_only=False to allow modifications."
            )

    def remove_metadata_columns(self, columns: list[str]) -> None:
        """Remove specified metadata columns from the store."""
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
        """
        Store metadata columns from a DataFrame. Subsets of samples are allowed;
        missing samples will have empty strings for the metadata.
        """
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

        # Reindex to match store order, filling gaps with empty strings or existing values
        meta_subset = meta_subset.set_index(sample_column)

        # Optional: Validate hashes if provided in metadata and available in store
        if "sample_hash" in meta_subset.columns and hasattr(self, "sample_hashes"):
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
                    "The metadata provided does not seem to match the samples in this dataset."
                )

        for col in meta_subset.columns:
            target_col = str(col)
            key = f"metadata_{target_col}"

            # If merging and column exists, start with existing values
            if merge and key in self.root.attrs:
                current_values = list(self.root.attrs[key])
                # Update only provided samples
                for i, sample in enumerate(self.sample_names):
                    if sample in meta_subset.index:
                        current_values[i] = str(meta_subset.loc[sample, col])
                values = self._to_str_list(current_values)
            else:
                # Full overwrite or new column: reindex filling with ""
                values = self._to_str_list(
                    meta_subset[col].reindex(self.sample_names, fill_value="").tolist()
                )

            self.root.attrs[key] = values
        
        self.clear_metadata_cache()

    def update_metadata(self, updates: dict[str, list[Any] | dict[str, Any]]) -> None:
        """Update metadata columns using a dictionary."""
        self._check_writable()
        for col, values in updates.items():
            key = f"metadata_{col}"

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
        """Convert a list of items to a list of strings for Zarr attributes."""
        return [str(i) if not pd.isna(i) else "" for i in items]

    def metadata_to_csv(self, path: Path | str) -> None:
        """Export current metadata to CSV."""
        self.get_metadata().to_csv(path)

    def metadata_to_json(self, path: Path | str) -> None:
        """Export current metadata to JSON."""
        self.get_metadata().reset_index().to_json(path, orient="records", indent=2)

    @property
    def completed_mask(self) -> np.ndarray:
        return self.completed_mask_raw

    def get_chrom(self, chrom: str):
        return self.root[chrom]

    def valid_sample_indices(self) -> np.ndarray:
        return np.nonzero(self.completed_mask)[0]

    def to_xarray(
        self,
        chromosomes: list[str] | None = None,
        chunks: str | dict | None = None,
    ) -> dict[str, xr.DataArray]:
        """
        Extract the dataset as a dictionary of per-chromosome Xarray DataArrays.

        Each DataArray uses lazy dask arrays for efficient memory usage. All samples
        must be marked complete; incomplete samples will raise an error.

        Parameters
        ----------
        chromosomes : list[str], optional
            Specific chromosomes to extract. If None, extracts all chromosomes.
        chunks : str or dict, optional
            Dask chunking strategy. Default matches store's chunk_len.
            Can be "auto" for automatic optimization.

        Returns
        -------
        dict[str, xr.DataArray]
            Dictionary mapping chromosome name to DataArray with lazy dask arrays,
            including all metadata columns as coordinates.

        Raises
        ------
        RuntimeError
            If any sample is marked incomplete.
        ValueError
            If requested chromosomes are not in the store.
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

        # Default chunking: prefer root attrs, fall back to actual array chunk shape
        if chunks is None:
            first_chrom = chroms_to_extract[0]
            chunk_len = self.root.attrs.get("chunk_len") or self.root[first_chrom].chunks[1]
            chunks = {"sample": 1, "position": chunk_len}

        # Extract metadata
        metadata_df = self.metadata
        
        result = {}
        for chrom in chroms_to_extract:
            chrom_size = self.chromsizes[chrom]
            # Load zarr array as dask array with lazy evaluation
            zarr_array = self.root[chrom]
            dask_arr = da.from_zarr(zarr_array, chunks=chunks)
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
                    coords[col] = ("sample", metadata_df[col].to_numpy(dtype=object, na_value=None))

            # Create DataArray with coordinates and metadata attributes
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
        normalise : {"cpm", "rpkm"}, optional
            Normalise the extracted signal before returning it. If omitted,
            raw coverage is returned.
        normalize : {"cpm", "rpkm"}, optional
            American-English alias for ``normalise``.
        library_sizes : pd.Series or dict, optional
            Total mapped reads per sample, indexed by sample name. Overrides
            automatic lookup from the store when ``normalise`` is used.
        
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
        """
        from ..utils import parse_genomic_region

        if normalise is not None and normalize is not None and normalise != normalize:
            raise ValueError("Specify only one normalisation method: 'normalise' or 'normalize'")
        normalise = normalise if normalise is not None else normalize
        
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
        if chrom not in self.chromosomes:
             # Check if it exists as a stranded array
             if not (f"{chrom}_fwd" in self.root or f"{chrom}_rev" in self.root):
                raise ValueError(
                    f"Chromosome '{chrom}' not in store. Available: {self.chromosomes}"
                )
        
        # Default start/end to whole chromosome
        # For stranded arrays, we need the size from the total array or one of the strands
        if chrom in self.chromsizes:
            chrom_size = self.chromsizes[chrom]
        elif f"{chrom}_fwd" in self.root:
            chrom_size = self.root[f"{chrom}_fwd"].shape[1]
        elif f"{chrom}_rev" in self.root:
            chrom_size = self.root[f"{chrom}_rev"].shape[1]
        else:
            raise ValueError(f"Chromosome '{chrom}' size not found")

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
                    if s not in self._sample_name_to_idx:
                        raise ValueError(f"Sample '{s}' not found in store")
                    idx = self._sample_name_to_idx[s]
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
        incomplete_samples = [sample_names[i] for i, idx in enumerate(sample_indices) if not self.completed_mask[idx]]
        if incomplete_samples:
            raise RuntimeError(
                f"Cannot extract region: {len(incomplete_samples)} sample(s) incomplete: {incomplete_samples}"
            )
        
        # Extract data from zarr store
        if strand is not None:
            if strand not in ("+", "-"):
                raise ValueError(f"strand must be '+', '-', or None, got {strand!r}")
            array_key = f"{chrom}_fwd" if strand == "+" else f"{chrom}_rev"
            if array_key not in self.root:
                raise RuntimeError(
                    f"Strand-specific array '{array_key}' not found in store."
                )
            zarr_array = self.root[array_key]
        else:
            zarr_array = self.root[chrom]
            
        if not as_xarray:
            # Return computed numpy array (eagerly slice zarr)
            result_np = zarr_array[sample_indices.tolist(), start:end]
            if normalise is None:
                return result_np

            from ..analysis.normalise import normalise as _normalise

            result_xr = xr.DataArray(
                result_np,
                dims=("sample", "position"),
                coords={
                    "sample": sample_names,
                    "position": np.arange(start, end),
                },
            )
            return _normalise(
                result_xr,
                self,
                method=normalise,
                library_sizes=library_sizes,
            ).values
        
        # Wrap in xarray DataArray with lazy dask array
        # Prefer root attrs, fall back to actual array chunk shape
        chunk_len = self.root.attrs.get("chunk_len") or zarr_array.chunks[1]
        region_len = end - start
        if region_len < 10 * chunk_len:
            # Small region: eagerly load only the touched zarr chunks, then wrap in dask.
            # Building da.from_zarr over the full chromosome would create thousands of
            # graph tasks for a query that touches just a handful of chunks.
            data = zarr_array.oindex[sample_indices.tolist(), start:end]
            dask_arr = da.from_array(data, chunks={0: 1, 1: min(chunk_len, region_len)})
        else:
            # Large region (whole-chromosome or similar): keep fully lazy graph.
            dask_arr = da.from_zarr(zarr_array, chunks={0: 1, 1: chunk_len})
            dask_arr = dask_arr[sample_indices.tolist(), start:end]

        
        # Build coordinates with metadata
        metadata_df = self.metadata
        metadata_subset = metadata_df.iloc[sample_indices]
        
        coords = {
            "sample": sample_names,
            "position": np.arange(start, end),
        }
        
        # Add metadata columns as coordinates
        for col in metadata_subset.columns:
            if col != "sample_id":  # sample_id redundant with "sample" coordinate
                coords[col] = ("sample", metadata_subset[col].to_numpy(dtype=object, na_value=None))
        
        # Create DataArray
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

        return _normalise(
            da_xr,
            self,
            method=normalise,
            library_sizes=library_sizes,
        )




# # Subset promoter signal to RNA samples
# assay_by_sample = list(ds.attrs.get("assay_by_sample", []))
# sample_names = [str(s) for s in ds.attrs.get("sample_names", [])]
# rna_samples = [s for s, a in zip(sample_names, assay_by_sample) if a == "RNA"]

# sample_dim = "sample" if "sample" in promoter_ds.dims else "sample_id"
# promoter_ds = promoter_ds.assign_coords({sample_dim: sample_names})

# coord_samples = [str(s) for s in promoter_ds.coords[sample_dim].values.tolist()]
# valid_samples = [s for s in rna_samples if s in coord_samples]

# promoter_rna_ds = promoter_ds.sel({sample_dim: valid_samples})
# print("RNA samples in promoter_rna_ds:", promoter_rna_ds.coords[sample_dim].values)
    

QuantNadoDataset = BaseStore

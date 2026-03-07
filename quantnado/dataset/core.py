from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import zarr
import xarray as xr
import dask.array as da

from .bam import DEFAULT_CHUNK_LEN
from .metadata import extract_metadata


class QuantNadoDataset:
    """Lightweight wrapper for the per-chromosome Zarr layout."""

    def __init__(self, path: Path | str, **_: Any) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"The specified path does not exist: {self.path}")

        self.root = zarr.open_group(str(self.path), mode="r")
        self.meta = self.root.get("metadata")
        if self.meta is None:
            raise ValueError("Zarr store missing required 'metadata' group")

        stored_names = None
        if self.meta is not None and "sample_names" in self.meta:
            stored_names = self.meta["sample_names"][:]
        if stored_names is None:
            stored_names = self.root.attrs.get("sample_names")
        if stored_names is None:
            raise ValueError("Sample names not found in metadata or attributes")
        self.sample_names = [s.decode() if isinstance(s, (bytes, bytearray)) else str(s) for s in stored_names]
        self.completed_mask = self.meta["completed"][:].astype(bool)
        self.chromosomes = [k for k in self.root.keys() if k != "metadata"]
        self.chromsizes = self.root.attrs.get(
            "chromsizes", {c: self.root[c].shape[1] for c in self.chromosomes}
        )

    def get_chrom(self, chrom: str):
        return self.root[chrom]

    def valid_sample_indices(self) -> np.ndarray:
        return np.nonzero(self.completed_mask)[0]

    @property
    def metadata(self) -> pd.DataFrame:
        return extract_metadata(self.root)

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

        # Default chunking: get from root attrs if available
        chunk_len = self.root.attrs.get("chunk_len", DEFAULT_CHUNK_LEN)
        if chunks is None:
            chunks = {"sample": 1, "position": chunk_len}

        # Extract metadata
        metadata_df = self.metadata
        
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
        >>> data = ds.extract_region("chr9:77,418,764-78,339,335")
        >>> 
        >>> # Separate parameters
        >>> data = ds.extract_region(chrom="chr9", start=77418764, end=78339335)
        >>> 
        >>> # Whole chromosome
        >>> data = ds.extract_region(chrom="chr1")
        >>> 
        >>> # Subset samples
        >>> data = ds.extract_region("chr1:1000-2000", samples=["s1", "s2"])
        >>> 
        >>> # Get numpy array instead of xarray
        >>> arr = ds.extract_region("chr1:1000-2000", as_xarray=False)
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
        if chrom not in self.chromosomes:
            raise ValueError(
                f"Chromosome '{chrom}' not in store. Available: {self.chromosomes}"
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
        incomplete_samples = [sample_names[i] for i, idx in enumerate(sample_indices) if not self.completed_mask[idx]]
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
        
        return da_xr



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
    





        
    

   
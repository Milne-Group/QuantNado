"""Combine multiple single-sample QuantNado stores into a unified multiomics xarray Dataset."""

from __future__ import annotations

import shutil
from pathlib import Path

import xarray as xr
import zarr
from loguru import logger


def combine_multiomics_stores(
    stores: dict[str, list[str | Path]] | list[str | Path],
    output_path: str | Path | None = None,
    *,
    overwrite: bool = False,
) -> xr.Dataset:
    """Combine per-assay Zarr stores into a unified multiomics xarray Dataset.

    Parameters
    ----------
    stores
        Dictionary mapping assay name to store paths, or list of store paths.
    output_path
        Optional path to write combined store. If None, only returns in-memory Dataset.
    overwrite
        If True, delete existing store at output_path before writing.

    Returns
    -------
    xr.Dataset
        Merged multiomics dataset with data-variable-specific position dimensions
        (e.g., coverage_position, methylation_pct_position) to avoid conflicts.
    """
    import glob

    # Normalize input: if list, expand globs or use as-is
    if isinstance(stores, list):
        store_paths = []
        for item in stores:
            item_str = str(item)
            if "*" in item_str:
                store_paths.extend(glob.glob(item_str))
            else:
                store_paths.append(item_str)
        stores = {"default": store_paths}

    if not stores:
        raise ValueError("stores must contain at least one assay")

    # Load all stores as xarray Datasets (lazy, via zarr)
    all_datasets = []
    all_assays = []
    all_sample_names = []
    sample_idx = 0

    for assay_name, store_paths in stores.items():
        for store_path in store_paths:
            root = zarr.open(str(store_path), mode="r")
            sample_names = list(root.attrs.get("sample_names", []))

            data_vars = {}
            coords = {}

            for key in root.keys():
                if key == "metadata":
                    continue
                arr = root[key]
                if isinstance(arr, zarr.Array):
                    if arr.ndim == 1:
                        # Skip 1D coordinates to avoid conflicts
                        pass
                    elif arr.ndim == 2 and arr.shape[1] == len(sample_names):
                        # Each data variable gets its own position dimension
                        pos_dim = f"{key}_position"
                        data_vars[key] = ([pos_dim, "sample"], arr)

            if data_vars:
                # Use absolute sample indices across all stores
                coords["sample"] = range(sample_idx, sample_idx + len(sample_names))
                ds = xr.Dataset(data_vars, coords=coords)
                all_datasets.append(ds)

                all_assays.extend([assay_name] * len(sample_names))
                all_sample_names.extend(sample_names)
                sample_idx += len(sample_names)

    if not all_datasets:
        raise ValueError("No valid stores found")

    # Merge all datasets (now have non-overlapping sample indices)
    ds_combined = xr.merge(all_datasets, join="override")

    # Add assay and original sample name as coordinates
    ds_combined.coords["assay"] = ("sample", all_assays)
    ds_combined.coords["sample_name"] = ("sample", all_sample_names)
    ds_combined.attrs["multiomics"] = True

    # Write to disk if output path provided
    if output_path:
        output_path = Path(output_path)
        if output_path.exists():
            if overwrite:
                if output_path.is_dir():
                    shutil.rmtree(output_path)
                else:
                    output_path.unlink()
            else:
                raise FileExistsError(f"{output_path} already exists")
        logger.info(f"Writing multiomics store to {output_path}")
        ds_combined.to_zarr(str(output_path), mode="w")

    return ds_combined

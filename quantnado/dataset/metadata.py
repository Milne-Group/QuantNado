from __future__ import annotations

import pandas as pd
import zarr


def extract_metadata(ds) -> pd.DataFrame:
    """Extract sample-level metadata from the Zarr-based layout."""

    attrs = getattr(ds, "attrs", {})
    sample_labels = None

    if "sample_names" in attrs:
        sample_labels = attrs.get("sample_names")
    elif hasattr(ds, "meta") and "sample_names" in ds.meta:
        stored = ds.meta["sample_names"][:]
        sample_labels = [s.decode() if isinstance(s, (bytes, bytearray)) else str(s) for s in stored]

    if sample_labels is None:
        if hasattr(ds, "sample"):
            sample_labels = getattr(ds, "sample").values.astype(str)
        else:
            # Fallback for core dataset if it has root
            root = getattr(ds, "root", None)
            if root is not None:
                sample_labels = root.attrs.get("sample_names")

    if sample_labels is None:
        raise ValueError("Unable to determine sample labels from dataset")

    metadata_df = pd.DataFrame({"sample_id": sample_labels})

    # Extract all metadata_* columns
    metadata_cols = {
        k.replace("metadata_", ""): v
        for k, v in attrs.items()
        if k.startswith("metadata_") and hasattr(v, "__len__") and len(v) == len(sample_labels)
    }
    
    for col, values in metadata_cols.items():
        # Convert empty strings to NA
        cleaned_values = [v if v != "" else pd.NA for v in values]
        metadata_df[col] = cleaned_values

    # Try to incorporate core metrics from the 'metadata' group if present
    meta_group = None
    if hasattr(ds, "get"):
        # Look for a child group named 'metadata'
        maybe_meta = ds.get("metadata")
        if isinstance(maybe_meta, zarr.Group):
            meta_group = maybe_meta

        if meta_group is not None:
            if "sample_hashes" in meta_group:
                arr = meta_group["sample_hashes"][:]
                hashes = []
                for row in arr:
                    # treat all-zero rows as missing (hash not yet computed)
                    if (row == 0).all():
                        hashes.append(pd.NA)
                    else:
                        hashes.append("".join(f"{int(b):02x}" for b in row))
                metadata_df["sample_hash"] = hashes
            if "completed" in meta_group:
                metadata_df["completed"] = meta_group["completed"][:].astype(bool)
            if "sparsity" in meta_group:
                metadata_df["sparsity"] = meta_group["sparsity"][:]

    # Reorder to keep sample_id and assay (if present) at the front
    front_cols = ["sample_id"]
    if "assay" in metadata_df.columns:
        front_cols.append("assay")
    
    remaining = [c for c in metadata_df.columns if c not in front_cols]
    metadata_df = metadata_df[front_cols + remaining]
    
    return metadata_df.set_index("sample_id")

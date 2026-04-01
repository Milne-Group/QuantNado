"""Combine multiple single-sample (or small-batch) QuantNado Zarr stores into one
multi-sample store rechunked for fast multi-sample reads.

Typical workflow::

    # Write each BAM independently (parallelisable, no locking)
    for bam in bam_files:
        BamStore.from_bam_files([bam], store_path=f"{bam}.zarr", ...)

    # Combine into one analysis-ready store
    combine_bam_stores(
        store_paths=["sample1.zarr", "sample2.zarr", ...],
        output_path="combined.zarr",
    )

The combined store has chunks shaped ``(n_samples, chunk_len)`` so any region
query touches exactly one chunk on the sample axis regardless of sample count.
"""

from __future__ import annotations

from pathlib import Path

import dask.array as da
import numpy as np
import zarr
from loguru import logger
from zarr.storage import LocalStore

from quantnado.analysis.core import QuantNadoDataset


def combine_bam_stores(
    store_paths: list[str | Path],
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> QuantNadoDataset:
    """Combine per-sample QuantNado Zarr stores into a single multi-sample store.

    The output store has chunks shaped ``(n_samples, chunk_len)`` — the entire
    sample axis fits in one chunk per position chunk — so reading all samples
    for any genomic region is a single I/O operation per position chunk.

    Parameters
    ----------
    store_paths:
        Paths to per-sample (or small-batch) QuantNado Zarr stores.
        All stores must have the same chromosomes and chromsizes.
    output_path:
        Destination path for the combined Zarr store.
    overwrite:
        If *True*, delete any existing store at *output_path* before writing.

    Returns
    -------
    QuantNadoDataset
        Opened (read-only) view of the newly written store.
    """
    output_path = Path(output_path)
    store_paths = [Path(p) for p in store_paths]

    if not store_paths:
        raise ValueError("store_paths must contain at least one path")

    if output_path.exists():
        if overwrite:
            import shutil
            shutil.rmtree(output_path)
        else:
            raise FileExistsError(
                f"{output_path} already exists. Pass overwrite=True to replace it."
            )

    logger.info(f"Opening {len(store_paths)} source stores")
    datasets = [QuantNadoDataset(p) for p in store_paths]

    # Validate compatibility
    ref = datasets[0]
    for ds in datasets[1:]:
        if set(ds.chromosomes) != set(ref.chromosomes):
            raise ValueError(
                f"Store {ds.store_path} has different chromosomes than {ref.store_path}"
            )
        if ds.chromsizes != ref.chromsizes:
            raise ValueError(
                f"Store {ds.store_path} has different chromsizes than {ref.store_path}"
            )

    all_sample_names: list[str] = []
    for ds in datasets:
        all_sample_names.extend(ds.sample_names)
    n_total = len(all_sample_names)

    chunk_len: int = ref.root.attrs.get("chunk_len") or ref.root[ref.chromosomes[0]].chunks[1]

    logger.info(
        f"Combining {n_total} samples across {len(ref.chromosomes)} chromosomes "
        f"with chunk_len={chunk_len}"
    )

    # --- Build output store ---
    out_store = LocalStore(str(output_path))
    out_root = zarr.group(store=out_store, overwrite=True, zarr_format=3)

    # Copy root attrs from first store, updating sample-level fields
    src_attrs = dict(ref.root.attrs)
    src_attrs["sample_names"] = all_sample_names
    src_attrs["n_samples"] = n_total
    # sample_names_hash is per-sample; clear it — will be rebuilt from metadata
    src_attrs.pop("sample_names_hash", None)
    out_root.attrs.update(src_attrs)

    # --- Write chromosome arrays ---
    # Collect all array keys (handles stranded _fwd/_rev pairs)
    akeys = [k for k in ref.root.keys() if k != "metadata"]

    for akey in akeys:
        logger.info(f"Writing {akey}")
        arrays = []
        for ds in datasets:
            if akey not in ds.root:
                # Per-sample store may not have stranded arrays if unstranded
                chrom_len = ref.root[akey].shape[1]
                arrays.append(
                    da.zeros(
                        (len(ds.sample_names), chrom_len),
                        dtype=np.uint32,
                        chunks=(len(ds.sample_names), chunk_len),
                    )
                )
            else:
                zarr_arr = ds.root[akey]
                arrays.append(da.from_zarr(zarr_arr, chunks=(len(ds.sample_names), chunk_len)))

        # Concatenate along sample axis → (n_total, chrom_len)
        combined = da.concatenate(arrays, axis=0)
        # Rechunk so all samples are in one chunk per position chunk
        combined = combined.rechunk((n_total, chunk_len))

        chrom_len = int(combined.shape[1])
        out_arr = out_root.create_array(
            name=akey,
            shape=(n_total, chrom_len),
            chunks=(n_total, chunk_len),
            dtype=np.uint32,
            fill_value=0,
            overwrite=True,
        )
        da.store(combined, out_arr)

    # --- Write metadata group ---
    meta_group = out_root.require_group("metadata")

    def _concat_meta(attr: str, dtype, shape_per_sample):
        arrays = []
        for ds in datasets:
            if "metadata" in ds.root and attr in ds.root["metadata"]:
                arrays.append(ds.root["metadata"][attr][:])
            else:
                n = len(ds.sample_names)
                fill = np.zeros((n, *shape_per_sample) if shape_per_sample else (n,), dtype=dtype)
                arrays.append(fill)
        return np.concatenate(arrays, axis=0)

    completed = _concat_meta("completed", bool, ())
    total_reads = _concat_meta("total_reads", np.int64, ())
    mean_read_length = _concat_meta("mean_read_length", np.float32, ())
    sparsity = _concat_meta("sparsity", np.float32, ())
    sample_hashes = _concat_meta("sample_hashes", np.uint8, (16,))

    for name, data in [
        ("completed", completed),
        ("total_reads", total_reads),
        ("mean_read_length", mean_read_length),
        ("sparsity", sparsity),
        ("sample_hashes", sample_hashes),
    ]:
        arr = meta_group.create_array(
            name=name,
            shape=data.shape,
            dtype=data.dtype,
            fill_value=0,
            overwrite=True,
        )
        arr[:] = data

    # Consolidate metadata for fast store-open
    zarr.consolidate_metadata(str(output_path))
    logger.success(f"Combined store written to {output_path}")

    return QuantNadoDataset(output_path)

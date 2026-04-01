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

import tempfile
import zipfile
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
    compression: bool = False,
    temp_dir: str | Path | None = None,
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
        Destination path for the combined Zarr store. Use a ``.zarr`` directory
        for a standard store, or ``.zarr.zip`` for a portable zip archive
        (write-once; good for archiving and sharing).
    overwrite:
        If *True*, delete any existing store at *output_path* before writing.
    compression:
        If *True* and *output_path* ends in ``.zarr.zip``, use DEFLATE
        compression inside the zip archive (smaller file, slower write).
        Ignored for plain ``.zarr`` directory stores.
    temp_dir:
        Directory to use for the intermediate ``.zarr`` store when writing a
        ``.zarr.zip``. Defaults to a system temporary directory. The
        intermediate store is deleted after the zip is created.

    Returns
    -------
    QuantNadoDataset
        Opened (read-only) view of the newly written store.
    """
    output_path = Path(output_path)
    store_paths = [Path(p) for p in store_paths]
    _zip_mode = output_path.name.endswith(".zarr.zip")

    if not store_paths:
        raise ValueError("store_paths must contain at least one path")

    if output_path.exists():
        if overwrite:
            if _zip_mode:
                output_path.unlink()
            else:
                import shutil
                shutil.rmtree(output_path)
        else:
            raise FileExistsError(
                f"{output_path} already exists. Pass overwrite=True to replace it."
            )

    # For zip output, write to a temp directory first then archive it,
    # because zarr v3 ZipStore doesn't support incremental random-write access.
    if _zip_mode:
        if temp_dir is not None:
            _tmp_dir = str(temp_dir)
            Path(_tmp_dir).mkdir(parents=True, exist_ok=True)
        else:
            _tmp_dir = tempfile.mkdtemp(prefix="quantnado_combine_")
        _write_path = Path(_tmp_dir) / "store"
    else:
        _tmp_dir = None
        _managed_tmp = False
        _write_path = output_path

    logger.info(f"Opening {len(store_paths)} source stores")
    datasets = [QuantNadoDataset(p) for p in store_paths]

    # Validate compatibility — stores may have different chromosome subsets
    # (e.g. when created with filter_chromosomes=True), so we only require that
    # shared chromosomes have consistent sizes.
    ref = datasets[0]
    all_chromsizes: dict[str, int] = dict(ref.chromsizes)
    for ds in datasets[1:]:
        for chrom, size in ds.chromsizes.items():
            if chrom in all_chromsizes and all_chromsizes[chrom] != size:
                raise ValueError(
                    f"Store {ds.path}: chromosome {chrom!r} has size {size} "
                    f"but {ref.path} has size {all_chromsizes[chrom]}"
                )
            all_chromsizes[chrom] = size
        diff = set(ds.chromosomes) ^ set(ref.chromosomes)
        if diff:
            logger.warning(
                f"Store {ds.path} has different chromosomes than {ref.path} "
                f"(symmetric diff: {diff}). Missing chromosomes will be filled with zeros."
            )

    all_sample_names: list[str] = []
    for ds in datasets:
        all_sample_names.extend(ds.sample_names)
    n_total = len(all_sample_names)

    all_chroms = sorted(all_chromsizes.keys())

    # Detect store layout: new (coverage/ group, shape (chrom_len, n_samples)) vs
    # old (chromosomes at root, shape (n_samples, chrom_len))
    _new_layout = "coverage" in ref.root

    if ref.root.attrs.get("chunk_len"):
        chunk_len: int = int(ref.root.attrs["chunk_len"])
    elif _new_layout:
        chunk_len = ref.root["coverage"][ref.chromosomes[0]].chunks[0]
    else:
        chunk_len = ref.root[ref.chromosomes[0]].chunks[1]

    logger.info(
        f"Combining {n_total} samples across {len(all_chroms)} chromosomes "
        f"with chunk_len={chunk_len} ({'new' if _new_layout else 'old'} layout)"
    )

    # --- Build output store ---
    out_store = LocalStore(str(_write_path))
    out_root = zarr.group(store=out_store, overwrite=True, zarr_format=3)

    # Copy root attrs from first store, updating sample-level fields
    src_attrs = dict(ref.root.attrs)
    src_attrs["sample_names"] = all_sample_names
    src_attrs["n_samples"] = n_total
    src_attrs["chromsizes"] = all_chromsizes
    # sample_names_hash is per-sample; clear it — will be rebuilt from metadata
    src_attrs.pop("sample_names_hash", None)
    out_root.attrs.update(src_attrs)

    # --- Write chromosome arrays ---
    if _new_layout:
        # New layout: chromosomes live in coverage/ (and coverage_fwd/, coverage_rev/) groups.
        # Array shape is (chrom_len, n_samples); concatenate along axis=1.
        cov_groups = sorted(
            {gkey for ds in datasets for gkey in ds.root.keys()
             if gkey != "metadata" and isinstance(ds.root[gkey], zarr.Group)}
        )
        for gkey in cov_groups:
            out_cov_group = out_root.require_group(gkey)
            all_chroms_in_group = sorted(
                {chrom for ds in datasets if gkey in ds.root for chrom in ds.root[gkey].keys()}
            )
            for chrom in all_chroms_in_group:
                logger.info(f"Writing {gkey}/{chrom}")
                chrom_len = all_chromsizes[chrom]
                arrays = []
                for ds in datasets:
                    n = len(ds.sample_names)
                    if gkey not in ds.root or chrom not in ds.root[gkey]:
                        arrays.append(da.zeros((chrom_len, n), dtype=np.uint32, chunks=(chunk_len, n)))
                    else:
                        arrays.append(da.from_array(ds.root[gkey][chrom], chunks=(chunk_len, n)))
                combined = da.concatenate(arrays, axis=1).rechunk((chunk_len, n_total))
                out_arr = out_cov_group.create_array(
                    name=chrom,
                    shape=(chrom_len, n_total),
                    chunks=(chunk_len, n_total),
                    dtype=np.uint32,
                    fill_value=0,
                    overwrite=True,
                )
                da.store(combined, out_arr)
    else:
        # Old layout: chromosomes at root, shape (n_samples, chrom_len); concatenate along axis=0.
        akeys = sorted(
            {k for ds in datasets for k in ds.root.keys() if k != "metadata"}
        )
        for akey in akeys:
            logger.info(f"Writing {akey}")
            chrom_len = next(ds.root[akey].shape[1] for ds in datasets if akey in ds.root)
            arrays = []
            for ds in datasets:
                n = len(ds.sample_names)
                if akey not in ds.root:
                    arrays.append(da.zeros((n, chrom_len), dtype=np.uint32, chunks=(n, chunk_len)))
                else:
                    arrays.append(da.from_array(ds.root[akey], chunks=(n, chunk_len)))
            combined = da.concatenate(arrays, axis=0).rechunk((n_total, chunk_len))
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
    zarr.consolidate_metadata(str(_write_path))

    # If zip output, archive the temp directory then clean up
    if _zip_mode:
        zip_compression = zipfile.ZIP_DEFLATED if compression else zipfile.ZIP_STORED
        logger.info(f"Archiving to {output_path}")
        with zipfile.ZipFile(output_path, mode="w", compression=zip_compression) as zf:
            for file in sorted(_write_path.rglob("*")):
                zf.write(file, file.relative_to(_write_path))
        import shutil as _shutil
        _shutil.rmtree(_tmp_dir)
        logger.success(f"Combined store written to {output_path}")
        return QuantNadoDataset(output_path)

    logger.success(f"Combined store written to {output_path}")
    return QuantNadoDataset(output_path)

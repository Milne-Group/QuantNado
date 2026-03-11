"""Normalisation of genomic signal from QuantNado stores.

Works on the outputs of ``reduce()``, ``extract()``, and ``count_features()``.

Supported methods
-----------------
cpm   Counts Per Million — divide by library size / 1e6. Works for all data types.
rpkm  Reads Per Kilobase per Million — CPM / feature_length_kb. Requires feature lengths
      (inferred from range_length coord for xr.Dataset, or passed via feature_lengths).
tpm   Transcripts Per Million — length-normalise first, then scale so column sums = 1e6.
      Self-normalising; does not require library sizes. pd.DataFrame only.

Examples
--------
>>> signal = ds.reduce(intervals_path="promoters.bed")
>>> normalised = normalise(signal, ds, method="cpm")

>>> binned = ds.extract(feature_type="transcript", gtf_path="genes.gtf", bin_size=50)
>>> normalised = normalise(binned, ds, method="cpm")

>>> counts, features = ds.count_features(gtf_file="genes.gtf")
>>> rpkm = normalise(counts, ds, method="rpkm", feature_lengths=features["range_length"])
>>> tpm  = normalise(counts, ds, method="tpm",  feature_lengths=features["range_length"])
"""

from __future__ import annotations
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da

from loguru import logger


# ---------------------------------------------------------------------------
# Library size retrieval
# ---------------------------------------------------------------------------

def get_library_sizes(dataset) -> pd.Series:
    """
    Return total mapped reads per sample as a ``pd.Series`` indexed by sample name.

    Reads ``metadata/total_reads`` from the underlying Zarr store (written during
    BAM construction).  Raises ``RuntimeError`` for older stores that pre-date this
    field; in that case pass ``library_sizes`` explicitly to :func:`normalise`.

    Parameters
    ----------
    dataset : QuantNado | BamStore | MultiomicsStore
        Any QuantNado object with a coverage store attached.

    Returns
    -------
    pd.Series
        Integer read counts, indexed by sample name.

    Raises
    ------
    RuntimeError
        If the store does not contain ``total_reads`` metadata.
    """
    bam_store = _resolve_bam_store(dataset)

    meta = bam_store.meta
    if "total_reads" not in meta:
        raise RuntimeError(
            "This store does not contain 'total_reads' metadata (it was built before "
            "library-size tracking was added). Either rebuild the store from BAM files, "
            "or pass library_sizes explicitly to normalise()."
        )

    reads = meta["total_reads"][:].astype(np.int64)
    completed = bam_store.completed_mask
    reads_float = reads.astype(float)
    reads_float[~completed] = np.nan
    return pd.Series(reads_float, index=bam_store.sample_names, name="library_size")


def _resolve_bam_store(dataset):
    """Extract a BamStore from any QuantNado-family object."""
    # QuantNado API facade
    if hasattr(dataset, "coverage") and dataset.coverage is not None:
        return dataset.coverage
    # BamStore directly (has meta + sample_names)
    if hasattr(dataset, "meta") and hasattr(dataset, "sample_names"):
        return dataset
    raise TypeError(
        f"Cannot resolve a BamStore from {type(dataset).__name__}. "
        "Pass a QuantNado, BamStore, or MultiomicsStore."
    )


# ---------------------------------------------------------------------------
# Core normalise dispatcher
# ---------------------------------------------------------------------------

def normalise(
    data: xr.Dataset | xr.DataArray | pd.DataFrame,
    dataset=None,
    *,
    method: str = "cpm",
    library_sizes: pd.Series | dict | None = None,
    feature_lengths: pd.Series | np.ndarray | None = None,
) -> xr.Dataset | xr.DataArray | pd.DataFrame:
    """
    Normalise coverage signal or feature counts.

    Parameters
    ----------
    data : xr.Dataset | xr.DataArray | pd.DataFrame
        Output of ``reduce()``, ``extract()``, or ``count_features()``.
    dataset : QuantNado | BamStore | MultiomicsStore, optional
        Source dataset used to look up library sizes automatically.
        Not required when ``library_sizes`` is provided explicitly, or for
        ``method="tpm"`` (which is self-normalising).
    method : {"cpm", "rpkm", "tpm"}, default "cpm"
        Normalisation method:
        - ``"cpm"``  — Counts Per Million (all data types)
        - ``"rpkm"`` — RPKM (xr.Dataset and pd.DataFrame; needs feature_lengths)
        - ``"tpm"``  — TPM (pd.DataFrame only; self-normalising, no library sizes needed)
    library_sizes : pd.Series or dict, optional
        Total mapped reads per sample, indexed by sample name.
        Overrides automatic lookup from ``dataset``.
    feature_lengths : pd.Series or array-like, optional
        Feature lengths **in base-pairs**, aligned to data rows.
        Required for ``method="rpkm"`` and ``method="tpm"`` on pd.DataFrames.
        Inferred automatically from the ``range_length`` coord for xr.Dataset.

    Returns
    -------
    Same type as ``data``, with normalised values (new object, data not modified in-place).

    Raises
    ------
    ValueError
        Unknown method, or required inputs (feature_lengths / library_sizes) are missing.
    RuntimeError
        Library sizes cannot be resolved from the store.

    Examples
    --------
    >>> # Normalise a reduce() output for plotting
    >>> signal = ds.reduce(intervals_path="promoters.bed")
    >>> cpm_signal = normalise(signal, ds, method="cpm")

    >>> # Normalise a binned extract() output (e.g. for metaplots)
    >>> binned = ds.extract(feature_type="transcript", gtf_path="genes.gtf", bin_size=50)
    >>> cpm_binned = normalise(binned, ds, method="cpm")

    >>> # Normalise count matrix for DESeq2 pre-inspection or plotting
    >>> counts, features = ds.count_features(gtf_file="genes.gtf")
    >>> tpm = normalise(counts, ds, method="tpm", feature_lengths=features["range_length"])
    """
    method = method.lower()
    if method not in {"cpm", "rpkm", "tpm"}:
        raise ValueError(
            f"Unknown normalisation method {method!r}. Choose 'cpm', 'rpkm', or 'tpm'."
        )

    # Resolve library sizes (not needed for TPM)
    lib_sizes: pd.Series | None = None
    if method in {"cpm", "rpkm"}:
        lib_sizes = _resolve_library_sizes(dataset, library_sizes)

    if isinstance(data, pd.DataFrame):
        return _normalise_dataframe(data, method=method, lib_sizes=lib_sizes, feature_lengths=feature_lengths)
    if isinstance(data, xr.Dataset):
        return _normalise_xr_dataset(data, method=method, lib_sizes=lib_sizes, feature_lengths=feature_lengths)
    if isinstance(data, xr.DataArray):
        return _normalise_xr_dataarray(data, method=method, lib_sizes=lib_sizes)

    raise TypeError(
        f"data must be xr.Dataset, xr.DataArray, or pd.DataFrame, got {type(data).__name__}"
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_library_sizes(dataset, library_sizes) -> pd.Series:
    if library_sizes is not None:
        if isinstance(library_sizes, dict):
            library_sizes = pd.Series(library_sizes, name="library_size")
        return library_sizes.astype(float)

    if dataset is None:
        raise ValueError(
            "Provide either 'dataset' (to auto-read library sizes) or 'library_sizes' explicitly."
        )
    return get_library_sizes(dataset)


def _scale_per_sample(lib_sizes: pd.Series, sample_labels: list[str]) -> np.ndarray:
    """Return CPM scale factors (library_size / 1e6) aligned to sample_labels."""
    missing = set(sample_labels) - set(lib_sizes.index)
    if missing:
        raise ValueError(f"Library sizes missing for samples: {sorted(missing)}")
    return np.array([lib_sizes[s] for s in sample_labels], dtype=np.float64) / 1e6


# --- pd.DataFrame -----------------------------------------------------------

def _normalise_dataframe(
    data: pd.DataFrame,
    method: str,
    lib_sizes: pd.Series | None,
    feature_lengths: pd.Series | np.ndarray | None,
) -> pd.DataFrame:
    sample_labels = list(data.columns)

    if method == "tpm":
        if feature_lengths is None:
            raise ValueError("feature_lengths (in bp) is required for TPM normalisation.")
        lengths_kb = np.asarray(feature_lengths, dtype=float) / 1000.0
        if lengths_kb.shape[0] != data.shape[0]:
            raise ValueError(
                f"feature_lengths length ({lengths_kb.shape[0]}) does not match "
                f"data rows ({data.shape[0]})."
            )
        rpk = data.div(lengths_kb, axis=0)
        scaling = rpk.sum(axis=0) / 1e6
        result = rpk.div(scaling, axis=1)
        logger.info("Normalised DataFrame to TPM.")
        return result

    scale = _scale_per_sample(lib_sizes, sample_labels)
    scale_series = pd.Series(scale, index=sample_labels)

    if method == "cpm":
        result = data.div(scale_series, axis=1)
        logger.info("Normalised DataFrame to CPM.")
        return result

    # rpkm
    if feature_lengths is None:
        raise ValueError("feature_lengths (in bp) is required for RPKM normalisation.")
    lengths_kb = np.asarray(feature_lengths, dtype=float) / 1000.0
    if lengths_kb.shape[0] != data.shape[0]:
        raise ValueError(
            f"feature_lengths length ({lengths_kb.shape[0]}) does not match "
            f"data rows ({data.shape[0]})."
        )
    result = data.div(scale_series, axis=1).div(lengths_kb, axis=0)
    logger.info("Normalised DataFrame to RPKM.")
    return result


# --- xr.Dataset (output of reduce()) ----------------------------------------

def _normalise_xr_dataset(
    data: xr.Dataset,
    method: str,
    lib_sizes: pd.Series,
    feature_lengths: pd.Series | np.ndarray | None = None,
) -> xr.Dataset:
    sample_labels = list(data["sample"].values)
    scale = _scale_per_sample(lib_sizes, sample_labels)  # (n_samples,)
    scale_da = da.from_array(scale, chunks=-1)[np.newaxis, :]  # (1, n_samples)

    def _to_float_dask(var_name: str) -> da.Array:
        arr = data[var_name].data
        if not isinstance(arr, da.Array):
            arr = da.from_array(arr)
        if not np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.float32)
        return arr

    vars_to_norm = [v for v in data.data_vars if v != "count"]

    normed: dict[str, tuple] = {}
    for v in vars_to_norm:
        normed[v] = (data[v].dims, _to_float_dask(v) / scale_da)

    if method == "rpkm":
        # Try range_length coord first, fall back to feature_lengths arg
        if "range_length" in data.coords:
            lengths_bp = data["range_length"].values.astype(float)
        elif feature_lengths is not None:
            lengths_bp = np.asarray(feature_lengths, dtype=float)
        else:
            raise ValueError(
                "RPKM requires feature lengths. Pass feature_lengths= or use reduce() output "
                "(which includes a 'range_length' coordinate)."
            )
        lengths_kb = da.from_array(lengths_bp / 1000.0, chunks=-1)[:, np.newaxis]  # (n_ranges, 1)
        normed = {v: (dims, arr / lengths_kb) for v, (dims, arr) in normed.items()}

    # Preserve count unchanged
    if "count" in data.data_vars:
        normed["count"] = (data["count"].dims, data["count"].data)

    result = xr.Dataset(normed, coords=data.coords, attrs={**data.attrs, "normalised": method})
    logger.info(f"Normalised xr.Dataset to {method.upper()} ({len(vars_to_norm)} variables).")
    return result


# --- xr.DataArray (output of extract()) -------------------------------------

def _normalise_xr_dataarray(
    data: xr.DataArray,
    method: str,
    lib_sizes: pd.Series,
) -> xr.DataArray:
    """
    Normalise a per-position / binned xr.DataArray across the sample axis.

    Assumes the last axis of `data` is the sample axis (signal layout:
    (interval, position/bin, sample)). `method` is one of {'cpm','rpkm'}.
    `lib_sizes` is a pd.Series indexed by sample label (or compatible with
    _scale_per_sample implementation).

    Returns a DataArray with the same dims/coords but normalized values and
    `attrs["normalised"] = method`.
    """

    if method == "tpm":
        raise ValueError(
            "TPM is not defined for per-position signal (extract() output). "
            "Use 'cpm' or 'rpkm' for signal tracks, or normalise a count matrix instead."
        )

    # sample labels and per-sample scale (function expected to return 1D array-like)
    sample_labels = list(data["sample"].values)
    scale = _scale_per_sample(lib_sizes, sample_labels)  # expected shape: (n_samples,)

    arr = data.data
    if not isinstance(arr, da.Array):
        arr = da.from_array(arr)
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float32)

    # sanity checks
    if "sample" not in data.dims:
        raise ValueError("Input DataArray must have a 'sample' dimension.")
    n_samples = int(data.sizes["sample"])
    scale = np.asarray(scale)
    if scale.ndim != 1:
        raise ValueError(f"Scale must be 1D, got shape {scale.shape}.")
    if scale.shape[0] != n_samples:
        raise ValueError(f"Scale length ({scale.shape[0]}) does not match number of samples ({n_samples}).")

    # Build reshape shape so scale broadcasts over the last axis of `arr`,
    # regardless of number of leading axes.
    # Example: arr.ndim == 3 -> reshape_shape = (1, 1, -1)
    reshape_shape = tuple([1] * (arr.ndim - 1) + [-1])

    # create dask array for scale with a single chunk spanning the sample axis
    scale_da = da.from_array(scale, chunks=-1).reshape(reshape_shape)

    # normalise across last axis
    normed_arr = arr / scale_da

    if method == "rpkm":
        # Infer bin size in kb from the position/bin coordinate spacing.
        # For 50bp bins: spacing = 50 → bin_size_kb = 0.05
        # For per-position signal (no binning): spacing = 1bp → bin_size_kb = 0.001
        pos_dim: str | None = next(
            (d for d in data.dims if d in ("relative_position", "bin", "position")), None
        )
        if pos_dim is not None and data.sizes[pos_dim] > 1:
            coords = data.coords[pos_dim].values
            # coords might be non-numeric (but normally numeric); cast safely
            try:
                spacing = float(coords[1] - coords[0])
                bin_size_kb = abs(spacing) / 1000.0
            except Exception:
                # fallback: assume 1 bp if we cannot compute spacing
                bin_size_kb = 1.0 / 1000.0
        else:
            bin_size_kb = 1.0 / 1000.0  # assume 1bp

        normed_arr = normed_arr / bin_size_kb

    result = xr.DataArray(
        normed_arr,
        dims=data.dims,
        coords=data.coords,
        attrs={**data.attrs, "normalised": method},
    )
    logger.info(f"Normalised xr.DataArray to {method.upper()}.")
    return result

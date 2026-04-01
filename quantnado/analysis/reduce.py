"""Efficient reduction of per-chromosome signal data over genomic ranges."""

from __future__ import annotations

import numpy as np
import pandas as pd
import dask.array as da
import xarray as xr
import pyranges1 as pr
from typing import TYPE_CHECKING, Iterable
from loguru import logger

from ..dataset.enums import ReductionMethod, FeatureType, AnchorPoint
from .features import (
	extract_feature_ranges,
	extract_promoters,
	load_gtf,
)

if TYPE_CHECKING:
	pass


def _ensure_dask_2d(data: xr.DataArray | np.ndarray | da.Array) -> da.Array:
	"""Return a 2D dask array (positions x samples) for reduction."""
	arr = data
	if isinstance(arr, xr.DataArray):
		arr = arr.data
	if not isinstance(arr, da.Array):
		arr = da.from_array(arr)
	if arr.ndim != 2:
		raise ValueError("expected a 2D array (positions x samples)")
	arr = arr.rechunk({0: "auto"})
	return arr


def _log_chromosome_overlap(gtf_contigs: set[str], dataset_contigs: set[str], feature_source: str = "GTF") -> int:
	"""
	Log and return the number of shared chromosomes between GTF and dataset.

	Parameters
	----------
	gtf_contigs : set[str]
		Chromosome names from the GTF file.
	dataset_contigs : set[str]
		Chromosome names available in the dataset.
	feature_source : str
		Description of where features came from (default: "GTF").

	Returns
	-------
	int
		Number of shared chromosomes.
	"""
	shared = gtf_contigs & dataset_contigs
	gtf_only = gtf_contigs - dataset_contigs
	dataset_only = dataset_contigs - gtf_contigs

	logger.info(
		f"Chromosome compatibility check: {len(shared)} shared chromosomes out of "
		f"{len(gtf_contigs)} in {feature_source} and {len(dataset_contigs)} in dataset"
	)

	if shared:
		logger.debug(f"Shared chromosomes: {sorted(shared)}")

	if gtf_only:
		logger.warning(
			f"{feature_source} contains {len(gtf_only)} chromosome(s) not in dataset: "
			f"{sorted(gtf_only)}"
		)

	if dataset_only:
		logger.debug(
			f"Dataset contains {len(dataset_only)} chromosome(s) not in {feature_source}: "
			f"{sorted(dataset_only)}"
		)

	return len(shared)


def _resolve_ranges(
	ranges_df: pd.DataFrame | pr.PyRanges | None,
	intervals_path: str | None,
	feature_type: FeatureType | str | None,
	gtf_path: str | Iterable[str] | None,
	start_col: str,
	end_col: str,
	contig_col: str,
) -> tuple[pd.DataFrame, str, str, str]:
	"""
	Resolve genomic ranges from one of three input modes.

	Parameters
	----------
	ranges_df : pd.DataFrame or pr.PyRanges, optional
		Pre-parsed ranges.
	intervals_path : str, optional
		Path to a BED or GTF/GFF file.
	feature_type : FeatureType or str, optional
		Predefined feature type; requires gtf_path.
	gtf_path : str or Iterable[str], optional
		Path(s) to GTF file(s).
	start_col, end_col, contig_col : str
		Default column names (may be overridden for PyRanges output).

	Returns
	-------
	tuple[pd.DataFrame, str, str, str]
		(ranges_df, start_col, end_col, contig_col) with normalised column names.

	Raises
	------
	TypeError
		If none of the three input modes is provided.
	ValueError
		If the file format is unsupported or required columns are missing.
	"""
	if feature_type is not None and gtf_path is not None:
		if isinstance(feature_type, str):
			feature_type = FeatureType(feature_type)

		gtf_df = load_gtf(gtf_path)

		if feature_type == FeatureType.PROMOTER:
			ranges_df = extract_promoters(gtf_df, anchor_feature=FeatureType.GENE)
		else:
			ranges_df = extract_feature_ranges(gtf_df, feature_type=feature_type)

		if isinstance(ranges_df, pr.PyRanges):
			ranges_df = pd.DataFrame(ranges_df)
		start_col, end_col, contig_col = "Start", "End", "Chromosome"

	elif intervals_path is not None:
		if intervals_path.endswith((".bed", ".bed.gz")):
			ranges_df = pd.DataFrame(pr.read_bed(intervals_path))
		elif intervals_path.endswith((".gtf", ".gtf.gz", ".gff", ".gff3", ".gff.gz")):
			ranges_df = pd.DataFrame(pr.read_gtf(intervals_path))
		else:
			raise ValueError("Unsupported intervals file format. Use .bed or .gtf/.gff extensions.")
		start_col, end_col, contig_col = "Start", "End", "Chromosome"

	elif ranges_df is not None:
		if isinstance(ranges_df, pr.PyRanges):
			ranges_df = pd.DataFrame(ranges_df)
	else:
		raise TypeError(
			"Must provide one of: ranges_df, intervals_path, or (feature_type + gtf_path)"
		)

	# Normalise column names for common variants
	col_map: dict[str, str] = {}
	if contig_col not in ranges_df.columns:
		for col in ["contig", "seqname", "Chromosome", "chr"]:
			if col in ranges_df.columns:
				col_map[col] = contig_col
				break
	if start_col not in ranges_df.columns:
		for col in ["start", "Start"]:
			if col in ranges_df.columns:
				col_map[col] = start_col
				break
	if end_col not in ranges_df.columns:
		for col in ["end", "End"]:
			if col in ranges_df.columns:
				col_map[col] = end_col
				break
	if col_map:
		ranges_df = ranges_df.rename(columns=col_map)

	required = {contig_col, start_col, end_col}
	if not required.issubset(ranges_df.columns):
		missing = required - set(ranges_df.columns)
		raise ValueError(f"ranges_df must include columns {missing}. Found: {set(ranges_df.columns)}")

	return ranges_df, start_col, end_col, contig_col


def _select_samples(
	dataset,
	include_incomplete: bool,
	sample_indices: np.ndarray | None,
) -> tuple[np.ndarray, list[str], object]:
	"""
	Resolve which samples to use and return their indices, labels and the zarr root.

	Parameters
	----------
	dataset : BamStore or zarr.Group
		Source dataset.
	include_incomplete : bool
		If True, include samples not marked complete.
	sample_indices : np.ndarray, optional
		Explicit indices (overrides completion filter).

	Returns
	-------
	tuple[np.ndarray, list[str], zarr.Group]
		(sample_indices, sample_labels, root)

	Raises
	------
	ValueError
		If no samples are selected or no chromosome data is found.
	"""
	root = dataset.root if hasattr(dataset, "root") else dataset
	meta = root.get("metadata") if hasattr(root, "get") else None

	sample_names = getattr(dataset, "sample_names", root.attrs.get("sample_names", None))
	completed_mask = getattr(dataset, "completed_mask", None)
	if completed_mask is None and meta is not None and "completed" in meta:
		completed_mask = meta["completed"][:].astype(bool)

	first_chrom = next((k for k in root.keys() if k != "metadata"), None)
	if first_chrom is None:
		raise ValueError("No chromosome data found in dataset")
	total_samples = root[first_chrom].shape[0]

	if completed_mask is None:
		completed_mask = np.ones(total_samples, dtype=bool)

	if sample_indices is None:
		sample_indices = (
			np.arange(total_samples)[completed_mask]
			if not include_incomplete
			else np.arange(total_samples)
		)
	else:
		sample_indices = np.asarray(sample_indices, dtype=np.int64)
		# Apply completion filter to explicitly provided sample indices
		if not include_incomplete:
			sample_indices = sample_indices[completed_mask[sample_indices]]

	if sample_indices.size == 0:
		raise ValueError("No samples selected")

	all_labels = sample_names if sample_names is not None else [str(i) for i in range(total_samples)]
	sample_labels = [all_labels[i] for i in sample_indices]

	return sample_indices, sample_labels, root


def _reduce_byranges_prefix(
	row_starts: np.ndarray,
	row_ends: np.ndarray,
	data: xr.DataArray | np.ndarray | da.Array,
	*,
	min_count: int = 1,
	device: str = "cpu",
) -> dict[str, da.Array]:
	"""
	Reduce ranges via prefix sums (efficient for large range sets).

	Assumes row indices are 0-based, end-exclusive, and len(row_starts)==len(row_ends).
	Works for both float (NaN-aware) and integer data.

	Parameters
	----------
	row_starts : np.ndarray
		Start indices for each range.
	row_ends : np.ndarray
		End indices for each range.
	data : xr.DataArray | np.ndarray | da.Array
		2D array (positions x samples).
	min_count : int
		Minimum count threshold for valid mean values.

	Returns
	-------
	dict[str, da.Array]
		Dictionary with keys {'sum', 'count', 'mean'} containing reduced values.
	"""

	if row_starts.shape != row_ends.shape:
		raise ValueError("row_starts and row_ends must have the same shape")

	starts = np.asarray(row_starts, dtype=np.int64)
	ends = np.asarray(row_ends, dtype=np.int64)

	arr = _ensure_dask_2d(data)

	# NaN-aware reductions require a floating dtype.
	if not np.issubdtype(arr.dtype, np.floating):
		arr = arr.astype(np.float32)

	if device != "cpu":
		try:
			import torch
			arr_np = arr.compute() if isinstance(arr, da.Array) else np.asarray(arr)
			t = torch.from_numpy(arr_np.astype(np.float32)).to(device)
			zero_row = torch.zeros((1, t.shape[1]), dtype=t.dtype, device=device)
			nan_mask = torch.isnan(t)
			values_t = torch.nan_to_num(t, nan=0.0)
			mask_t = (~nan_mask).to(torch.int64)
			sum_pref_t = torch.cat([zero_row, torch.cumsum(values_t, dim=0)], dim=0)
			count_pref_t = torch.cat(
				[torch.zeros((1, t.shape[1]), dtype=torch.int64, device=device), torch.cumsum(mask_t, dim=0)],
				dim=0,
			)
			sums_np = (sum_pref_t[ends] - sum_pref_t[starts]).cpu().numpy()
			counts_np = (count_pref_t[ends] - count_pref_t[starts]).cpu().numpy()
			with np.errstate(invalid="ignore"):
				means_np = np.where(counts_np >= min_count, sums_np / counts_np, np.nan)
			sums_da = da.from_array(sums_np, chunks=sums_np.shape)
			counts_da = da.from_array(counts_np.astype(np.int64), chunks=counts_np.shape)
			means_da = da.from_array(means_np, chunks=means_np.shape)
			return {"sum": sums_da, "count": counts_da, "mean": means_da}
		except Exception:
			pass

	is_float = np.issubdtype(arr.dtype, np.floating)
	# Prepare value and mask arrays for prefix sums.
	values = da.nan_to_num(arr) if is_float else arr
	mask = (~da.isnan(arr)) if is_float else da.ones_like(arr, dtype=np.int64)

	# Prefix sums along positions (axis 0). Prepend a zero row so end can equal len.
	sum_pref = da.concatenate(
		[da.zeros((1, arr.shape[1]), dtype=values.dtype), da.cumsum(values, axis=0)],
		axis=0,
	)
	count_pref = da.concatenate(
		[da.zeros((1, arr.shape[1]), dtype=np.int64), da.cumsum(mask, axis=0)],
		axis=0,
	)

	# Gather prefix rows for starts/ends; da.take handles dask-aware indexing.
	sum_start = da.take(sum_pref, starts, axis=0)
	sum_end = da.take(sum_pref, ends, axis=0)
	count_start = da.take(count_pref, starts, axis=0)
	count_end = da.take(count_pref, ends, axis=0)

	sums = sum_end - sum_start
	counts = count_end - count_start

	# Mean with minimum count threshold.
	means = da.ma.filled(
		sums / da.ma.masked_less(counts, min_count),
		np.nan,
	)

	return {"sum": sums, "count": counts.astype(np.int64), "mean": means}


def _reduce_ranges_vectorized(
	data: xr.DataArray | np.ndarray | da.Array,
	starts: np.ndarray,
	ends: np.ndarray,
	reduction: str,
	device: str = "cpu",
) -> da.Array:
	"""
	Reduce ranges using vectorized operations (max/min/median).

	Faster than prefix sums for small range sets; slower for large ones.

	Parameters
	----------
	data : xr.DataArray | np.ndarray | da.Array
		2D array (positions x samples).
	starts : np.ndarray
		Start indices for each range.
	ends : np.ndarray
		End indices for each range.
	reduction : str
		Reduction method: 'max', 'min', 'median', 'sum'.

	Returns
	-------
	da.Array
		Reduced data with shape (n_ranges, n_samples).
	"""

	arr = _ensure_dask_2d(data)

	starts = np.asarray(starts, dtype=np.int64)
	ends = np.asarray(ends, dtype=np.int64)
	lengths = ends - starts
	if lengths.size == 0:
		return da.empty((0, arr.shape[1]), dtype=arr.dtype)
	if np.any(lengths <= 0):
		raise ValueError("starts/ends must define non-empty ranges")

	max_len = int(lengths.max())
	arr_len = int(arr.shape[0])

	# Pad right so gathering doesn't go OOB for short ranges near contig end.
	pad_right = int(max(0, int(starts.max() + max_len) - arr_len))
	if pad_right:
		arr = da.pad(arr, ((0, pad_right), (0, 0)), mode="constant", constant_values=np.nan)

	offsets = np.arange(max_len, dtype=np.int64)
	indices = starts[:, None] + offsets[None, :]
	flat = indices.reshape(-1)
	gathered = da.take(arr, flat, axis=0).reshape((starts.shape[0], max_len, arr.shape[1]))

	mask = offsets[None, :] < lengths[:, None]
	mask_da = da.from_array(mask, chunks=(min(mask.shape[0], 256), mask.shape[1]))
	masked = da.where(mask_da[:, :, None], gathered, np.nan)

	if device != "cpu":
		try:
			import torch
			gathered_np = masked.compute() if isinstance(masked, da.Array) else np.asarray(masked)
			g_t = torch.from_numpy(gathered_np.astype(np.float32)).to(device)
			if reduction == "max":
				result_np = torch.amax(g_t, dim=1).cpu().numpy()
			elif reduction == "min":
				result_np = torch.amin(g_t, dim=1).cpu().numpy()
			elif reduction == "median":
				result_np = torch.nanmedian(g_t, dim=1).values.cpu().numpy()
			elif reduction == "sum":
				result_np = torch.nansum(g_t, dim=1).cpu().numpy()
			else:
				raise ValueError(f"Unknown reduction: {reduction}")
			return da.from_array(result_np, chunks=result_np.shape)
		except Exception:
			pass

	if reduction == "max":
		return da.nanmax(masked, axis=1)
	if reduction == "min":
		return da.nanmin(masked, axis=1)
	if reduction == "median":
		return da.nanpercentile(masked, 50, axis=1)
	if reduction == "sum":
		return da.nansum(masked, axis=1)

	raise ValueError(f"Unknown reduction: {reduction}")


def _bin_array(arr: da.Array, bin_size: int, agg_func: str = "mean", axis: int = 0) -> da.Array:
	"""
	Bin a dask array along a specified axis into fixed-size bins.

	Parameters
	----------
	arr : da.Array
		Input dask array (2D or 3D).
	bin_size : int
		Size of each bin along the binned axis.
	agg_func : str
		Aggregation function: 'mean', 'sum', 'max', 'min', 'median'.
	axis : int, default 0
		Axis to bin along.

	Returns
	-------
	da.Array
		Binned array with NaN-aware aggregation.
	"""
	n_pos = int(arr.shape[axis])
	if n_pos == 0:
		new_shape = list(arr.shape)
		new_shape[axis] = 0
		return da.empty(tuple(new_shape), dtype=arr.dtype)

	n_bins = n_pos // bin_size
	if n_bins == 0:
		new_shape = list(arr.shape)
		new_shape[axis] = 0
		return da.empty(tuple(new_shape), dtype=arr.dtype)

	# Slice to exact multiple of bin_size
	slc = [slice(None)] * arr.ndim
	slc[axis] = slice(0, n_bins * bin_size)
	trimmed = arr[tuple(slc)]

	# Reshape to (..., n_bins, bin_size, ...)
	new_shape = trimmed.shape[:axis] + (n_bins, bin_size) + trimmed.shape[axis+1:]
	reshaped = trimmed.reshape(new_shape)

	# Aggregation axis is axis + 1 in the reshaped array
	agg_axis = axis + 1

	if agg_func == "mean":
		return da.nanmean(reshaped, axis=agg_axis)
	elif agg_func == "sum":
		return da.nansum(reshaped, axis=agg_axis)
	elif agg_func == "max":
		return da.nanmax(reshaped, axis=agg_axis)
	elif agg_func == "min":
		return da.nanmin(reshaped, axis=agg_axis)
	elif agg_func == "median":
		return da.nanpercentile(reshaped, 50, axis=agg_axis)

	raise ValueError(f"Unknown aggregation function: {agg_func}")


def _estimate_interval_batch_size(
	target_bases: int,
	n_samples: int,
	*,
	stranded: bool = False,
	target_bytes: int = 64 * 1024**2,
	max_intervals: int = 512,
	min_intervals: int = 16,
) -> int:
	"""Choose a conservative interval batch size for extraction gathers."""
	if target_bases <= 0:
		return max_intervals

	arrays_per_batch = 2 if stranded else 1
	bytes_per_value = np.dtype(np.float32).itemsize
	bytes_per_base = 8 + arrays_per_batch * max(1, n_samples) * bytes_per_value
	batch_size = target_bytes // max(1, target_bases * bytes_per_base)
	batch_size = max(1, int(batch_size))
	return int(min(max(batch_size, min_intervals), max_intervals))


def _iter_interval_slices(n_intervals: int, batch_size: int):
	"""Yield contiguous slices over the interval axis."""
	for start in range(0, n_intervals, batch_size):
		yield slice(start, min(n_intervals, start + batch_size))


def _iter_interval_slices_by_span(
	start_positions: np.ndarray,
	width: int,
	batch_size: int,
	max_span: int,
):
	"""Yield slices over sorted intervals bounded by count and genomic span."""
	n_intervals = start_positions.shape[0]
	if n_intervals == 0:
		return

	batch_start = 0
	batch_min = int(start_positions[0])
	batch_max_end = batch_min + width

	for idx in range(1, n_intervals):
		current_start = int(start_positions[idx])
		current_end = current_start + width
		span = max(batch_max_end, current_end) - batch_min
		if (idx - batch_start) >= batch_size or span > max_span:
			yield slice(batch_start, idx)
			batch_start = idx
			batch_min = current_start
			batch_max_end = current_end
		else:
			batch_max_end = max(batch_max_end, current_end)

	yield slice(batch_start, n_intervals)


def _bin_array_numpy(arr: np.ndarray, bin_size: int, agg_func: str = "mean", axis: int = 0) -> np.ndarray:
	"""Bin a NumPy array along a specified axis into fixed-size bins."""
	n_pos = int(arr.shape[axis])
	if n_pos == 0:
		new_shape = list(arr.shape)
		new_shape[axis] = 0
		return np.empty(tuple(new_shape), dtype=arr.dtype)

	n_bins = n_pos // bin_size
	if n_bins == 0:
		new_shape = list(arr.shape)
		new_shape[axis] = 0
		return np.empty(tuple(new_shape), dtype=arr.dtype)

	slc = [slice(None)] * arr.ndim
	slc[axis] = slice(0, n_bins * bin_size)
	trimmed = arr[tuple(slc)]

	new_shape = trimmed.shape[:axis] + (n_bins, bin_size) + trimmed.shape[axis + 1 :]
	reshaped = trimmed.reshape(new_shape)
	agg_axis = axis + 1

	if agg_func == "mean":
		valid = ~np.isnan(reshaped)
		counts = valid.sum(axis=agg_axis, dtype=np.int32)
		sums = np.where(valid, reshaped, 0.0).sum(axis=agg_axis, dtype=np.float32)
		means = np.full(sums.shape, np.nan, dtype=np.float32)
		np.divide(sums, counts, out=means, where=counts > 0)
		return means
	if agg_func == "sum":
		return np.nansum(reshaped, axis=agg_axis)
	if agg_func == "max":
		return np.nanmax(reshaped, axis=agg_axis)
	if agg_func == "min":
		return np.nanmin(reshaped, axis=agg_axis)
	if agg_func == "median":
		return np.nanpercentile(reshaped, 50, axis=agg_axis)

	raise ValueError(f"Unknown aggregation function: {agg_func}")


def _read_contig_matrix(zarr_array, sample_indices: np.ndarray, start: int, end: int) -> np.ndarray:
	"""Read a contiguous positions-by-samples slice from a per-chromosome Zarr array."""
	if end <= start:
		return np.empty((0, len(sample_indices)), dtype=np.float32)

	def _read_basic(sample_sel: slice) -> np.ndarray:
		return np.asarray(
			zarr_array.get_basic_selection((sample_sel, slice(start, end))),
			dtype=np.float32,
		)

	def _read_sorted_runs(sorted_sample_indices: np.ndarray) -> np.ndarray:
		run_breaks = np.flatnonzero(np.diff(sorted_sample_indices) != 1) + 1
		run_starts = np.concatenate(([0], run_breaks))
		run_ends = np.concatenate((run_breaks, [sorted_sample_indices.size]))
		run_blocks = [
			_read_basic(slice(int(sorted_sample_indices[run_start]), int(sorted_sample_indices[run_end - 1]) + 1))
			for run_start, run_end in zip(run_starts, run_ends, strict=False)
		]
		return np.concatenate(run_blocks, axis=0).T

	sample_indices = np.asarray(sample_indices, dtype=np.int64)
	if sample_indices.size == 1:
		sample_start = int(sample_indices[0])
		block = _read_basic(slice(sample_start, sample_start + 1))
		return block.T

	order = np.argsort(sample_indices, kind="mergesort")
	sorted_indices = sample_indices[order]
	restore_order = np.empty_like(order)
	restore_order[order] = np.arange(order.size)

	span_start = int(sorted_indices[0])
	span_end = int(sorted_indices[-1]) + 1
	span_len = span_end - span_start

	# Prefer plain slicing when selected samples are contiguous or nearly contiguous.
	if np.array_equal(sorted_indices, np.arange(span_start, span_end, dtype=np.int64)):
		block = _read_basic(slice(span_start, span_end))
		return block[(sorted_indices - span_start), :].T[:, restore_order]

	if span_len <= max(sorted_indices.size * 4, sorted_indices.size + 8):
		block = _read_basic(slice(span_start, span_end))
		return block[(sorted_indices - span_start), :].T[:, restore_order]

	merged = _read_sorted_runs(sorted_indices)
	return merged.T[:, restore_order]


def _gather_numpy_batch(
	arr: np.ndarray,
	start_positions: np.ndarray,
	width: int,
	*,
	valid_lengths: np.ndarray | None = None,
	source_start: int,
	arr_len: int,
) -> np.ndarray:
	"""Gather fixed-width windows from a NumPy positions-by-samples array with NaN padding."""
	n_intervals = start_positions.shape[0]
	n_samples = arr.shape[1]
	if n_intervals == 0 or width == 0:
		return np.empty((n_intervals, width, n_samples), dtype=np.float32)

	offsets = np.arange(width, dtype=np.int64)
	abs_indices = start_positions[:, None] + offsets[None, :]
	valid = (abs_indices >= 0) & (abs_indices < arr_len)
	if valid_lengths is not None:
		valid &= offsets[None, :] < valid_lengths[:, None]

	if arr.shape[0] == 0:
		return np.full((n_intervals, width, n_samples), np.nan, dtype=np.float32)

	safe = np.clip(abs_indices - source_start, 0, arr.shape[0] - 1)
	gathered = arr[safe]
	return np.where(valid[:, :, None], gathered, np.nan).astype(np.float32, copy=False)


def _gather_binned_numpy_batch(
	arr: np.ndarray,
	start_positions: np.ndarray,
	*,
	total_width: int,
	bin_size: int,
	agg_func: str,
	source_start: int,
	arr_len: int,
	valid_lengths: np.ndarray | None = None,
	target_bytes: int = 16 * 1024**2,
) -> np.ndarray:
	"""Gather and aggregate fixed-size bins without materializing the full per-base window."""
	n_intervals = start_positions.shape[0]
	n_samples = arr.shape[1]
	if n_intervals == 0 or total_width == 0:
		return np.empty((n_intervals, 0, n_samples), dtype=np.float32)

	n_bins = total_width // bin_size
	if n_bins == 0:
		return np.empty((n_intervals, 0, n_samples), dtype=np.float32)

	if agg_func in {"mean", "sum"}:
		bin_offsets = np.arange(n_bins, dtype=np.int64) * bin_size
		bin_starts = start_positions[:, None] + bin_offsets[None, :]
		bin_ends = bin_starts + bin_size

		valid_starts = np.clip(bin_starts, 0, arr_len)
		valid_ends = np.clip(bin_ends, 0, arr_len)
		if valid_lengths is not None:
			interval_ends = start_positions + valid_lengths
			valid_ends = np.minimum(valid_ends, interval_ends[:, None])

		counts = np.maximum(valid_ends - valid_starts, 0).astype(np.int32, copy=False)
		local_starts = np.clip(valid_starts - source_start, 0, arr.shape[0])
		local_ends = np.clip(valid_ends - source_start, 0, arr.shape[0])
		prefix = np.empty((arr.shape[0] + 1, n_samples), dtype=np.float32)
		prefix[0, :] = 0.0
		np.cumsum(arr, axis=0, dtype=np.float32, out=prefix[1:, :])
		binned = prefix[local_ends] - prefix[local_starts]
		if agg_func == "sum":
			return binned.astype(np.float32, copy=False)

		means = np.full(binned.shape, np.nan, dtype=np.float32)
		np.divide(binned, counts[:, :, None], out=means, where=counts[:, :, None] > 0)
		return means

	bytes_per_bin = max(1, n_intervals) * max(1, n_samples) * bin_size * np.dtype(np.float32).itemsize
	bins_per_chunk = max(1, min(n_bins, target_bytes // max(1, bytes_per_bin)))
	chunk_outputs: list[np.ndarray] = []

	for bin_start in range(0, n_bins, bins_per_chunk):
		chunk_bins = min(bins_per_chunk, n_bins - bin_start)
		chunk_width = chunk_bins * bin_size
		chunk_starts = start_positions + bin_start * bin_size
		chunk_valid_lengths = None
		if valid_lengths is not None:
			remaining = valid_lengths - bin_start * bin_size
			chunk_valid_lengths = np.clip(remaining, 0, chunk_width)

		chunk = _gather_numpy_batch(
			arr,
			chunk_starts,
			chunk_width,
			valid_lengths=chunk_valid_lengths,
			source_start=source_start,
			arr_len=arr_len,
		)
		chunk_outputs.append(_bin_array_numpy(chunk, bin_size, agg_func=agg_func, axis=1))

	return chunk_outputs[0] if len(chunk_outputs) == 1 else np.concatenate(chunk_outputs, axis=1)


def extract_byranges_signal(
	dataset,
	ranges_df: pd.DataFrame | pr.PyRanges | None = None,
	intervals_path: str | None = None,
	feature_type: FeatureType | str | None = None,
	gtf_path: str | Iterable[str] | None = None,
	start_col: str = "Start",
	end_col: str = "End",
	contig_col: str = "Chromosome",
	fixed_width: int | None = None,
	upstream: int | None = None,
	downstream: int | None = None,
	anchor: AnchorPoint | str = AnchorPoint.MIDPOINT,
	bin_size: int | None = None,
	bin_agg: ReductionMethod | str = ReductionMethod.MEAN,
	include_incomplete: bool = False,
	sample_indices: np.ndarray | None = None,
	strand_aware: bool = False,
	force_strand: str | None = None,
	max_workers: int = 1,
) -> xr.DataArray:
	"""
	Extract raw per-position signal over genomic ranges.

	Unlike reduce_byranges_signal, this returns the full signal vector for each
	interval, optionally resized to fixed_width and binned.

	Supports three input modes for ranges:
	1. `ranges_df`: directly provide ranges as DataFrame or PyRanges
	2. `intervals_path`: path to BED/GTF file
	3. `feature_type` + `gtf_path`: predefined feature selection from GTF

	Parameters
	----------
	dataset : BamStore or zarr.Group
		Source dataset. Must expose `.root` (zarr.Group) or be a zarr Group itself.
	ranges_df : pd.DataFrame or pr.PyRanges, optional
		DataFrame/PyRanges with columns [contig, start, end] or [Chromosome, Start, End].
		Required if `intervals_path` and `feature_type` are not provided.
	intervals_path : str, optional
		Path to intervals file (BED or GTF format).
	feature_type : FeatureType or str, optional
		Predefined feature type ('gene', 'transcript', 'exon', 'promoter').
		Requires `gtf_path` to be set.
	gtf_path : str or Iterable[str], optional
		Path(s) to GTF file(s) for feature extraction.
	start_col : str
		Column name for start position (default: "Start").
	end_col : str
		Column name for end position (default: "End").
	contig_col : str
		Column name for chromosome/contig (default: "Chromosome").
	fixed_width : int, optional
		If set, all intervals are resized to this width (centered on anchor).
		Must be divisible by bin_size if bin_size is also set.
	anchor : AnchorPoint or str, default "midpoint"
		Anchor point for fixed_width: 'midpoint', 'start', or 'end'.
		'start' and 'end' are strand-aware (5'/3') if Strand column is present.
	bin_size : int, optional
		If set, aggregate positions into bins of this size (e.g., 50 bp).
		Must divide fixed_width evenly if fixed_width is also set.
	bin_agg : ReductionMethod or str, default "mean"
		Aggregation method for binning: 'mean', 'sum', 'max', 'min', 'median'.
	include_incomplete : bool
		If False (default), only use samples marked as complete in metadata.
	sample_indices : np.ndarray, optional
		Explicit sample indices to keep (applied after completion filter).
	strand_aware : bool, default False
		If True and the store was built with ``stranded`` set (i.e. ``{chrom}_fwd``
		and ``{chrom}_rev`` arrays exist), select per-interval coverage from the
		appropriate strand array based on the ``Strand`` column.  Intervals on ``"+"``
		are drawn from ``{chrom}_fwd``; ``"-"`` intervals from ``{chrom}_rev``.
		Falls back to total coverage when stranded arrays are absent.
	force_strand : {"+"  , "-"}, optional
		Force all intervals to use the forward (``"+"`` → ``{chrom}_fwd``) or reverse
		(``"-"`` → ``{chrom}_rev``) strand array regardless of their strand annotation.
		Takes precedence over ``strand_aware``.  Falls back to total coverage when
		stranded arrays are absent.
	max_workers : int, default 1
		Number of chromosome groups to extract in parallel. Useful when ranges span
		multiple contigs stored as separate arrays.

	Returns
	-------
	xr.DataArray
		Array with dimensions (interval, relative_position, sample).
		Coordinates include interval metadata (start, end, contig, etc.) and sample names.
		Intervals shorter than fixed_width (if set) are padded with NaN.

	Raises
	------
	ValueError
		If fixed_width is not divisible by bin_size.
		If no valid ranges or samples are provided.
	TypeError
		If neither ranges_df nor intervals_path nor (feature_type, gtf_path) are provided.
	"""

	# Normalize parameters
	anchor = AnchorPoint(anchor) if isinstance(anchor, str) else anchor
	bin_agg_str = str(ReductionMethod(bin_agg) if isinstance(bin_agg, str) else bin_agg)

	# Resolve anchor window from upstream/downstream or fixed_width
	if upstream is not None or downstream is not None:
		if fixed_width is not None:
			raise ValueError("Cannot specify both fixed_width and upstream/downstream")
		_upstream = upstream if upstream is not None else 0
		_downstream = downstream if downstream is not None else 0
		_total_width = _upstream + _downstream
	elif fixed_width is not None:
		_upstream = fixed_width // 2
		_downstream = fixed_width - _upstream
		_total_width = fixed_width
	else:
		_upstream = _downstream = _total_width = None

	# Validate window divisible by bin_size
	if _total_width is not None and bin_size is not None:
		if _total_width % bin_size != 0:
			raise ValueError(
				f"Total window ({_total_width}) must be divisible by bin_size ({bin_size})"
			)

	ranges_df, start_col, end_col, contig_col = _resolve_ranges(
		ranges_df, intervals_path, feature_type, gtf_path, start_col, end_col, contig_col
	)
	sample_indices, sample_labels, root = _select_samples(dataset, include_incomplete, sample_indices)

	# Log chromosome overlap
	ranges_contigs = set(ranges_df[contig_col].unique())
	dataset_contigs = set(k for k in root.keys() if k != "metadata")
	feature_source = "GTF" if feature_type is not None else ("BED/GTF file" if intervals_path else "input ranges")
	_log_chromosome_overlap(ranges_contigs, dataset_contigs, feature_source)

	# Check for Strand column for strand-aware anchoring
	has_strand = "Strand" in ranges_df.columns
	name_col = next(
		(c for c in ("Name", "name", "interval_name", "interval", "id") if c in ranges_df.columns),
		None,
	)

	# Determine global extraction width.
	# If bin_size is provided, drop remainder bases (exact multiple of bin_size only).
	if _total_width is None:
		contig_lengths = {k: int(root[k].shape[1]) for k in root.keys() if k != "metadata"}
		contig_len = ranges_df[contig_col].map(contig_lengths)
		starts_all = np.asarray(ranges_df[start_col], dtype=np.int64)
		ends_all = np.asarray(ranges_df[end_col], dtype=np.int64)
		contig_len_arr = np.asarray(contig_len, dtype=np.float64)
		valid_contig = ~np.isnan(contig_len_arr)
		if not np.any(valid_contig):
			raise ValueError("No valid contigs found for extraction")

		clipped_starts = np.maximum(starts_all[valid_contig], 0)
		clipped_ends = np.minimum(ends_all[valid_contig], contig_len_arr[valid_contig].astype(np.int64))
		lengths = clipped_ends - clipped_starts
		lengths = lengths[lengths > 0]
		if lengths.size == 0:
			raise ValueError("No valid intervals found for extraction")

		if bin_size is not None:
			lengths = (lengths // bin_size) * bin_size
			lengths = lengths[lengths > 0]
			if lengths.size == 0:
				raise ValueError(
					"All intervals are shorter than bin_size after clipping; nothing to extract"
				)
			target_bases = int(lengths.max())
		else:
			target_bases = int(lengths.max())
	else:
		target_bases = int(_total_width)

	# Extract per contig using lazy Dask graph construction.
	outputs: list[da.Array] = []
	idx_order: list[np.ndarray] = []
	starts_meta: list[np.ndarray] = []
	ends_meta: list[np.ndarray] = []
	contigs_meta: list[np.ndarray] = []
	strands_meta: list[np.ndarray] = []
	names_meta: list[np.ndarray] = []

	for contig, group in ranges_df.groupby(contig_col, observed=True):
		if contig not in root:
			continue

		orig_idx = group.index.to_numpy()
		starts = np.asarray(group[start_col], dtype=np.int64)
		ends = np.asarray(group[end_col], dtype=np.int64)
		strands = np.asarray(group["Strand"], dtype=object) if has_strand else None
		names = np.asarray(group[name_col], dtype=object) if name_col is not None else None

		use_forced_strand = (
			force_strand in ("+", "-")
			and f"{contig}_fwd" in root
			and f"{contig}_rev" in root
		)
		use_stranded = (
			not use_forced_strand
			and strand_aware
			and has_strand
			and f"{contig}_fwd" in root
			and f"{contig}_rev" in root
		)
		# Get chromosome length from zarr shape without loading data
		_ref_akey = (f"{contig}_fwd" if force_strand == "+" else f"{contig}_rev") if use_forced_strand else contig
		arr_len = int(root[_ref_akey].shape[1])

		clipped_starts = np.maximum(starts, 0)
		clipped_ends = np.minimum(ends, arr_len)
		valid = clipped_ends > clipped_starts
		if not np.all(valid):
			orig_idx = orig_idx[valid]
			starts = starts[valid]
			ends = ends[valid]
			clipped_starts = clipped_starts[valid]
			clipped_ends = clipped_ends[valid]
			if has_strand:
				strands = strands[valid]
			if names is not None:
				names = names[valid]
			group = group.loc[valid]

		if starts.size == 0:
			continue

		# Compute anchor positions and span of positions to load
		if _total_width is not None:
			if anchor == AnchorPoint.MIDPOINT:
				anchor_pos = (starts + ends) // 2
			elif anchor == AnchorPoint.START:
				anchor_pos = np.where(strands == "-", ends, starts) if has_strand else starts
			elif anchor == AnchorPoint.END:
				anchor_pos = np.where(strands == "-", starts, ends) if has_strand else ends
			else:
				raise ValueError(f"Unknown anchor point: {anchor}")
			extract_starts = anchor_pos - _upstream
			span_start = int(max(0, extract_starts.min()))
			span_end = int(min(arr_len, int(extract_starts.max()) + _total_width))
		else:
			anchor_pos = None
			extract_starts = None
			span_start = int(clipped_starts.min())
			span_end = int(clipped_ends.max())

		# Load only the span of positions needed, transposed to (positions, samples)
		def _load_arr(akey: str) -> "da.Array":
			region = root[akey].oindex[sample_indices.tolist(), span_start:span_end].T.astype(np.float32)
			return da.from_array(region, chunks=("auto", len(sample_indices)))

		if use_forced_strand:
			arr = _load_arr(f"{contig}_fwd" if force_strand == "+" else f"{contig}_rev")
		elif use_stranded:
			arr_fwd = _load_arr(f"{contig}_fwd")
			arr_rev = _load_arr(f"{contig}_rev")
			arr = arr_fwd
		else:
			arr = _load_arr(contig)

		# arr_len is now the loaded span length; rebase coordinates to span_start
		arr_len = int(arr.shape[0])
		clipped_starts = clipped_starts - span_start
		clipped_ends = clipped_ends - span_start

		if _total_width is not None:
			extract_starts = extract_starts - span_start
			pad_left = int(max(0, -int(extract_starts.min())))
			pad_right = int(max(0, int(extract_starts.max() + _total_width) - arr_len))

			if use_stranded:
				if pad_left or pad_right:
					arr_fwd = da.pad(arr_fwd, ((pad_left, pad_right), (0, 0)), mode="constant", constant_values=np.nan)
					arr_rev = da.pad(arr_rev, ((pad_left, pad_right), (0, 0)), mode="constant", constant_values=np.nan)
				start_idx = (extract_starts + pad_left).astype(np.int64)
				offsets = np.arange(_total_width, dtype=np.int64)
				indices = start_idx[:, None] + offsets[None, :]
				n_intervals = start_idx.shape[0]
				gathered_fwd = da.take(arr_fwd, indices.reshape(-1), axis=0).reshape((n_intervals, _total_width, arr_fwd.shape[1]))
				gathered_rev = da.take(arr_rev, indices.reshape(-1), axis=0).reshape((n_intervals, _total_width, arr_rev.shape[1]))
				is_plus = da.from_array((strands == "+")[:, None, None])
				gathered = da.where(is_plus, gathered_fwd, gathered_rev)
			else:
				if pad_left or pad_right:
					arr = da.pad(arr, ((pad_left, pad_right), (0, 0)), mode="constant", constant_values=np.nan)
				start_idx = (extract_starts + pad_left).astype(np.int64)
				offsets = np.arange(_total_width, dtype=np.int64)
				indices = start_idx[:, None] + offsets[None, :]
				gathered = da.take(arr, indices.reshape(-1), axis=0).reshape((start_idx.shape[0], _total_width, arr.shape[1]))
			signal = gathered

			if bin_size is not None:
				n_bins = _total_width // bin_size
				reshaped = signal.reshape((signal.shape[0], n_bins, bin_size, signal.shape[2]))
				if bin_agg_str == "mean":
					signal = da.nanmean(reshaped, axis=2)
				elif bin_agg_str == "sum":
					signal = da.nansum(reshaped, axis=2)
				elif bin_agg_str == "max":
					signal = da.nanmax(reshaped, axis=2)
				elif bin_agg_str == "min":
					signal = da.nanmin(reshaped, axis=2)
				elif bin_agg_str == "median":
					signal = da.nanpercentile(reshaped, 50, axis=2)
				else:
					raise ValueError(f"Unknown bin aggregation function: {bin_agg_str}")

			out = signal
		else:
			lengths = clipped_ends - clipped_starts
			if bin_size is not None:
				lengths = (lengths // bin_size) * bin_size
				valid_len = lengths > 0
				if not np.all(valid_len):
					orig_idx = orig_idx[valid_len]
					starts = starts[valid_len]
					ends = ends[valid_len]
					clipped_starts = clipped_starts[valid_len]
					clipped_ends = clipped_ends[valid_len]
					lengths = lengths[valid_len]
					if has_strand:
						strands = strands[valid_len]
					if names is not None:
						names = names[valid_len]
					group = group.loc[valid_len]
				if lengths.size == 0:
					continue

			start_idx = clipped_starts.astype(np.int64)
			offsets = np.arange(target_bases, dtype=np.int64)
			indices = start_idx[:, None] + offsets[None, :]

			if use_stranded:
				pad_right = int(max(0, int(start_idx.max() + target_bases) - arr_len))
				if pad_right:
					arr_fwd = da.pad(arr_fwd, ((0, pad_right), (0, 0)), mode="constant", constant_values=np.nan)
					arr_rev = da.pad(arr_rev, ((0, pad_right), (0, 0)), mode="constant", constant_values=np.nan)
				n_intervals = start_idx.shape[0]
				gathered_fwd = da.take(arr_fwd, indices.reshape(-1), axis=0).reshape((n_intervals, target_bases, arr_fwd.shape[1]))
				gathered_rev = da.take(arr_rev, indices.reshape(-1), axis=0).reshape((n_intervals, target_bases, arr_rev.shape[1]))
				is_plus = da.from_array((strands == "+")[:, None, None])
				gathered = da.where(is_plus, gathered_fwd, gathered_rev)
			else:
				pad_right = int(max(0, int(start_idx.max() + target_bases) - arr_len))
				if pad_right:
					arr = da.pad(arr, ((0, pad_right), (0, 0)), mode="constant", constant_values=np.nan)
				gathered = da.take(arr, indices.reshape(-1), axis=0).reshape((start_idx.shape[0], target_bases, arr.shape[1]))

			mask = offsets[None, :] < lengths[:, None]
			mask_da = da.from_array(mask, chunks=(min(mask.shape[0], 256), mask.shape[1]))
			signal = da.where(mask_da[:, :, None], gathered, np.nan)

			if bin_size is not None:
				n_bins = target_bases // bin_size
				reshaped = signal.reshape((signal.shape[0], n_bins, bin_size, signal.shape[2]))
				if bin_agg_str == "mean":
					signal = da.nanmean(reshaped, axis=2)
				elif bin_agg_str == "sum":
					signal = da.nansum(reshaped, axis=2)
				elif bin_agg_str == "max":
					signal = da.nanmax(reshaped, axis=2)
				elif bin_agg_str == "min":
					signal = da.nanmin(reshaped, axis=2)
				elif bin_agg_str == "median":
					signal = da.nanpercentile(reshaped, 50, axis=2)
				else:
					raise ValueError(f"Unknown bin aggregation function: {bin_agg_str}")

			out = signal

		outputs.append(out)
		idx_order.append(orig_idx)
		starts_meta.append(starts)
		ends_meta.append(ends)
		contigs_meta.append(np.asarray([contig] * starts.shape[0]))
		if has_strand:
			strands_meta.append(strands)
		if names is not None:
			names_meta.append(names)

	if not outputs:
		raise ValueError("No valid intervals found for extraction")

	stacked = da.concatenate(outputs, axis=0)
	range_index = np.concatenate(idx_order)
	sort_order = np.argsort(range_index)
	stacked = da.take(stacked, sort_order, axis=0)

	starts_cat = np.concatenate(starts_meta)[sort_order]
	ends_cat = np.concatenate(ends_meta)[sort_order]
	contigs_cat = np.concatenate(contigs_meta)[sort_order]

	relative_position_name = "bin" if bin_size is not None else "relative_position"
	# When an anchor window is used, express positions as bp offset from anchor.
	if _total_width is not None:
		if bin_size is not None:
			n_bins = _total_width // bin_size
			pos_values = np.arange(n_bins, dtype=np.int64) * bin_size - _upstream
		else:
			pos_values = np.arange(-_upstream, _downstream, dtype=np.int64)
	else:
		pos_values = np.arange(int(stacked.shape[1]), dtype=int)
	coords: dict[str, object] = {
		"interval": np.arange(int(stacked.shape[0]), dtype=int),
		relative_position_name: pos_values,
		"sample": np.asarray(sample_labels),
		"start": ("interval", starts_cat),
		"end": ("interval", ends_cat),
		"contig": ("interval", contigs_cat),
	}

	if has_strand:
		strands_cat = np.concatenate(strands_meta)[sort_order]
		coords["strand"] = ("interval", strands_cat)
	if name_col is not None:
		names_cat = np.concatenate(names_meta)[sort_order]
		coords["name"] = ("interval", names_cat)

	return xr.DataArray(
		stacked,
		dims=("interval", relative_position_name, "sample"),
		coords=coords,
		attrs={
			"upstream": _upstream,
			"downstream": _downstream,
			"anchor": str(anchor),
			"bin_size": bin_size,
			"bin_agg": bin_agg_str if bin_size is not None else None,
		},
	)


def reduce_byranges_signal(
	dataset,
	ranges_df: pd.DataFrame | pr.PyRanges | None = None,
	intervals_path: str | None = None,
	feature_type: FeatureType | str | None = None,
	gtf_path: str | Iterable[str] | None = None,
	start_col: str = "Start",
	end_col: str = "End",
	contig_col: str = "Chromosome",
	min_count: int = 1,
	reduction: ReductionMethod | str = ReductionMethod.MEAN,
	include_incomplete: bool = False,
	sample_indices: np.ndarray | None = None,
	strand_mode: int = 0,
) -> xr.Dataset:
	"""
	Summarize per-chromosome Zarr arrays over genomic ranges using efficient reduction.

	Supports three input modes for ranges:
	1. `ranges_df`: directly provide ranges as DataFrame or PyRanges
	2. `intervals_path`: path to BED/GTF file
	3. `feature_type` + `gtf_path`: predefined feature selection from GTF

	Parameters
	----------
	dataset : QuantNadoDataset | zarr.Group | BamStore
		Source dataset. Must expose `.root` (zarr.Group) or be a zarr Group itself.
	ranges_df : pd.DataFrame or pr.PyRanges, optional
		DataFrame/PyRanges with columns [contig, start, end] or [Chromosome, Start, End].
		Required if `intervals_path` and `feature_type` are not provided.
	intervals_path : str, optional
		Path to intervals file (BED or GTF format).
	feature_type : FeatureType or str, optional
		Predefined feature type ('gene', 'transcript', 'exon', 'promoter').
		Requires `gtf_path` to be set.
	gtf_path : str or Iterable[str], optional
		Path(s) to GTF file(s) for feature extraction.
	start_col : str
		Column name for start position (default: "Start" for PyRanges convention).
	end_col : str
		Column name for end position (default: "End" for PyRanges convention).
	contig_col : str
		Column name for chromosome/contig (default: "Chromosome" for PyRanges convention).
	min_count : int
		Minimum count threshold for valid mean values (default: 1).
	reduction : ReductionMethod or str
		Reduction method: 'mean', 'sum', 'max', 'min', 'median' (default: 'mean').
	include_incomplete : bool
		If False (default), only use samples marked as complete in metadata.
	sample_indices : np.ndarray, optional
		Explicit sample indices to keep (applied after completion filter).
	strand_mode : int, default 0
		0 — use total coverage (unstranded). When a ``Strand``/``strand`` column
		is present in the ranges, modes 1 and 2 perform per-feature strand-aware
		counting: each feature's reads are drawn from the array that corresponds to
		its annotated strand.

		- ``1`` (F / ISF / ligation): ``+`` features → ``{chrom}_fwd``,
		  ``-`` features → ``{chrom}_rev``.
		- ``2`` (R / ISR / dUTP): ``+`` features → ``{chrom}_rev``,
		  ``-`` features → ``{chrom}_fwd``.

		Requires the BamStore to have been built with ``stranded`` set. Falls back
		to total coverage for features with no strand annotation or when the
		stranded arrays are absent.

	Returns
	-------
	xr.Dataset
		Dataset with coordinates [ranges, sample] and data variables [sum, count, mean, reduction].
		Coordinates include range metadata: range_index, start, end, range_length, contig.

	Raises
	------
	ValueError
		If no valid ranges or samples are provided.
	TypeError
		If neither ranges_df nor intervals_path nor (feature_type, gtf_path) are provided.
	"""

	# Normalize reduction method
	reduction = ReductionMethod(reduction) if isinstance(reduction, str) else reduction
	reduction_str = str(reduction)

	ranges_df, start_col, end_col, contig_col = _resolve_ranges(
		ranges_df, intervals_path, feature_type, gtf_path, start_col, end_col, contig_col
	)
	sample_indices, sample_labels, root = _select_samples(dataset, include_incomplete, sample_indices)

	# Log chromosome overlap between ranges and dataset
	ranges_contigs = set(ranges_df[contig_col].unique())
	dataset_contigs = set(
		k for k in root.keys()
		if k != "metadata" and not k.endswith("_fwd") and not k.endswith("_rev")
	)
	feature_source = "GTF" if feature_type is not None else ("BED/GTF file" if intervals_path else "input ranges")
	_log_chromosome_overlap(ranges_contigs, dataset_contigs, feature_source)

	_array_suffix = {1: "_fwd", 2: "_rev"}.get(strand_mode, "")
	# Per-feature strand selection: split features by their annotation and route to
	# the matching _fwd/_rev array.  Applies when strand_mode != 0 and a strand
	# column is present in the ranges (takes precedence over the global suffix).
	strand_col_name = next((c for c in ("Strand", "strand") if c in ranges_df.columns), None)
	_strand_per_feature = strand_mode != 0 and strand_col_name is not None

	# Group ranges by contig and reduce
	outputs = []
	idx_order = []
	starts_all = []
	ends_all = []
	contigs_all = []
	strands_all = []
	names_all = []
	has_strand = "Strand" in ranges_df.columns
	name_col = next(
		(c for c in ("Name", "name", "interval_name", "interval", "id") if c in ranges_df.columns),
		None,
	)

	for contig, group in ranges_df.groupby(contig_col, observed=True):
		# Build (sub_group, array_key) pairs.
		# When strand_mode != 0 and a strand column is present, split features by
		# their annotation so that each half reads from the correct _fwd/_rev array:
		#   mode 1 (F / ISF / ligation):  + → _fwd,  - → _rev
		#   mode 2 (R / ISR / dUTP):      + → _rev,  - → _fwd
		# Unstranded features (strand not in +/-) fall back to total coverage.
		if _strand_per_feature:
			_strand_to_key = (
				{"+": f"{contig}_fwd", "-": f"{contig}_rev"}
				if strand_mode == 1
				else {"+": f"{contig}_rev", "-": f"{contig}_fwd"}
			)
			pairs = [
				(sg, _strand_to_key.get(str(sv), contig))
				for sv, sg in group.groupby(strand_col_name)
			]
		else:
			_akey = f"{contig}{_array_suffix}" if _array_suffix and f"{contig}{_array_suffix}" in root else contig
			pairs = [(group, _akey)]

		for sg, akey in pairs:
			if akey not in root:
				continue

			starts = np.asarray(sg[start_col], dtype=np.int64)
			ends = np.asarray(sg[end_col], dtype=np.int64)
			names = np.asarray(sg[name_col], dtype=object) if name_col is not None else None

			# Chromosome length from zarr shape; no need to load data yet
			arr_len = int(root[akey].shape[1])

			# Clip coordinates to valid range
			starts = starts.clip(min=0)
			ends = ends.clip(max=arr_len)

			# Filter out invalid ranges
			valid = (ends > starts) & (starts < arr_len) & (ends > 0)
			if not np.all(valid):
				starts = starts[valid]
				ends = ends[valid]
				if names is not None:
					names = names[valid]
				sg = sg.loc[valid]

			if starts.size == 0:
				continue

			# Load only the span of positions touched by these ranges.
			# Avoids building a full-chromosome dask graph when ranges cluster in a small region.
			span_start = int(starts.min())
			span_end = int(ends.max())
			region_np = root[akey].oindex[sample_indices.tolist(), span_start:span_end].T
			arr = da.from_array(region_np, chunks=("auto", len(sample_indices)))

			# Rebase range coordinates to be relative to span_start
			adj_starts = starts - span_start
			adj_ends = ends - span_start

			# Reduce using appropriate method
			# Always compute sum/count/mean via prefix sums (fast + consistent API)
			reduced = _reduce_byranges_prefix(adj_starts, adj_ends, arr, min_count=min_count)
			if reduction_str == "mean":
				reduction_data = reduced["mean"]
			else:
				reduction_data = _reduce_ranges_vectorized(arr, adj_starts, adj_ends, reduction_str)

			outputs.append(
				{
					"sum": reduced["sum"],
					"count": reduced["count"],
					"mean": reduced["mean"],
					reduction_str: reduction_data,
				}
			)
			idx_order.append(sg.index.to_numpy())
			starts_all.append(starts)
			ends_all.append(ends)
			contigs_all.append(np.asarray(sg[contig_col]))
			if has_strand:
				strands_all.append(np.asarray(sg["Strand"]))
			if names is not None:
				names_all.append(names)

	if not outputs:
		raise ValueError("No valid ranges found for provided contigs")

	# Concatenate across contigs preserving original order
	sums = da.concatenate([o["sum"] for o in outputs], axis=0)
	counts = da.concatenate([o["count"] for o in outputs], axis=0)
	means = da.concatenate([o["mean"] for o in outputs], axis=0)
	red_data = da.concatenate([o[reduction_str] for o in outputs], axis=0)

	range_index = np.concatenate(idx_order)
	starts_cat = np.concatenate(starts_all)
	ends_cat = np.concatenate(ends_all)
	contigs_cat = np.concatenate(contigs_all)
	if name_col is not None:
		names_cat = np.concatenate(names_all)

	sort_order = np.argsort(range_index)

	coords: dict[str, object] = {
		"ranges": np.arange(range_index.size, dtype=int),
		"range_index": ("ranges", range_index[sort_order]),
		"sample": np.asarray(sample_labels),
		"start": ("ranges", starts_cat[sort_order]),
		"end": ("ranges", ends_cat[sort_order]),
		"range_length": ("ranges", ends_cat[sort_order] - starts_cat[sort_order]),
		"contig": ("ranges", contigs_cat[sort_order]),
	}

	if has_strand:
		strands_cat = np.concatenate(strands_all)
		coords["strand"] = ("ranges", strands_cat[sort_order])
	if name_col is not None:
		coords["name"] = ("ranges", names_cat[sort_order])

	return xr.Dataset(
		{
			"sum": (("ranges", "sample"), sums[sort_order]),
			"count": (("ranges", "sample"), counts[sort_order]),
			"mean": (("ranges", "sample"), means[sort_order]),
			reduction_str: (("ranges", "sample"), red_data[sort_order]),
		},
		coords=coords,
	)

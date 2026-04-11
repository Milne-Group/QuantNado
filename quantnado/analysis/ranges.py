from __future__ import annotations

import numpy as np
import pyranges1 as pr
import dask.array as da


def merge_ranges(ranges: pr.PyRanges, columns: list[str] | None = None, strand: bool | None = None, sep: str = ";") -> pr.PyRanges:
    """Merge a PyRanges object and concatenate selected annotation columns per cluster."""
    if not isinstance(ranges, pr.PyRanges):
        raise ValueError("ranges must be a PyRanges object")

    cols = columns or ["Gene_name", "Gene_type", "Gene_id", "Tag", "Level"]

    clustered = ranges.cluster_overlaps(use_strand=strand or False)
    merged = clustered.merge_overlaps(use_strand=strand or False, match_by="Cluster")

    for col in cols:
        values = []
        for cluster_id in merged["Cluster"].values:
            mask = clustered["Cluster"].values == cluster_id
            joined = sep.join([str(v) for v in clustered[col].values[mask]])
            values.append(joined)
        merged[col] = np.asarray(values)

    return merged


def get_fixed_windows(contig_lengths: dict[str, int], window_size: int = 50_000) -> pr.PyRanges:
    """Create non-overlapping fixed windows per contig length."""
    chroms = list(contig_lengths.keys())
    starts = [0] * len(chroms)
    ends = [contig_lengths[c] for c in chroms]
    gr = pr.PyRanges({"Chromosome": chroms, "Start": starts, "End": ends})
    gr = gr.window_ranges(window_size, use_strand=False)
    gr["Ranges_ID"] = np.arange(len(gr))
    return gr


def masked_array_fromranges(positions_array: da.Array | np.ndarray, contig: str, ranges: pr.PyRanges) -> np.ndarray:
    """Return boolean mask for positions indicating overlap with ranges on a contig."""
    if not isinstance(positions_array, (da.Array, np.ndarray)) or not np.issubdtype(positions_array.dtype, np.integer):
        raise ValueError("positions_array must be an integer array")
    if not np.all(np.diff(np.asarray(positions_array)) > 0):
        raise ValueError("positions_array must be monotonically increasing")
    if not isinstance(ranges, pr.PyRanges):
        raise ValueError("ranges must be a PyRanges object")

    max_pos = int(np.max(positions_array))
    mask = np.zeros(max_pos + 1, dtype=bool)

    df = ranges.loci[contig]

    for start, end in zip(df["Start"].values, df["End"].values):
        mask[start:end] = True

    return mask[positions_array]


def default_position_mask(positions_array: da.Array | np.ndarray) -> np.ndarray:
    """Return an all-False mask for validated integer, increasing positions."""
    if not isinstance(positions_array, (da.Array, np.ndarray)) or not np.issubdtype(positions_array.dtype, np.integer):
        raise ValueError("positions_array must be an integer array")
    if not np.all(np.diff(np.asarray(positions_array)) > 0):
        raise ValueError("positions_array must be monotonically increasing")
    return np.zeros(len(positions_array), dtype=bool)


def ranges_loader(
    ranges: pr.PyRanges | list[pr.PyRanges],
    ranges_are_1based: bool = False,
    merge_intervals: bool = False,
) -> pr.PyRanges:
    """Normalize input ranges: concat, optionally shift from 1-based, reject stranded, optionally merge."""
    if isinstance(ranges, list):
        if not ranges:
            raise ValueError("No ranges provided")
        ranges = pr.concat(ranges)
        ranges["Ranges_ID"] = np.arange(len(ranges))

    if not isinstance(ranges, pr.PyRanges):
        raise ValueError("ranges must be a PyRanges object or list of them")

    if ranges_are_1based:
        ranges["Start"] = ranges["Start"] - 1
        ranges["End"] = ranges["End"] - 1

    if ranges.strand_valid:
        raise ValueError("Stranded ranges not supported; drop Strand first")

    if merge_intervals:
        ranges = ranges.merge_overlaps()

    return ranges


def extract_signal_into_bins(
    intervals: list[tuple[str, int, int]],
    dataset,
    assay: str,
    bin_size: int,
    samples: list[str] | None = None,
) -> np.ndarray:
    """Extract signal from dataset into fixed-width bins over intervals.

    Parameters
    ----------
    intervals : list of (chrom, start, end) tuples
        1-based inclusive genomic intervals.
    dataset : QuantNadoDataset
        Dataset to extract signal from.
    assay : str
        Assay name (e.g., "atac", "chip_h3k27ac", "rna_fwd").
    bin_size : int
        Width of each bin in bp.
    samples : list of str, optional
        Sample names to extract. If None, uses all samples.

    Returns
    -------
    np.ndarray
        Shape (n_intervals, n_bins, n_samples) with summed signal per bin.
    """
    from collections import defaultdict

    if samples is None:
        samples = dataset.sample_names

    if not intervals:
        raise ValueError("No intervals provided")

    n_intervals = len(intervals)
    n_samples = len(samples)

    _, start0, end0 = intervals[0]
    interval_width = end0 - start0 + 1
    n_bins = max(1, interval_width // bin_size)

    output = np.zeros((n_intervals, n_bins, n_samples), dtype=np.float32)

    # Group intervals by chromosome so each chrom is loaded from zarr only once.
    chrom_groups: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for i, (chrom, start, end) in enumerate(intervals):
        chrom_groups[chrom].append((i, start, end))

    chromsizes = dataset.chromsizes

    for chrom, chrom_intervals in chrom_groups.items():
        if chrom not in chromsizes:
            continue

        chrom_len = chromsizes[chrom]
        all_starts = [t[1] for t in chrom_intervals]
        all_ends = [t[2] for t in chrom_intervals]
        load_start = max(1, min(all_starts))
        load_end = min(chrom_len, max(all_ends))

        try:
            region = dataset.sel(chrom=chrom, start=load_start, end=load_end)
        except (ValueError, KeyError):
            continue

        if assay not in region.data_vars:
            continue

        # Load entire spanning range into memory once — (n_samples, n_pos)
        signal_chrom = region[assay].sel(sample=samples).values
        if signal_chrom.ndim == 1:
            signal_chrom = signal_chrom[np.newaxis, :]

        for idx, start, end in chrom_intervals:
            # Translate genomic coords to offsets within the loaded chunk
            s_off = start - load_start
            e_off = end - load_start + 1
            s_off = max(0, s_off)
            e_off = min(e_off, signal_chrom.shape[1])
            if s_off >= e_off:
                continue

            signal = signal_chrom[:, s_off:e_off].T  # (n_pos, n_samples)
            n_pos = signal.shape[0]

            # Vectorised binning via reduceat — replaces inner Python loop
            bin_edges = np.arange(0, n_pos, bin_size)
            binned = np.add.reduceat(signal, bin_edges, axis=0)  # (n_bins_actual, n_samples)
            n_bins_actual = min(binned.shape[0], n_bins)
            output[idx, :n_bins_actual, :] = binned[:n_bins_actual]

    return output
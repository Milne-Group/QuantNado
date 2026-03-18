from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyranges1 as pr
from loguru import logger


def _merge_adjacent_regions(peaks_df: pd.DataFrame) -> pd.DataFrame:
    """Merge overlapping or directly adjacent intervals within each chromosome."""
    if peaks_df.empty:
        return peaks_df

    peaks_df = peaks_df.sort_values(["Chromosome", "Start", "End"]).reset_index(drop=True)
    merged_parts: list[pd.DataFrame] = []

    for chrom, grp in peaks_df.groupby("Chromosome", sort=False):
        starts = grp["Start"].to_numpy(dtype=np.int64, copy=False)
        ends = grp["End"].to_numpy(dtype=np.int64, copy=False)

        if len(starts) == 1:
            merged_parts.append(pd.DataFrame({"Chromosome": [chrom], "Start": starts, "End": ends}))
            continue

        prev_max_end = np.maximum.accumulate(ends[:-1])
        new_group = np.empty(len(starts), dtype=bool)
        new_group[0] = True
        # Start a new region only when there is a strict gap; adjacency is merged.
        new_group[1:] = starts[1:] > prev_max_end

        group_starts = np.flatnonzero(new_group)
        merged_starts = starts[group_starts]
        merged_ends = np.maximum.reduceat(ends, group_starts)

        merged_parts.append(
            pd.DataFrame(
                {
                    "Chromosome": np.repeat(chrom, len(merged_starts)),
                    "Start": merged_starts,
                    "End": merged_ends,
                }
            )
        )

    return pd.concat(merged_parts, ignore_index=True)


def call_quantile_peaks(
    signal: pd.Series,
    chroms: pd.Series,
    starts: pd.Series,
    ends: pd.Series,
    tilesize: int = 128,
    quantile: float = 0.98,
    blacklist_file: Optional[Path] = None,
    merge: bool = True,
) -> Optional[pr.PyRanges]:
    """Call quantile-based peaks from a single sample's tile signal."""
    logger.info(f"Calling peaks for sample: {signal.name}")

    nonzero = signal[signal > 0]
    if nonzero.empty:
        logger.warning(f"[{signal.name}] No nonzero signal values.")
        return None

    if quantile >= 1.0:
        logger.warning(f"[{signal.name}] quantile=1.0 means no tile can exceed the maximum; returning None.")
        return None

    threshold = nonzero.quantile(quantile)
    logger.debug(f"[{signal.name}] Quantile {quantile} threshold = {threshold:.4f}")

    peaks = signal >= threshold

    peaks_df = pd.DataFrame(
        {
            "Chromosome": chroms,
            "Start": starts,
            "End": ends,
            "Score": signal,
            "is_peak": peaks.astype(int),
        }
    )
    peaks_df = peaks_df[peaks_df["is_peak"] == 1].drop(columns="is_peak")

    if peaks_df.empty:
        logger.warning(f"[{signal.name}] No peak tiles exceed threshold.")
        return None

    peaks_df = peaks_df.astype({"Start": int, "End": int, "Chromosome": str})
    peaks_df = peaks_df.sort_values(["Chromosome", "Start"]).reset_index(drop=True)

    if merge:
        peaks_df = _merge_adjacent_regions(peaks_df[["Chromosome", "Start", "End"]])

    pr_obj = pr.PyRanges(peaks_df)

    if blacklist_file and blacklist_file.exists():
        logger.debug(f"[{signal.name}] Subtracting blacklist regions: {blacklist_file}")
        blacklist = pr.read_bed(str(blacklist_file))
        pr_obj = pr_obj.subtract_overlaps(blacklist, strand_behavior="ignore")

    logger.info(f"[{signal.name}] Final peak count: {len(pr_obj)}")
    return pr_obj if len(pr_obj) > 0 else None


def call_peaks_from_zarr(
    zarr_path: Path,
    output_dir: Path,
    blacklist_file: Optional[Path] = None,
    tilesize: int = 128,
    window_overlap: int = 8,
    quantile: float = 0.98,
    merge: bool = True,
) -> list[str]:
    """Call quantile-based peaks from a QuantNado zarr coverage store.

    Reads per-base coverage from zarr, computes mean depth in sliding windows,
    applies RPKM normalisation (mean_depth × 1e9 / library_size), then log1p.
    """
    from ..dataset.store_bam import BamStore
    from ..analysis.normalise import get_library_sizes

    zarr_path = Path(zarr_path)
    output_dir = Path(output_dir)
    blacklist_file = Path(blacklist_file) if blacklist_file else None
    output_dir.mkdir(parents=True, exist_ok=True)

    if tilesize <= 0:
        raise ValueError(f"tilesize must be > 0, got {tilesize}")
    if window_overlap < 0 or window_overlap >= tilesize:
        raise ValueError(
            f"window_overlap must be >= 0 and < tilesize (tilesize={tilesize}), got {window_overlap}"
        )
    step = tilesize - window_overlap

    store = BamStore.open(zarr_path, read_only=True)
    chromsizes = {
        chrom: size
        for chrom, size in store.chromsizes.items()
        if "_" not in chrom
    }

    library_sizes = get_library_sizes(store)
    sample_names = store.sample_names
    completed = store.completed_mask
    valid_samples = [s for s, c in zip(sample_names, completed) if c]
    valid_indices = [i for i, c in enumerate(completed) if c]

    if not valid_samples:
        logger.error("No completed samples found in store.")
        return []

    lib_sizes_arr = np.array([library_sizes[s] for s in valid_samples], dtype=np.float64)
    logger.info(f"Found {len(valid_samples)} completed sample(s) in {zarr_path}")

    blacklist_pr = None
    if blacklist_file and blacklist_file.exists():
        logger.info(f"Loading blacklist: {blacklist_file}")
        blacklist_pr = pr.read_bed(str(blacklist_file))

    chrom_cov_parts: list[np.ndarray] = []
    tile_coord_parts: list[tuple[str, np.ndarray, np.ndarray]] = []

    for chrom, chrom_len in chromsizes.items():
        if chrom not in store.chromosomes:
            continue
        logger.debug(
            f"Sliding-window tiling {chrom} ({chrom_len:,} bp): size={tilesize}, overlap={window_overlap}, step={step}"
        )

        chrom_arr = store.root[chrom]
        # shape: (n_valid_samples, chrom_len)
        cov = chrom_arr[valid_indices, :chrom_len].astype(np.float32)

        tile_starts = np.arange(0, chrom_len, step, dtype=np.int64)
        tile_ends = np.minimum(tile_starts + tilesize, chrom_len).astype(np.int64)

        # Fast sliding-window means via cumulative sums.
        csum = np.pad(np.cumsum(cov, axis=1, dtype=np.float64), ((0, 0), (1, 0)), mode="constant")
        window_sums = csum[:, tile_ends] - csum[:, tile_starts]
        window_lengths = (tile_ends - tile_starts).astype(np.float64)
        tile_means = window_sums / window_lengths[np.newaxis, :]

        # Apply blacklist: drop tiles overlapping blacklist
        if blacklist_pr is not None:
            tiles_pr = pr.PyRanges(
                pd.DataFrame(
                    {
                        "Chromosome": np.repeat(chrom, len(tile_starts)),
                        "Start": tile_starts,
                        "End": tile_ends,
                    }
                )
            )
            keep_pr = tiles_pr.subtract_overlaps(blacklist_pr, strand_behavior="ignore")
            keep_df = pd.DataFrame(keep_pr)
            if keep_df.empty:
                continue
            # Vectorized: use np.isin to find which tile_starts appear in keep_df
            keep_mask = np.isin(tile_starts, keep_df["Start"].to_numpy(dtype=np.int64))
            tile_means = tile_means[:, keep_mask]
            tile_starts = tile_starts[keep_mask]
            tile_ends = tile_ends[keep_mask]

        # Accumulate as numpy arrays, not lists
        chrom_cov_parts.append(tile_means)
        tile_coord_parts.append((chrom, tile_starts, tile_ends))

    # Post-loop: build coordinate arrays via concatenation
    all_chroms_arr = np.repeat(
        [c for c, _, _ in tile_coord_parts],
        [len(ts) for _, ts, _ in tile_coord_parts]
    )
    all_starts_arr = np.concatenate([ts for _, ts, _ in tile_coord_parts])
    all_ends_arr = np.concatenate([te for _, _, te in tile_coord_parts])

    chroms_series = pd.Series(all_chroms_arr, name="Chromosome")
    starts_series = pd.Series(all_starts_arr, name="Start")
    ends_series = pd.Series(all_ends_arr, name="End")

    # RPKM: mean_depth × 1e9 / library_size (tile-length-independent)
    # Concatenate all tile_means across chromosomes: (n_samples, total_tiles)
    signals_matrix = np.concatenate(chrom_cov_parts, axis=1)
    signals_df = pd.DataFrame(signals_matrix.T, columns=valid_samples)
    rpkm_df = signals_df.multiply(1e9 / lib_sizes_arr, axis=1)
    log_rpkm_df = np.log1p(rpkm_df)
    parquet_path = output_dir / f"log_rpkm_{tilesize}bp_{window_overlap}bp_overlap.parquet"
    log_rpkm_df.to_parquet(parquet_path, index=False)
    logger.info(f"Saved log-RPKM tile signal to {parquet_path}")

    results = []
    for sample_name in valid_samples:
        signal = log_rpkm_df[sample_name]
        signal.name = sample_name

        pr_obj = call_quantile_peaks(
            signal=signal,
            chroms=chroms_series,
            starts=starts_series,
            ends=ends_series,
            tilesize=tilesize,
            quantile=quantile,
            merge=merge,
        )

        if pr_obj is not None:
            output_bed = output_dir / f"{sample_name}.bed"
            pr_obj.to_bed(output_bed)
            logger.success(f"Peak BED saved to: {output_bed}")
            results.append(str(output_bed))
        else:
            logger.warning(f"[{sample_name}] No peaks detected.")

    return results



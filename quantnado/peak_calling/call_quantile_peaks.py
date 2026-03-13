from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyranges1 as pr
from loguru import logger


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

    pr_obj = pr.PyRanges(peaks_df)
    if merge:
        pr_obj = pr_obj.merge_overlaps()

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
    quantile: float = 0.98,
    merge: bool = True,
) -> list[str]:
    """Call quantile-based peaks from a QuantNado zarr coverage store.

    Reads per-base coverage from zarr, computes mean depth per tile, applies
    RPKM normalisation (mean_depth × 1e9 / library_size), then log1p.
    """
    from ..dataset.store_bam import BamStore
    from ..analysis.normalise import get_library_sizes

    zarr_path = Path(zarr_path)
    output_dir = Path(output_dir)
    blacklist_file = Path(blacklist_file) if blacklist_file else None
    output_dir.mkdir(parents=True, exist_ok=True)

    store = BamStore(zarr_path)
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

    signals_by_sample: dict[str, list[float]] = {s: [] for s in valid_samples}
    all_chroms: list[str] = []
    all_starts: list[int] = []
    all_ends: list[int] = []

    for chrom, chrom_len in chromsizes.items():
        if chrom not in store.chromosomes:
            continue
        logger.debug(f"Tiling {chrom} ({chrom_len:,} bp)")

        chrom_arr = store.root[chrom]
        # shape: (n_valid_samples, chrom_len)
        cov = chrom_arr[valid_indices, :chrom_len].astype(np.float32)

        n_complete = chrom_len // tilesize
        trimmed = n_complete * tilesize

        # complete tiles: (n_valid, n_complete, tilesize) → mean → (n_valid, n_complete)
        tile_means = cov[:, :trimmed].reshape(len(valid_indices), n_complete, tilesize).mean(axis=2)

        tile_starts = np.arange(n_complete, dtype=np.int32) * tilesize
        tile_ends = tile_starts + tilesize

        if chrom_len % tilesize != 0:
            last_mean = cov[:, trimmed:].mean(axis=1, keepdims=True)
            tile_means = np.concatenate([tile_means, last_mean], axis=1)
            tile_starts = np.append(tile_starts, trimmed)
            tile_ends = np.append(tile_ends, chrom_len)

        n_tiles = tile_means.shape[1]

        # Apply blacklist: drop tiles overlapping blacklist
        if blacklist_pr is not None:
            tiles_pr = pr.PyRanges(pd.DataFrame({
                "Chromosome": chrom,
                "Start": tile_starts,
                "End": tile_ends,
            }))
            keep_pr = tiles_pr.subtract_overlaps(blacklist_pr, strand_behavior="ignore")
            keep_df = pd.DataFrame(keep_pr)
            if keep_df.empty:
                continue
            # Match kept tiles back to indices by start position
            keep_set = set(keep_df["Start"].tolist())
            keep_mask = np.array([s in keep_set for s in tile_starts])
            tile_means = tile_means[:, keep_mask]
            tile_starts = tile_starts[keep_mask]
            tile_ends = tile_ends[keep_mask]
            n_tiles = tile_means.shape[1]

        all_chroms.extend([chrom] * n_tiles)
        all_starts.extend(tile_starts.tolist())
        all_ends.extend(tile_ends.tolist())

        for i, s in enumerate(valid_samples):
            signals_by_sample[s].extend(tile_means[i].tolist())

    chroms_series = pd.Series(all_chroms, name="Chromosome")
    starts_series = pd.Series(all_starts, name="Start")
    ends_series = pd.Series(all_ends, name="End")

    # RPKM: mean_depth × 1e9 / library_size (tile-length-independent)
    signals_df = pd.DataFrame(signals_by_sample)
    rpkm_df = signals_df.multiply(1e9 / lib_sizes_arr, axis=1)
    log_rpkm_df = np.log1p(rpkm_df)
    log_rpkm_df.to_parquet(output_dir / f"log_rpkm_{tilesize}bp.parquet", index=False)
    logger.info(f"Saved log-RPKM tile signal to {output_dir / f'log_rpkm_{tilesize}bp.parquet'}")

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



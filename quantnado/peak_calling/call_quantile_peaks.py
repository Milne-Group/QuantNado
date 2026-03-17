from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyranges1 as pr
from loguru import logger
from skimage.filters import threshold_li


def _fit_gmm_threshold(values: np.ndarray) -> float:
    """Fit a 2-component Gaussian mixture model and return the decision boundary threshold.

    The threshold is the value between the noise (lower-mean) and signal (higher-mean)
    components where their posterior probabilities are equal, i.e. P(signal|x) = 0.5.

    Parameters
    ----------
    values:
        1-D array of signal values to fit (typically nonzero log-RPKM tiles).

    Returns
    -------
    float
        The threshold separating noise from signal.
    """
    from sklearn.mixture import GaussianMixture

    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=0, max_iter=300)
    gmm.fit(values.reshape(-1, 1))

    means = gmm.means_.flatten()
    noise_idx = int(np.argmin(means))
    signal_idx = 1 - noise_idx
    noise_mean, signal_mean = means[noise_idx], means[signal_idx]

    # Scan for the decision boundary where the signal component's posterior first exceeds 0.5.
    # Using 2000 points gives <0.1 % relative error for typical log-RPKM ranges.
    xs = np.linspace(noise_mean, signal_mean, 2000).reshape(-1, 1)
    signal_proba = gmm.predict_proba(xs)[:, signal_idx]

    crossing = np.searchsorted(signal_proba, 0.5)
    if crossing == 0:
        threshold = float(noise_mean)
    elif crossing >= len(xs):
        threshold = float(signal_mean)
    else:
        # Linear interpolation for sub-grid precision
        x0, x1 = float(xs[crossing - 1, 0]), float(xs[crossing, 0])
        p0, p1 = float(signal_proba[crossing - 1]), float(signal_proba[crossing])
        threshold = x0 + (0.5 - p0) * (x1 - x0) / (p1 - p0)

    return threshold


def _merge_adjacent_regions(
    peaks_df: pd.DataFrame,
    scores: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Merge overlapping or directly adjacent intervals within each chromosome.

    If *scores* is provided (one value per row of *peaks_df*, aligned after
    sorting), each merged region receives the sum of its constituent tile
    scores in a ``Score`` column.
    """
    if peaks_df.empty:
        return peaks_df

    peaks_df = peaks_df.sort_values(["Chromosome", "Start", "End"]).reset_index(drop=True)
    if scores is not None:
        scores = np.asarray(scores, dtype=np.float64)[peaks_df.index]
    merged_parts: list[pd.DataFrame] = []

    offset = 0
    for chrom, grp in peaks_df.groupby("Chromosome", sort=False):
        starts = grp["Start"].to_numpy(dtype=np.int64, copy=False)
        ends = grp["End"].to_numpy(dtype=np.int64, copy=False)
        n = len(starts)

        if n == 1:
            row: dict = {"Chromosome": [chrom], "Start": starts, "End": ends}
            if scores is not None:
                row["Score"] = [scores[offset]]
            merged_parts.append(pd.DataFrame(row))
            offset += n
            continue

        prev_max_end = np.maximum.accumulate(ends[:-1])
        new_group = np.empty(n, dtype=bool)
        new_group[0] = True
        new_group[1:] = starts[1:] > prev_max_end

        group_starts = np.flatnonzero(new_group)
        merged_starts = starts[group_starts]
        merged_ends = np.maximum.reduceat(ends, group_starts)

        row = {
            "Chromosome": np.repeat(chrom, len(merged_starts)),
            "Start": merged_starts,
            "End": merged_ends,
        }
        if scores is not None:
            row["Score"] = np.add.reduceat(scores[offset : offset + n], group_starts)
        offset += n

        merged_parts.append(pd.DataFrame(row))

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
    use_gmm: bool = False,
    use_li: bool = False,
) -> Optional[pr.PyRanges]:
    """Call peaks from a single sample's tile signal.

    By default uses a quantile threshold on all nonzero tiles. When *use_gmm*
    is True, a 2-component GMM is first fitted on the nonzero log-RPKM values
    to separate the noise component from the signal component; the quantile
    threshold is then applied only to tiles that survive the GMM noise filter.
    """
    logger.info(f"Calling peaks for sample: {signal.name}")

    nonzero = signal[signal > 0]
    if nonzero.empty:
        logger.warning(f"[{signal.name}] No nonzero signal values.")
        return None

    if quantile >= 1.0:
        logger.warning(f"[{signal.name}] quantile=1.0 means no tile can exceed the maximum; returning None.")
        return None

    if use_gmm:
        gmm_threshold = _fit_gmm_threshold(nonzero.to_numpy(dtype=np.float64))
        logger.debug(f"[{signal.name}] GMM noise threshold = {gmm_threshold:.4f}")
        signal_component = nonzero[nonzero > gmm_threshold]
        if signal_component.empty:
            logger.warning(f"[{signal.name}] No tiles above GMM noise threshold.")
            return None
        threshold = signal_component.quantile(quantile)
        logger.debug(
            f"[{signal.name}] GMM+quantile {quantile} threshold = {threshold:.4f}"
            f" (signal tiles: {len(signal_component):,})"
        )
    else:
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

    # AUC per tile = RPKM × tilesize (proportional to reads, library-size normalised).
    # Summing these over a merged peak gives the area under the signal curve.
    auc_tile = np.expm1(peaks_df["Score"].to_numpy(dtype=np.float64)) * tilesize
    if merge:
        peaks_df = _merge_adjacent_regions(peaks_df[["Chromosome", "Start", "End"]], scores=auc_tile)
    else:
        peaks_df = peaks_df[["Chromosome", "Start", "End"]].copy()
        peaks_df["Score"] = auc_tile

    # filter by li threshold on merged peaks if requested
    if use_li:
        li_threshold = threshold_li(peaks_df["Score"].to_numpy(dtype=np.float64))
        logger.debug(f"[{signal.name}] Li threshold on merged peaks = {li_threshold:.4f}")
        peaks_df = peaks_df[peaks_df["Score"] >= li_threshold]
        if peaks_df.empty:
            logger.warning(f"[{signal.name}] No merged peaks exceed Li threshold.")
            return None
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
    use_gmm: bool = True,
    use_li: bool = False,
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

    signals_by_sample: dict[str, list[float]] = {s: [] for s in valid_samples}
    all_chroms: list[str] = []
    all_starts: list[int] = []
    all_ends: list[int] = []

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
            keep_pairs = set(
                zip(
                    keep_df["Start"].to_numpy(dtype=np.int64),
                    keep_df["End"].to_numpy(dtype=np.int64),
                    strict=False,
                )
            )
            keep_mask = np.fromiter(
                (((int(s), int(e)) in keep_pairs) for s, e in zip(tile_starts, tile_ends, strict=False)),
                dtype=bool,
                count=len(tile_starts),
            )
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
            use_gmm=use_gmm,
            use_li=use_li,
        )

        if pr_obj is not None:
            output_bed = output_dir / f"{sample_name}.bed"
            pr_obj.to_bed(output_bed)
            logger.success(f"Peak BED saved to: {output_bed}")
            results.append(str(output_bed))
        else:
            logger.warning(f"[{sample_name}] No peaks detected.")

    return results



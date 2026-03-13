"""
SEACR peak calling implemented natively in Python.

Translates the logic of SEACR_1.3.sh + SEACR_1.3.R without calling any
external R/bash processes.  The pipeline is:

  1. Compute AUC islands from per-base coverage (equivalent to the awk in the
     shell script that groups contiguous non-zero bedgraph runs).
  2. Compute AUC thresholds via the pctremain curve (control file) or a simple
     top-fraction percentile (numeric FDR).  Includes the Lorenz-peak
     normalization and anti-spurious-peak correction from the R script.
  3. Filter islands by AUC > threshold and Num > d0.
  4. Merge nearby islands (gap < mean_island_length / 10).
  5. (Optional) Remove experiment islands that overlap control islands.
  6. (Optional) Subtract a genomic blacklist.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from math import sqrt, pi

import numpy as np
import pandas as pd
import pyranges1 as pr
from loguru import logger
from scipy import ndimage


# ---------------------------------------------------------------------------
# Island computation  (equivalent to the awk in SEACR_1.3.sh)
# ---------------------------------------------------------------------------


def _compute_islands(cov: np.ndarray, chrom: str) -> list[dict]:
    """Compute SEACR-style signal islands from a per-base coverage array.

    Each island is a maximal contiguous stretch of non-zero coverage.
    Returns a list of dicts with keys:
        Chromosome, Start, End, AUC, MaxSignal, MaxRegion, Num
    where Num = number of distinct value runs within the island (RLE length),
    matching the bedgraph-entry count produced by the shell script's awk.
    """
    labeled, n_islands = ndimage.label(cov > 0)
    if n_islands == 0:
        return []

    labels = np.arange(1, n_islands + 1)
    aucs = ndimage.sum(cov, labeled, labels)
    maxes = ndimage.maximum(cov, labeled, labels)
    slices = ndimage.find_objects(labeled)

    islands = []
    for idx, sl in enumerate(slices):
        istart = sl[0].start
        iend = sl[0].stop
        island_cov = cov[istart:iend]
        max_val = float(maxes[idx])

        # MaxRegion: span from first to last base with max signal (matching shell awk:
        # when equal-max positions exist, coord extends to cover all of them)
        max_pos = np.where(island_cov == max_val)[0]
        mr_start = istart + int(max_pos[0])
        mr_end = istart + int(max_pos[-1]) + 1
        coord = f"{chrom}:{mr_start}-{mr_end}"

        # Num = number of constant-value runs (RLE length)
        num = int(np.sum(np.diff(island_cov) != 0)) + 1

        islands.append(
            {
                "Chromosome": chrom,
                "Start": istart,
                "End": iend,
                "AUC": float(aucs[idx]),
                "MaxSignal": max_val,
                "MaxRegion": coord,
                "Num": num,
            }
        )

    return islands


# ---------------------------------------------------------------------------
# ECDF and pctremain  (core of SEACR_1.3.R)
# ---------------------------------------------------------------------------


def _ecdf(sorted_arr: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Fraction of sorted_arr values ≤ x (vectorised ECDF)."""
    return np.searchsorted(sorted_arr, x, side="right") / len(sorted_arr)


def _pctremain(x: np.ndarray, exp_sorted: np.ndarray, both_sorted: np.ndarray) -> np.ndarray:
    """
    Proportion of experiment peaks remaining above x relative to combined
    (experiment + control) peaks remaining above x.

    pctremain(x) = (n_exp - ecdf_exp(x)*n_exp) / (n_both - ecdf_both(x)*n_both)
    """
    n_exp = len(exp_sorted)
    n_both = len(both_sorted)
    num = n_exp * (1.0 - _ecdf(exp_sorted, x))
    denom = n_both * (1.0 - _ecdf(both_sorted, x))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 0, num / denom, np.nan)


# ---------------------------------------------------------------------------
# Lorenz-peak normalisation  (norm="yes" branch in SEACR_1.3.R)
# ---------------------------------------------------------------------------


def _lorenz_peak_value(vec: np.ndarray) -> float:
    """
    Identify the AUC value at the 'peak of the Lorenz deviation from diagonal'.

    Builds a (rank-fraction, normalised-signal-fraction) curve and finds the
    point of maximum perpendicular distance from the y = x diagonal — exactly
    as the R dist2d / expframe logic does.  Falls back to the 90th-percentile
    value if the curve peak lies below it.
    """
    sorted_asc = np.sort(vec)
    sorted_vals = sorted_asc[::-1]
    n = len(vec)
    count = np.linspace(1.0, 0.0, n)
    max_v = sorted_vals[0] if sorted_vals[0] > 0 else 1.0
    quant = sorted_vals / max_v

    diff = np.abs(count - quant)
    mask = diff > 0.9 * diff.max()

    sub_vals = sorted_vals[mask]
    sub_count = count[mask]
    sub_quant = quant[mask]

    # Distance from y=x diagonal is proportional to (count − quant)
    candidate = float(sub_vals[np.argmax(sub_count - sub_quant)])
    # R uses sort(vec)[as.integer(0.9*n)] with 1-based indexing → 0-based index floor(0.9*n)-1
    fallback = float(sorted_asc[int(0.9 * n) - 1])
    return candidate if candidate > fallback else fallback


def _nrd0_bandwidth(vec: np.ndarray) -> float:
    """R density() default bandwidth (nrd0) for a 1D sample."""
    vec = np.asarray(vec, dtype=float)
    n = len(vec)
    if n < 2:
        return 1.0

    sd = float(np.std(vec, ddof=1))
    q75, q25 = np.percentile(vec, [75, 25])
    iqr = float(q75 - q25)
    scale = min(sd, iqr / 1.34) if iqr > 0 else sd
    if scale <= 0:
        scale = abs(float(vec[0])) if float(vec[0]) != 0 else 1.0
    return float(0.9 * scale * (n ** (-0.2)))


def _mode_via_kde(vec: np.ndarray, cutoff: float) -> float:
    """Estimate the mode of vec[vec <= cutoff] to mirror R density().

    This uses the same defaults as R's density():
      - Gaussian kernel
      - bandwidth = nrd0
      - n = 512 grid points
      - grid range extended by cut * bw on both sides (cut=3)
    """
    sub = np.asarray(vec[vec <= cutoff], dtype=float)
    if len(sub) == 0:
        return float(cutoff)
    if len(sub) == 1:
        return float(sub[0])

    bw = _nrd0_bandwidth(sub)
    if bw <= 0:
        return float(np.median(sub))

    n_grid = 512
    cut = 3.0
    lo = float(sub.min()) - cut * bw
    hi = float(sub.max()) + cut * bw
    grid = np.linspace(lo, hi, n_grid)

    # Gaussian kernel density estimate with explicit bandwidth, matching R's density shape.
    u = (grid[:, None] - sub[None, :]) / bw
    y = np.exp(-0.5 * u * u).sum(axis=1) / (len(sub) * bw * sqrt(2.0 * pi))
    return float(grid[int(np.argmax(y))])


def _compute_norm_constant(expvec: np.ndarray, ctrlvec: np.ndarray) -> float:
    """Return the normalisation constant (exp_mode / ctrl_mode) à la SEACR."""
    exp_cutoff = _lorenz_peak_value(expvec)
    ctrl_cutoff = _lorenz_peak_value(ctrlvec)
    exp_mode = _mode_via_kde(expvec, exp_cutoff)
    ctrl_mode = _mode_via_kde(ctrlvec, ctrl_cutoff)
    if ctrl_mode == 0:
        logger.warning("Control mode is 0; skipping normalisation.")
        return 1.0
    return exp_mode / ctrl_mode


# ---------------------------------------------------------------------------
# Threshold finding  (pctremain curve logic from SEACR_1.3.R)
# ---------------------------------------------------------------------------


def _find_relaxed_threshold(
    x: np.ndarray,
    pr: np.ndarray,
    x0: float,
) -> float:
    """
    Find the relaxed (knee) threshold z0 below x0.

    Locates the midpoint of the pctremain curve between its minimum (≤ x0)
    and its value at x0, then finds the median of the interval above that
    midpoint — matching the z/z2/z0 logic in the R script.
    """
    z_mask = x <= x0
    z = x[z_mask]
    pr_z = pr[z_mask]

    if len(z) == 0:
        return x0

    # x0 is always in x (it was selected from x), so pr_z[-1] == pr at x0
    mid = (float(pr_z[-1]) + float(np.nanmin(pr_z))) / 2.0

    z2 = float(z[np.nanargmin(np.abs(pr_z - mid))])

    if x0 != z2:
        z_sub = z[z > z2]
        if len(z_sub) == 0:
            return x0
        target = float(z_sub.max()) - 0.5 * float(z_sub.max() - z_sub.min())
        return float(z_sub[np.argmin(np.abs(z_sub - target))])
    else:
        return x0


def _anti_spurious_check(
    x: np.ndarray,
    pr: np.ndarray,
    x0: float,
    z0: float,
) -> tuple[float, float]:
    """
    If a secondary pctremain peak is ≥ 95% of the primary peak, use it instead.

    Filters out near-zero diff transitions via a quantile of |d(pctremain)/dx|,
    then re-runs the x0/z0 search on the filtered set — matching the
    frame/a/a0/b/b0 block in the R script.
    """
    if len(x) < 2:
        return x0, z0

    pr_x = pr  # already computed by caller; reuse to avoid duplicate _pctremain call
    diff_pr = np.abs(np.diff(pr_x))

    nonzero = diff_pr[diff_pr != 0]
    if len(nonzero) == 0:
        return x0, z0

    # nonzero contains only positive values by construction, so quantile is always > 0
    q = float(np.quantile(nonzero, 0.99))

    a_mask = (diff_pr != 0) & (diff_pr < q)
    a = x[:-1][a_mask]
    pr_a = pr_x[:-1][a_mask]

    valid_a = (pr_a < 1) & ~np.isnan(pr_a)
    if not np.any(valid_a):
        return x0, z0

    a0 = float(a[valid_a][np.argmax(pr_a[valid_a])])
    # pr at a0 == nanmax of pr_a[valid_a] (a0 is the argmax)
    max_pr_a = float(np.nanmax(pr_a[valid_a]))

    # b0 = relaxed threshold on filtered subset
    b_mask = a <= a0
    b = a[b_mask]
    pr_b = pr_a[b_mask]

    if len(b) == 0:
        return x0, z0

    mid_b = (max_pr_a + float(np.nanmin(pr_b))) / 2.0
    b2 = float(b[np.nanargmin(np.abs(pr_b - mid_b))])

    if a0 != b2:
        b_sub = b[b > b2]
        if len(b_sub) > 0:
            target_b = float(b_sub.max()) - 0.5 * float(b_sub.max() - b_sub.min())
            b0 = float(b_sub[np.argmin(np.abs(b_sub - target_b))])
        else:
            b0 = a0
    else:
        b0 = a0

    max_pr_x = float(np.nanmax(pr_x[pr_x < 1])) if np.any(pr_x < 1) else 0.0

    if max_pr_x > 0 and max_pr_a / max_pr_x > 0.95:
        logger.debug(
            f"Anti-spurious: secondary peak ratio={max_pr_a / max_pr_x:.3f}; "
            f"replacing x0={x0:.4g}→{a0:.4g}, z0={z0:.4g}→{b0:.4g}"
        )
        return a0, b0

    return x0, z0


def _thresholds_from_control(
    exp_auc: np.ndarray,
    exp_num: np.ndarray,
    ctrl_auc: np.ndarray,
    ctrl_num: np.ndarray,
    norm: bool,
) -> tuple[float, float, float, float | None]:
    """
    Compute (x0_stringent, z0_relaxed, d0_num, norm_constant) from control data.

    Translates the is.na(numtest) branch of SEACR_1.3.R.
    """
    norm_constant: float | None = None

    if norm:
        norm_constant = _compute_norm_constant(exp_auc, ctrl_auc)
        ctrl_auc = ctrl_auc * norm_constant
        logger.debug(f"\nSEACR normalisation constant: {norm_constant:.4f}")

    both_auc = np.concatenate([exp_auc, ctrl_auc])
    exp_sorted = np.sort(exp_auc)
    both_sorted = np.sort(both_auc)

    def _pr(vals: np.ndarray) -> np.ndarray:
        return _pctremain(vals, exp_sorted, both_sorted)

    def _pick_first_tied(values: np.ndarray, metric: np.ndarray) -> float:
        """Pick first value among ties at metric maximum, matching R's x[which(...)][1]."""
        max_metric = np.nanmax(metric)
        idx = np.where(np.isclose(metric, max_metric, rtol=0.0, atol=1e-12))[0]
        return float(values[int(idx[0])])

    def _relaxed_from(primary: float, domain: np.ndarray) -> float:
        z = domain[domain <= primary]
        if len(z) == 0:
            return primary

        pr_primary = float(_pr(np.array([primary]))[0])
        pr_z = _pr(z)
        mid = (pr_primary + float(np.nanmin(pr_z))) / 2.0
        delta = np.abs(mid - pr_z)
        z2 = float(z[int(np.where(np.isclose(delta, np.nanmin(delta), rtol=0.0, atol=1e-12))[0][0])])

        if primary != z2:
            z_sub = z[z > z2]
            if len(z_sub) == 0:
                return primary
            target = float(z_sub.max()) - 0.5 * float(z_sub.max() - z_sub.min())
            d = np.abs(z_sub - target)
            return float(z_sub[int(np.where(np.isclose(d, np.nanmin(d), rtol=0.0, atol=1e-12))[0][0])])
        return primary

    x = np.unique(both_auc)
    pr_x = _pr(x)

    valid = (pr_x < 1.0) & ~np.isnan(pr_x)
    if not np.any(valid):
        x0 = float(x[-1])
    else:
        x_valid = x[valid]
        pr_valid = pr_x[valid]
        x0 = _pick_first_tied(x_valid, pr_valid)

    z0 = _relaxed_from(x0, x)

    # Anti-spurious branch — direct translation of SEACR_1.3.R frame/a/a0/b/b0 logic.
    if len(x) > 1:
        frame_thresh = x[:-1]
        frame_pct = pr_x[:-1]
        frame_diff = np.abs(np.diff(pr_x))
        frame_valid = ~np.isnan(frame_pct) & ~np.isnan(frame_diff)

        f_thresh = frame_thresh[frame_valid]
        f_diff = frame_diff[frame_valid]

        if len(f_thresh) > 0:
            i = 2
            output = 0.0
            test3 = 0.99
            while output == 0.0 and i < 12:
                test3 = float("0." + ("9" * i))
                output = float(np.quantile(f_diff, test3))
                i += 1

            q = float(np.quantile(f_diff, test3))
            a = f_thresh[(f_diff != 0) & (f_diff < q)]

            if len(a) > 0:
                pr_a = _pr(a)
                a_valid = (pr_a < 1.0) & ~np.isnan(pr_a)

                if np.any(a_valid):
                    a0 = _pick_first_tied(a[a_valid], pr_a[a_valid])
                    b = a[a <= a0]

                    if len(b) > 0:
                        pr_a0 = float(_pr(np.array([a0]))[0])
                        pr_b = _pr(b)
                        mid_b = (pr_a0 + float(np.nanmin(pr_b))) / 2.0
                        delta_b = np.abs(mid_b - pr_b)
                        b2 = float(
                            b[
                                int(
                                    np.where(
                                        np.isclose(
                                            delta_b,
                                            np.nanmin(delta_b),
                                            rtol=0.0,
                                            atol=1e-12,
                                        )
                                    )[0][0]
                                )
                            ]
                        )

                        if a0 != b2:
                            b_sub = b[b > b2]
                            if len(b_sub) > 0:
                                target_b = float(b_sub.max()) - 0.5 * float(b_sub.max() - b_sub.min())
                                d_b = np.abs(b_sub - target_b)
                                b0 = float(
                                    b_sub[
                                        int(
                                            np.where(
                                                np.isclose(
                                                    d_b,
                                                    np.nanmin(d_b),
                                                    rtol=0.0,
                                                    atol=1e-12,
                                                )
                                            )[0][0]
                                        )
                                    ]
                                )
                            else:
                                b0 = a0
                        else:
                            b0 = a0

                        max_pr_a = float(np.nanmax(pr_a[a_valid]))
                        x_valid2 = (pr_x < 1.0) & ~np.isnan(pr_x)
                        max_pr_x = float(np.nanmax(pr_x[x_valid2])) if np.any(x_valid2) else 0.0

                        if max_pr_x > 0 and (max_pr_a / max_pr_x) > 0.95:
                            x0 = a0
                            z0 = b0

    # d0: minimum num value where control enrichment exceeds experiment
    d_vals = np.unique(np.concatenate([exp_num, ctrl_num]))
    exp_num_sorted = np.sort(exp_num)
    ctrl_num_sorted = np.sort(ctrl_num)
    pr2 = 1.0 - (_ecdf(exp_num_sorted, d_vals) - _ecdf(ctrl_num_sorted, d_vals))
    d0 = float(np.min(d_vals[pr2 > 1])) if np.any(pr2 > 1) else 1.0

    logger.debug(
        f"SEACR thresholds — stringent AUC: {x0:.4g}, relaxed AUC: {z0:.4g}, num: {d0:.4g}"
    )
    return x0, z0, d0, norm_constant


def _thresholds_from_fdr(
    exp_auc: np.ndarray,
    exp_num: np.ndarray,
    fdr: float,
) -> tuple[float, float, float]:
    """
    Compute (x0, z0, d0) from a numeric FDR fraction.

    Translates the numeric branch of SEACR_1.3.R.
    x0 = minimum AUC in the top `fdr` fraction of AUC values.
    z0 = minimum Num in the top `fdr` fraction of Num values.
    d0 = 0 (no num filter).
    """
    x0 = float(np.quantile(exp_auc, 1.0 - fdr))
    z0 = float(np.quantile(exp_num, 1.0 - fdr))

    logger.debug(f"\nSEACR FDR={fdr} thresholds — stringent AUC: {x0:.4g}, relaxed num: {z0:.4g}")
    return x0, z0, 0.0


# ---------------------------------------------------------------------------
# Island merging  (the awk merging step in SEACR_1.3.sh)
# ---------------------------------------------------------------------------


def _merge_nearby_islands(df: pd.DataFrame, gap: float) -> pd.DataFrame:
    """
    Merge islands on the same chromosome whose starts are within *gap* of the
    previous island's end — summing AUC, taking max signal, extending coords.
    """
    if df.empty:
        return df

    chrom = df["Chromosome"].to_numpy()
    start = df["Start"].to_numpy(dtype=np.int64, copy=False)
    end = df["End"].to_numpy(dtype=np.int64, copy=False)
    auc = df["AUC"].to_numpy(dtype=np.float64, copy=False)
    max_signal = df["MaxSignal"].to_numpy(dtype=np.float64, copy=False)
    max_region = df["MaxRegion"].to_numpy(dtype=object, copy=False)

    prev_end = np.empty(len(df), dtype=np.float64)
    prev_end[0] = -gap - 1.0
    prev_end[1:] = end[:-1]

    chrom_change = np.empty(len(df), dtype=bool)
    chrom_change[0] = True
    chrom_change[1:] = chrom[1:] != chrom[:-1]
    new_group = chrom_change | (start >= (prev_end + gap))

    group_starts = np.flatnonzero(new_group)
    group_ends = np.r_[group_starts[1:], len(df)]

    merged_auc = np.add.reduceat(auc, group_starts)
    merged_max_signal = np.maximum.reduceat(max_signal, group_starts)
    merged_rows: list[dict] = []

    for group_idx, (gstart, gend) in enumerate(zip(group_starts, group_ends, strict=False)):
        group_max = merged_max_signal[group_idx]
        local_max_idx = np.flatnonzero(max_signal[gstart:gend] == group_max)
        first_max_idx = gstart + int(local_max_idx[0])
        last_max_idx = gstart + int(local_max_idx[-1])

        first_region = str(max_region[first_max_idx])
        if first_max_idx != last_max_idx:
            last_end = str(max_region[last_max_idx]).rsplit("-", 1)[1]
            merged_region = f"{first_region.rsplit('-', 1)[0]}-{last_end}"
        else:
            merged_region = first_region

        merged_rows.append(
            {
                "Chromosome": chrom[gstart],
                "Start": int(start[gstart]),
                "End": int(end[gend - 1]),
                "AUC": float(merged_auc[group_idx]),
                "MaxSignal": float(group_max),
                "MaxRegion": merged_region,
            }
        )

    return pd.DataFrame(merged_rows)


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------


def call_seacr_peaks_from_zarr(
    zarr_path: Path,
    output_dir: Path,
    control_zarr_path: Optional[Path] = None,
    fdr_threshold: float = 0.01,
    norm: str = "non",
    stringency: str = "stringent",
    blacklist_file: Optional[Path] = None,
) -> list[str]:
    """Call SEACR peaks from a QuantNado zarr coverage store (pure Python).

    Replicates SEACR_1.3.sh + SEACR_1.3.R without calling any external
    R/bash processes.  Raw read counts are used directly — SEACR operates on
    fragment coverage, not RPKM-normalised signal.

    Parameters
    ----------
    zarr_path:
        Path to the experimental QuantNado zarr store.
    output_dir:
        Directory where output BED files (one per sample) are written.
    control_zarr_path:
        Optional control (IgG) zarr store.  All completed samples are averaged
        into a single control signal used for empirical threshold estimation.
    fdr_threshold:
        Numeric FDR fraction (0-1) used when no control zarr is provided.
    norm:
        ``"norm"`` to normalise control signal to experimental before threshold
        estimation (via density-peak ratio); ``"non"`` to skip.
    stringency:
        ``"stringent"`` uses the peak of the pctremain curve (x0);
        ``"relaxed"`` uses the knee (z0).
    blacklist_file:
        Optional BED file; peaks overlapping blacklisted regions are removed
        from the final output.
    """
    from ..dataset.store_bam import BamStore

    zarr_path = Path(zarr_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if stringency not in ("stringent", "relaxed"):
        raise ValueError(f"stringency must be 'stringent' or 'relaxed', got {stringency!r}")
    if norm not in ("norm", "non"):
        raise ValueError(f"norm must be 'norm' or 'non', got {norm!r}")

    store = BamStore.open(zarr_path, read_only=True)
    chromsizes = {chrom: size for chrom, size in store.chromsizes.items() if "_" not in chrom}

    sample_names = store.sample_names
    completed = store.completed_mask
    valid_samples = [s for s, c in zip(sample_names, completed) if c]
    valid_indices = [i for i, c in enumerate(completed) if c]

    if not valid_samples:
        logger.error("No completed samples found in store.")
        return []

    logger.info(f"Found {len(valid_samples)} completed sample(s) in {zarr_path}")

    # -------------- load optional blacklist --------------
    blacklist_pr = None
    if blacklist_file and Path(blacklist_file).exists():
        logger.info(f"Loading blacklist: {blacklist_file}")
        blacklist_pr = pr.read_bed(str(blacklist_file))

    # -------------- load experimental islands --------------
    # islands_by_sample[sample] = list of island dicts
    islands_by_sample: dict[str, list[dict]] = {s: [] for s in valid_samples}

    for chrom, chrom_len in chromsizes.items():
        if chrom not in store.chromosomes:
            continue
        logger.debug(f"Computing islands for {chrom} ({chrom_len:,} bp)")
        chrom_arr = store.root[chrom]
        cov = chrom_arr[valid_indices, :chrom_len].astype(np.float32)

        for i, sample in enumerate(valid_samples):
            islands_by_sample[sample].extend(_compute_islands(cov[i], chrom))

    # -------------- load control islands --------------
    ctrl_auc_arr: np.ndarray | None = None
    ctrl_num_arr: np.ndarray | None = None
    ctrl_islands_df: pd.DataFrame | None = None

    if control_zarr_path is not None:
        ctrl_store = BamStore.open(Path(control_zarr_path), read_only=True)
        ctrl_samples = [s for s, c in zip(ctrl_store.sample_names, ctrl_store.completed_mask) if c]
        ctrl_indices = [i for i, c in enumerate(ctrl_store.completed_mask) if c]

        if not ctrl_samples:
            logger.warning("No completed samples in control zarr; falling back to FDR threshold.")
        else:
            ctrl_islands_all: list[dict] = []
            for chrom, chrom_len in chromsizes.items():
                if chrom not in ctrl_store.chromosomes:
                    continue
                ctrl_arr = ctrl_store.root[chrom]
                ctrl_cov = ctrl_arr[ctrl_indices, :chrom_len].astype(np.float32)
                # Average across control samples before island-calling
                ctrl_islands_all.extend(_compute_islands(ctrl_cov.mean(axis=0), chrom))

            if ctrl_islands_all:
                ctrl_islands_df = pd.DataFrame(ctrl_islands_all)
                ctrl_auc_arr = ctrl_islands_df["AUC"].to_numpy()
                ctrl_num_arr = ctrl_islands_df["Num"].to_numpy()
                logger.info(f"Control: {len(ctrl_islands_df):,} islands")

    # -------------- per-sample peak calling --------------
    results: list[str] = []

    for sample in valid_samples:
        islands = islands_by_sample[sample]
        if not islands:
            logger.warning(f"[{sample}] No islands found.")
            continue

        islands_df = pd.DataFrame(islands)
        exp_auc = islands_df["AUC"].to_numpy()
        exp_num = islands_df["Num"].to_numpy()

        logger.info(f"[{sample}] {len(islands_df):,} islands before filtering")

        # Determine thresholds
        if ctrl_auc_arr is not None and ctrl_num_arr is not None:
            x0, z0, d0, norm_constant = _thresholds_from_control(
                exp_auc,
                exp_num,
                ctrl_auc_arr,
                ctrl_num_arr,
                norm=(norm == "norm"),
            )
            # Normalize control AUC for filtering (matches shell awk: the control bed
            # is multiplied by the norm constant *before* the AUC > thresh filter)
            ctrl_auc_filt = ctrl_auc_arr * norm_constant if norm_constant is not None else ctrl_auc_arr
        else:
            x0, z0, d0 = _thresholds_from_fdr(exp_auc, exp_num, fdr_threshold)
            ctrl_auc_filt = None

        auc_thresh = x0 if stringency == "stringent" else z0

        # Filter islands
        keep = (islands_df["AUC"] > auc_thresh) & (islands_df["Num"] > d0)
        filtered = islands_df[keep].copy()
        logger.info(
            f"[{sample}] {len(filtered):,} islands after threshold (AUC>{auc_thresh:.4g}, Num>{d0:.4g})"
        )

        if filtered.empty:
            logger.warning(f"[{sample}] No peaks exceed threshold.")
            continue

        # Merge nearby islands (gap = mean_island_length / 10)
        mean_len = float((filtered["End"] - filtered["Start"]).mean())
        gap = mean_len / 10.0
        merged = _merge_nearby_islands(
            filtered.sort_values(["Chromosome", "Start"]).reset_index(drop=True),
            gap,
        )
        logger.debug(f"[{sample}] {len(merged):,} islands after merging (gap={gap:.1f} bp)")

        # Remove experiment islands overlapping control islands (control-enriched blacklist)
        if ctrl_islands_df is not None and ctrl_auc_filt is not None:
            # Filter using (possibly normalised) ctrl AUC against the stringent threshold,
            # matching the shell script which normalises the control bed before applying awk.
            ctrl_keep = ctrl_auc_filt > x0
            ctrl_filtered = ctrl_islands_df[ctrl_keep]
            if not ctrl_filtered.empty:
                exp_pr = pr.PyRanges(
                    merged[["Chromosome", "Start", "End", "AUC", "MaxSignal", "MaxRegion"]]
                    .astype({"Start": int, "End": int})
                )
                ctrl_pr = pr.PyRanges(
                    ctrl_filtered[["Chromosome", "Start", "End"]].astype({"Start": int, "End": int})
                )
                # Match bedtools intersect -wa -v semantics from SEACR_1.3.sh:
                # keep only experiment peaks with no overlap to control peaks.
                exp_pr = exp_pr.overlap(ctrl_pr, strand_behavior="ignore", invert=True)
                merged = pd.DataFrame(exp_pr)
                logger.debug(f"[{sample}] {len(merged):,} islands after control subtraction")

        if merged.empty:
            logger.warning(f"[{sample}] No peaks remain after control subtraction.")
            continue

        # Optional genomic blacklist subtraction
        if blacklist_pr is not None:
            peaks_pr = pr.PyRanges(
                merged[["Chromosome", "Start", "End"]].astype({"Start": int, "End": int})
            )
            peaks_pr = peaks_pr.subtract_overlaps(blacklist_pr, strand_behavior="ignore")
            merged = pd.DataFrame(peaks_pr)
            logger.debug(f"[{sample}] {len(merged):,} peaks after blacklist subtraction")

        if merged.empty:
            logger.warning(f"[{sample}] No peaks remain after blacklist subtraction.")
            continue

        # Write BED: chr start end AUC max_signal max_region
        out_cols = ["Chromosome", "Start", "End", "AUC", "MaxSignal", "MaxRegion"]
        out_df = merged[[c for c in out_cols if c in merged.columns]].copy()
        # Match awk's default numeric formatting used by SEACR_1.3.sh
        # (OFMT/CONVFMT = "%.6g"): 6 significant digits, no fixed decimals.
        out_df["AUC"] = out_df["AUC"].map(lambda x: format(float(x), ".6g"))
        out_df["MaxSignal"] = out_df["MaxSignal"].map(lambda x: format(float(x), ".6g"))

        output_bed = output_dir / f"{sample}.{stringency}.bed"
        out_df.to_csv(output_bed, sep="\t", header=False, index=False)

        logger.success(f"[{sample}] {len(out_df):,} peaks → {output_bed}")
        results.append(str(output_bed))

    return results

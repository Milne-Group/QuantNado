"""LanceOtron peak calling via a PyTorch port of the wide-and-deep v5.03 Keras model.

Pipeline per sample
-------------------
1. Load per-bp uint32 coverage via QuantNadoDataset.extract_region()
2. RPKM normalise: cov * 1e9 / total_reads
3. 400 bp rolling mean → candidate peak detection (dynamic threshold)
4. Feature extraction:
   - Deep:  2000 bp window centred on summit (n_peaks, 2000)
   - Wide:  max_height + 10 Poisson -log10(p) at 10–100 kb scales + seq_depth (n_peaks, 12)
5. Feature normalisation (pure numpy, no sklearn):
   - Deep step 1: per-sample z-score across peaks (fit on current data)
   - Deep step 2: (X - deep_mean) / deep_scale  from pre-fitted .npy arrays
   - Wide:        (X - wide_mean) / wide_scale   from pre-fitted .npy arrays
6. Batched PyTorch inference → overall / shape / enrichment scores
7. Filter by overall_score > threshold, write BED

Static assets (committed, pre-converted from Keras v5.03 weights)
    quantnado/peak_calling/static/lanceotron/
        lanceotron_v5_03.pt        – PyTorch state_dict
        wide_scaler_mean.npy       – (12,)
        wide_scaler_scale.npy      – (12,)
        deep_scaler_mean.npy       – (2000,)
        deep_scaler_scale.npy      – (2000,)

Use scripts/setup_lanceotron_weights.py to (re)generate these files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyranges1 as pr
from loguru import logger
from scipy import ndimage
from scipy.stats import poisson

# ---------------------------------------------------------------------------
# Constants matching original LanceOtron hyper-parameters
# ---------------------------------------------------------------------------

SMOOTH_WINDOW = 400          # bp rolling mean for candidate detection
DEEP_WINDOW = 2000           # bp window centred on summit
LAMBDA_COV = 100_000         # total bp range for Poisson lambda steps
N_LAMBDA_STEPS = 10          # → 10 kb, 20 kb, ..., 100 kb
INITIAL_THRESHOLD_FACTOR = 4 # candidate threshold = chrom_mean * this
MIN_PEAK_WIDTH = 50          # bp
MAX_PEAK_WIDTH = 2000        # bp

STATIC_DIR = Path(__file__).parent / "static" / "lanceotron"

# ---------------------------------------------------------------------------
# PyTorch model (port of wide_and_deep_fully_trained_v5_03.h5)
# ---------------------------------------------------------------------------


def _build_model():
    """Construct the LanceOtronModel. Requires torch."""
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:
        raise ImportError(
            "LanceOtron peak calling requires PyTorch. "
            "Install with:  pip install quantnado[lanceotron]"
        ) from exc

    class _SamePadConv1d(nn.Module):
        """Conv1d with Keras-style 'same' padding (asymmetric for even kernels)."""

        def __init__(self, in_ch: int, out_ch: int, kernel_size: int) -> None:
            super().__init__()
            self.kernel_size = kernel_size
            self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=0)

        def forward(self, x):
            pad_total = self.kernel_size - 1
            pad_l = pad_total // 2
            pad_r = pad_total - pad_l
            x = F.pad(x, (pad_l, pad_r))
            return self.conv(x)

    class _ConvBlock(nn.Module):
        def __init__(self, in_ch: int, out_ch: int, kernel_size: int) -> None:
            super().__init__()
            self.conv = _SamePadConv1d(in_ch, out_ch, kernel_size)
            # eps=0.001 matches Keras BatchNormalization default
            self.bn = nn.BatchNorm1d(out_ch, eps=0.001)
            self.act = nn.LeakyReLU(negative_slope=0.3)

        def forward(self, x):
            return self.act(self.bn(self.conv(x)))

    class LanceOtronModel(nn.Module):
        """Wide-and-deep 1D CNN peak classifier (PyTorch port of Keras v5.03)."""

        def __init__(self) -> None:
            super().__init__()
            # ── Deep path ────────────────────────────────────────────────
            # Input: (batch, 1, 2000)
            self.deep_entry = _ConvBlock(1, 70, 9)            # → (batch, 70, 2000)
            self.deep_blocks = nn.ModuleList([
                _ConvBlock(70, 120, 6),                       # block 0: 70→120
                _ConvBlock(120, 120, 6),                      # blocks 1-3: 120→120
                _ConvBlock(120, 120, 6),
                _ConvBlock(120, 120, 6),
            ])
            self.pool = nn.MaxPool1d(2)
            # After 4 pools: (batch, 120, 125)
            # Dense(10) on channel axis: project 120 → 10 per position
            self.deep_dense = nn.Linear(120, 10)
            self.deep_dense_bn = nn.BatchNorm1d(10, eps=0.001)
            self.deep_dense_act = nn.LeakyReLU(0.3)
            self.dropout = nn.Dropout(0.5)
            # Flatten → (batch, 1250)

            # ── Output heads ─────────────────────────────────────────────
            # shape_output: deep features → 2 classes
            self.shape_out = nn.Linear(1250, 2)
            # pvalue_output: wide features (12,) → 2 classes
            self.pvalue_out = nn.Linear(12, 2)
            # overall: concat(wide=12, deep=1250, pvalue=2) = 1264 → 2 classes
            self.combined1 = nn.Linear(1264, 70)
            self.combined1_bn = nn.BatchNorm1d(70, eps=0.001)
            self.combined1_act = nn.LeakyReLU(0.3)
            self.combined2 = nn.Linear(70, 70)
            self.combined2_bn = nn.BatchNorm1d(70, eps=0.001)
            self.combined2_act = nn.LeakyReLU(0.3)
            self.overall_out = nn.Linear(70, 2)

        def _deep_forward(self, x_deep):
            """x_deep: (batch, 2000) → deep_features: (batch, 1250)"""
            x = x_deep.unsqueeze(1)         # (batch, 1, 2000)
            x = self.deep_entry(x)           # (batch, 70, 2000)
            for block in self.deep_blocks:
                x = block(x)                 # (batch, 120, L)
                x = self.pool(x)             # (batch, 120, L/2)
            # x: (batch, 120, 125)
            # Dense(10) on channel dim — permute so Linear sees (120,) last
            x = x.permute(0, 2, 1)          # (batch, 125, 120)
            x = self.deep_dense(x)           # (batch, 125, 10)
            # BN1d(10) expects (batch, 10, length)
            x = x.permute(0, 2, 1)          # (batch, 10, 125)
            x = self.deep_dense_bn(x)
            x = self.deep_dense_act(x)
            x = self.dropout(x)
            x = x.flatten(1)                 # (batch, 1250)
            return x

        def forward(self, x_deep, x_wide):
            """
            x_deep: (batch, 2000)  float32
            x_wide: (batch, 12)    float32
            Returns tuple: (overall, shape, enrichment) each (batch, 2) softmax
            """
            deep_feat = self._deep_forward(x_deep)       # (batch, 1250)
            shape_logits = self.shape_out(deep_feat)      # (batch, 2)
            pvalue_logits = self.pvalue_out(x_wide)       # (batch, 2)

            concat = torch.cat([x_wide, deep_feat, pvalue_logits], dim=1)  # (batch, 1264)
            h = self.combined1_act(self.combined1_bn(self.combined1(concat)))
            h = self.combined2_act(self.combined2_bn(self.combined2(h)))
            overall_logits = self.overall_out(h)          # (batch, 2)

            return (
                torch.softmax(overall_logits, dim=1),
                torch.softmax(shape_logits, dim=1),
                torch.softmax(pvalue_logits, dim=1),
            )

    return LanceOtronModel


# ---------------------------------------------------------------------------
# Weight / scaler loading
# ---------------------------------------------------------------------------


def _check_static_assets() -> None:
    """Raise a clear error if pre-converted model assets are missing."""
    required = [
        "lanceotron_v5_03.pt",
        "wide_scaler_mean.npy", "wide_scaler_scale.npy",
        "deep_scaler_mean.npy", "deep_scaler_scale.npy",
    ]
    missing = [f for f in required if not (STATIC_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"LanceOtron static assets missing from {STATIC_DIR}:\n"
            + "\n".join(f"  {f}" for f in missing)
            + "\nRun scripts/setup_lanceotron_weights.py to generate them."
        )


def _load_scalers() -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
]:
    """Return (wide_mean, wide_scale), (deep_mean, deep_scale)."""
    wide_mean = np.load(STATIC_DIR / "wide_scaler_mean.npy")
    wide_scale = np.load(STATIC_DIR / "wide_scaler_scale.npy")
    deep_mean = np.load(STATIC_DIR / "deep_scaler_mean.npy")
    deep_scale = np.load(STATIC_DIR / "deep_scaler_scale.npy")
    return (wide_mean, wide_scale), (deep_mean, deep_scale)


def _load_model():
    """Build LanceOtronModel and load pre-trained weights."""
    import torch

    _check_static_assets()
    ModelClass = _build_model()
    model = ModelClass()
    state = torch.load(STATIC_DIR / "lanceotron_v5_03.pt", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Signal preprocessing
# ---------------------------------------------------------------------------


def _rpkm_normalise(cov: np.ndarray, total_reads: int) -> np.ndarray:
    """Convert uint32 per-bp counts to RPKM-equivalent float32.

    Equivalent to bamCoverage --normaliseUsing RPKM at 1 bp resolution:
        normalised = cov / (total_reads / 1e9)
    """
    if total_reads == 0:
        return cov.astype(np.float32)
    return (cov.astype(np.float64) * 1e9 / total_reads).astype(np.float32)


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Fast uniform rolling mean using scipy.ndimage."""
    return ndimage.uniform_filter1d(arr.astype(np.float32), size=window, mode="nearest")


# ---------------------------------------------------------------------------
# Candidate peak detection (LanceOtron dynamic threshold recursion)
# ---------------------------------------------------------------------------


def _find_candidate_peaks(
    smooth_cov: np.ndarray,
    chrom_mean: float,
    threshold_factor: float = INITIAL_THRESHOLD_FACTOR,
) -> list[tuple[int, int]]:
    """Find enriched regions using LanceOtron's recursive threshold algorithm.

    Regions wider than MAX_PEAK_WIDTH are recursively split with a higher
    threshold until they fit or fall below MIN_PEAK_WIDTH.
    """
    threshold = chrom_mean * threshold_factor
    if threshold <= 0:
        return []

    labeled, n = ndimage.label(smooth_cov > threshold)
    if n == 0:
        return []

    candidates: list[tuple[int, int]] = []
    slices = ndimage.find_objects(labeled)
    for sl in slices:
        start = sl[0].start
        end = sl[0].stop
        width = end - start
        if width < MIN_PEAK_WIDTH:
            continue
        if width > MAX_PEAK_WIDTH:
            # Recurse with a higher threshold to split this region
            sub = _find_candidate_peaks(
                smooth_cov[start:end],
                chrom_mean,
                threshold_factor=threshold_factor * 1.5,
            )
            candidates.extend((start + s, start + e) for s, e in sub)
        else:
            candidates.append((start, end))

    return candidates


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def _extract_features(
    norm_cov: np.ndarray,
    candidates: list[tuple[int, int]],
    seq_depth: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract deep (2000 bp) and wide (12) features for candidate peaks.

    Parameters
    ----------
    norm_cov:
        RPKM-normalised per-bp coverage for the chromosome.
    candidates:
        List of (start, end) coordinate pairs.
    seq_depth:
        total_reads / 1e9 (used as the final wide feature).

    Returns
    -------
    X_deep : (n_peaks, 2000)
    X_wide : (n_peaks, 12)
        [max_height, -log10p(10kb), ..., -log10p(100kb), seq_depth]
    """
    n = len(candidates)
    half_deep = DEEP_WINDOW // 2
    chrom_len = len(norm_cov)

    X_deep = np.zeros((n, DEEP_WINDOW), dtype=np.float32)
    X_wide = np.zeros((n, N_LAMBDA_STEPS + 2), dtype=np.float32)  # +2: max_h + seq_depth

    lambda_step = LAMBDA_COV // N_LAMBDA_STEPS  # 10_000
    lambda_lengths = [lambda_step * (k + 1) for k in range(N_LAMBDA_STEPS)]

    for i, (start, end) in enumerate(candidates):
        summit = (start + end) // 2
        max_height = float(norm_cov[start:end].max()) if end > start else 0.0

        # ── Deep window ──────────────────────────────────────────────────
        d_start = max(0, summit - half_deep)
        d_end = d_start + DEEP_WINDOW
        if d_end > chrom_len:
            d_end = chrom_len
            d_start = max(0, d_end - DEEP_WINDOW)
        actual = d_end - d_start
        X_deep[i, :actual] = norm_cov[d_start:d_end]

        # ── Wide features ─────────────────────────────────────────────────
        X_wide[i, 0] = max_height

        for j, length in enumerate(lambda_lengths):
            pad = (LAMBDA_COV - length) // 2
            w_start = max(0, summit - LAMBDA_COV // 2 + pad)
            w_end = min(chrom_len, summit + LAMBDA_COV // 2 - pad)
            region = norm_cov[w_start:w_end]
            lam = float(region.mean()) if len(region) > 0 else 1e-9
            if lam <= 0:
                lam = 1e-9
            # -log10(1 - CDF(max_height; lambda)) = Poisson enrichment
            p_val = -np.log10(max(1.0 - poisson.cdf(max_height, lam), 1e-300))
            X_wide[i, j + 1] = float(p_val)

        X_wide[i, -1] = seq_depth

    return X_deep, X_wide


# ---------------------------------------------------------------------------
# Feature normalisation — pure numpy, no sklearn
# ---------------------------------------------------------------------------


def _normalise_features(
    X_deep: np.ndarray,
    X_wide: np.ndarray,
    wide_mean: np.ndarray,
    wide_scale: np.ndarray,
    deep_mean: np.ndarray,
    deep_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Two-pass normalisation matching the original LanceOtron preprocessing.

    Deep step 1 (per-sample z-score):
        Transpose → z-score across peaks → transpose back.
        Equivalent to sklearn StandardScaler().fit_transform(X_deep.T).T
    Deep step 2 (pre-trained offset):
        (X - deep_mean) / deep_scale
    Wide:
        (X_wide - wide_mean) / wide_scale
    """
    # ── Deep step 1: per-sample z-score ──────────────────────────────────
    T = X_deep.T.astype(np.float64)           # (2000, n_peaks)
    col_mean = T.mean(axis=0)                  # (n_peaks,)
    col_std = T.std(axis=0) + 1e-8
    T_norm = (T - col_mean) / col_std
    X_deep_norm = T_norm.T.astype(np.float32)  # (n_peaks, 2000)

    # ── Deep step 2: pre-trained scaler ──────────────────────────────────
    X_deep_norm = ((X_deep_norm - deep_mean) / (deep_scale + 1e-8)).astype(np.float32)

    # ── Wide ─────────────────────────────────────────────────────────────
    X_wide_norm = ((X_wide.astype(np.float64) - wide_mean) / (wide_scale + 1e-8)).astype(np.float32)

    return X_deep_norm, X_wide_norm


# ---------------------------------------------------------------------------
# Batched inference
# ---------------------------------------------------------------------------


def _score_candidates(
    X_deep: np.ndarray,
    X_wide: np.ndarray,
    model,
    batch_size: int = 512,
) -> np.ndarray:
    """Run batched inference. Returns (n_peaks, 3): overall, shape, enrichment scores.

    Each score is the class-0 softmax probability (higher = more peak-like).
    """
    import torch

    n = len(X_deep)
    scores = np.zeros((n, 3), dtype=np.float32)

    with torch.no_grad():
        for i in range(0, n, batch_size):
            d = torch.from_numpy(X_deep[i : i + batch_size])
            w = torch.from_numpy(X_wide[i : i + batch_size])
            overall, shape, enrich = model(d, w)
            scores[i : i + batch_size, 0] = overall[:, 0].cpu().numpy()
            scores[i : i + batch_size, 1] = shape[:, 0].cpu().numpy()
            scores[i : i + batch_size, 2] = enrich[:, 0].cpu().numpy()

    return scores


# ---------------------------------------------------------------------------
# Per-chromosome processing helper
# ---------------------------------------------------------------------------


def _process_chromosome(
    norm_cov: np.ndarray,
    chrom: str,
    seq_depth: float,
    model,
    wide_scaler: tuple[np.ndarray, np.ndarray],
    deep_scaler: tuple[np.ndarray, np.ndarray],
    score_threshold: float,
    smooth_window: int,
    initial_threshold_factor: float,
    batch_size: int,
) -> list[dict]:
    """Run the full LanceOtron pipeline on one chromosome's coverage array."""
    smooth = _rolling_mean(norm_cov, smooth_window)
    chrom_mean = float(smooth.mean())
    if chrom_mean == 0:
        logger.debug(f"  [{chrom}] skipping — zero coverage")
        return []

    candidates = _find_candidate_peaks(smooth, chrom_mean, initial_threshold_factor)
    if not candidates:
        logger.debug(f"  [{chrom}] no candidates above threshold")
        return []

    logger.debug(f"  [{chrom}] {len(candidates)} candidates")

    X_deep, X_wide = _extract_features(norm_cov, candidates, seq_depth)
    X_deep_n, X_wide_n = _normalise_features(
        X_deep, X_wide,
        wide_mean=wide_scaler[0], wide_scale=wide_scaler[1],
        deep_mean=deep_scaler[0], deep_scale=deep_scaler[1],
    )
    scores = _score_candidates(X_deep_n, X_wide_n, model, batch_size)

    peaks = []
    for (start, end), score_row in zip(candidates, scores):
        if score_row[0] >= score_threshold:
            peaks.append({
                "Chromosome": chrom,
                "Start": start,
                "End": end,
                "overall_score": float(score_row[0]),
                "shape_score": float(score_row[1]),
                "enrichment_score": float(score_row[2]),
            })

    logger.debug(f"  [{chrom}] {len(peaks)} peaks after score filter")
    return peaks


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def call_lanceotron_peaks_from_zarr(
    zarr_path: Path,
    output_dir: Path,
    score_threshold: float = 0.5,
    blacklist_file: Optional[Path] = None,
    smooth_window: int = SMOOTH_WINDOW,
    initial_threshold_factor: float = INITIAL_THRESHOLD_FACTOR,
    batch_size: int = 512,
) -> list[str]:
    """Call LanceOtron peaks from a QuantNado zarr coverage store.

    Parameters
    ----------
    zarr_path:
        Path to the QuantNado Zarr store.
    output_dir:
        Directory where BED files (one per sample) are written.
    score_threshold:
        Minimum overall_classification score to retain a peak (0–1).
    blacklist_file:
        Optional BED file; peaks overlapping blacklisted regions are removed.
    smooth_window:
        Rolling mean window size in bp (default 400).
    initial_threshold_factor:
        Candidate regions must exceed chrom_mean * this (default 4).
    batch_size:
        Inference batch size (default 512).

    Returns
    -------
    list[str]
        Paths of output BED files written.
    """
    from ..analysis.core import QuantNadoDataset
    from ..analysis.normalise import get_library_sizes

    zarr_path = Path(zarr_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── load assets ──────────────────────────────────────────────────────
    logger.info("Loading LanceOtron model and scaler assets …")
    model = _load_model()
    wide_scaler, deep_scaler = _load_scalers()

    # ── open dataset ─────────────────────────────────────────────────────
    ds = QuantNadoDataset(zarr_path)
    library_sizes = get_library_sizes(ds)

    valid_samples = [s for s, c in zip(ds.sample_names, ds.completed_mask) if c]
    if not valid_samples:
        logger.error("No completed samples in store.")
        return []
    logger.info(f"Processing {len(valid_samples)} sample(s)")

    # ── optional blacklist ────────────────────────────────────────────────
    blacklist_pr = None
    if blacklist_file and Path(blacklist_file).exists():
        blacklist_pr = pr.read_bed(str(blacklist_file))
        logger.info(f"Loaded blacklist: {blacklist_file}")

    # ── skip alt contigs ─────────────────────────────────────────────────
    chromosomes = [c for c in ds.chromosomes if "_" not in c]

    output_paths: list[str] = []

    for sample in valid_samples:
        logger.info(f"Sample: {sample}")
        total_reads = int(library_sizes[sample])
        seq_depth = total_reads / 1e9

        all_peaks: list[dict] = []

        for chrom in chromosomes:
            chrom_len = ds.chromsizes.get(chrom, 0)
            if chrom_len == 0:
                continue

            # Raw coverage: uint32 array shape (1, chrom_len) → squeeze
            raw = ds.extract_region(chrom=chrom, samples=[sample], as_xarray=False)
            cov = raw[0]  # (chrom_len,)

            norm_cov = _rpkm_normalise(cov, total_reads)

            peaks = _process_chromosome(
                norm_cov=norm_cov,
                chrom=chrom,
                seq_depth=seq_depth,
                model=model,
                wide_scaler=wide_scaler,
                deep_scaler=deep_scaler,
                score_threshold=score_threshold,
                smooth_window=smooth_window,
                initial_threshold_factor=initial_threshold_factor,
                batch_size=batch_size,
            )
            all_peaks.extend(peaks)

        if not all_peaks:
            logger.warning(f"[{sample}] No peaks called.")
            continue

        peaks_df = pd.DataFrame(all_peaks).sort_values(["Chromosome", "Start"]).reset_index(drop=True)

        # ── blacklist filter ──────────────────────────────────────────────
        if blacklist_pr is not None and not peaks_df.empty:
            peaks_pr = pr.PyRanges(peaks_df[["Chromosome", "Start", "End"]])
            clean = peaks_pr.subtract(blacklist_pr)
            if clean is not None and len(clean) > 0:
                clean_df = pd.DataFrame(clean)
                peaks_df = peaks_df.merge(
                    clean_df[["Chromosome", "Start", "End"]],
                    on=["Chromosome", "Start", "End"],
                    how="inner",
                )

        out_path = output_dir / f"{sample}_lanceotron_peaks.bed"
        peaks_df.to_csv(out_path, sep="\t", index=False, header=False)
        logger.success(f"[{sample}] {len(peaks_df)} peaks → {out_path}")
        output_paths.append(str(out_path))

    return output_paths

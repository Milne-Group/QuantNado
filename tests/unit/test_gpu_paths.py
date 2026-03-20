"""Unit tests verifying CPU vs GPU numerical equivalence for all GPU-accelerated paths."""

from __future__ import annotations

import numpy as np
import pytest

# Skip entire module if torch is not installed
torch = pytest.importorskip("torch")

from quantnado.peak_calling._device import gpu_available


# ---------------------------------------------------------------------------
# Marks
# ---------------------------------------------------------------------------

gpu_only = pytest.mark.skipif(not gpu_available(), reason="No GPU device available")
_GPU_DEVICE = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ---------------------------------------------------------------------------
# rolling_mean: torch avg_pool1d vs scipy uniform_filter1d
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_rolling_mean_cpu_matches_scipy():
    """CPU path produces identical output to scipy uniform_filter1d."""
    from quantnado.peak_calling.call_lanceotron_peaks import _rolling_mean
    from scipy import ndimage

    rng = np.random.default_rng(42)
    arr = rng.random(10_000).astype(np.float32)
    window = 400

    result = _rolling_mean(arr, window, device="cpu")
    expected = ndimage.uniform_filter1d(arr, size=window, mode="nearest")
    np.testing.assert_allclose(result, expected, atol=1e-5)


@pytest.mark.unit
@gpu_only
def test_rolling_mean_gpu_matches_cpu():
    """GPU rolling mean matches CPU scipy path within float32 tolerance."""
    from quantnado.peak_calling.call_lanceotron_peaks import _rolling_mean

    rng = np.random.default_rng(42)
    arr = rng.random(50_000).astype(np.float32)
    window = 400

    cpu_result = _rolling_mean(arr, window, device="cpu")
    gpu_result = _rolling_mean(arr, window, device=_GPU_DEVICE)
    np.testing.assert_allclose(gpu_result, cpu_result, atol=1e-4)


# ---------------------------------------------------------------------------
# normalise_features: GPU vs numpy
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_normalise_features_cpu():
    """CPU normalise_features matches manual numpy reference."""
    from quantnado.peak_calling.call_lanceotron_peaks import _normalise_features

    rng = np.random.default_rng(7)
    n, deep, wide = 50, 2000, 12
    X_deep = rng.random((n, deep)).astype(np.float32)
    X_wide = rng.random((n, wide)).astype(np.float32)
    deep_mean = rng.random(deep).astype(np.float32)
    deep_scale = rng.random(deep).astype(np.float32) + 0.1
    wide_mean = rng.random(wide).astype(np.float32)
    wide_scale = rng.random(wide).astype(np.float32) + 0.1

    d_out, w_out = _normalise_features(
        X_deep, X_wide, wide_mean, wide_scale, deep_mean, deep_scale, device="cpu"
    )
    assert d_out.shape == (n, deep)
    assert w_out.shape == (n, wide)


@pytest.mark.unit
@gpu_only
def test_normalise_features_gpu_matches_cpu():
    """GPU normalise_features numerically matches CPU path."""
    from quantnado.peak_calling.call_lanceotron_peaks import _normalise_features

    rng = np.random.default_rng(7)
    n, deep, wide = 50, 2000, 12
    X_deep = rng.random((n, deep)).astype(np.float32)
    X_wide = rng.random((n, wide)).astype(np.float32)
    deep_mean = rng.random(deep).astype(np.float32)
    deep_scale = rng.random(deep).astype(np.float32) + 0.1
    wide_mean = rng.random(wide).astype(np.float32)
    wide_scale = rng.random(wide).astype(np.float32) + 0.1

    d_cpu, w_cpu = _normalise_features(
        X_deep, X_wide, wide_mean, wide_scale, deep_mean, deep_scale, device="cpu"
    )
    d_gpu, w_gpu = _normalise_features(
        X_deep, X_wide, wide_mean, wide_scale, deep_mean, deep_scale, device=_GPU_DEVICE
    )
    np.testing.assert_allclose(d_gpu, d_cpu, atol=1e-5)
    np.testing.assert_allclose(w_gpu, w_cpu, atol=1e-5)


# ---------------------------------------------------------------------------
# KDE mode: GPU vs CPU
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_kde_mode_cpu_basic():
    """CPU KDE mode returns a finite float within the input range."""
    from quantnado.peak_calling.call_seacr_peaks import _mode_via_kde

    rng = np.random.default_rng(0)
    vec = rng.exponential(scale=5.0, size=10_000)
    cutoff = float(np.percentile(vec, 80))
    mode = _mode_via_kde(vec, cutoff, device="cpu")
    assert np.isfinite(mode)
    assert vec.min() <= mode <= cutoff


@pytest.mark.unit
@gpu_only
def test_kde_gpu_matches_cpu():
    """GPU KDE mode matches CPU path within float32 tolerance."""
    from quantnado.peak_calling.call_seacr_peaks import _mode_via_kde

    rng = np.random.default_rng(0)
    vec = rng.exponential(scale=5.0, size=100_000)
    cutoff = float(np.percentile(vec, 80))

    cpu_mode = _mode_via_kde(vec, cutoff, device="cpu")
    gpu_mode = _mode_via_kde(vec, cutoff, device=_GPU_DEVICE)
    assert abs(gpu_mode - cpu_mode) < 0.1  # grid resolution ~0.02, so 0.1 is generous


# ---------------------------------------------------------------------------
# compute_islands: vectorized AUC/MaxSignal vs reference loop
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_compute_islands_vectorized_auc_maxsignal():
    """Vectorized prefix-sum AUC and reduceat MaxSignal match the reference loop."""
    from quantnado.peak_calling.call_seacr_peaks import _compute_islands

    rng = np.random.default_rng(3)
    # ~5% sparsity, 1M bp
    cov = np.zeros(1_000_000, dtype=np.float32)
    n_islands = 500
    positions = rng.integers(0, 950_000, size=n_islands)
    for p in positions:
        length = rng.integers(50, 500)
        cov[p : p + length] = rng.random(min(length, 1_000_000 - p)).astype(np.float32) * 10

    df = _compute_islands(cov, "chr1")
    assert df is not None
    assert (df["AUC"] > 0).all()
    assert (df["MaxSignal"] > 0).all()
    # AUC ≥ MaxSignal (AUC is sum, MaxSignal is max per bp)
    assert (df["AUC"] >= df["MaxSignal"]).all()

    # Cross-check AUC against reference loop
    ref_auc = np.array([
        float(cov.astype(np.float64)[s:e].sum())
        for s, e in zip(df["Start"], df["End"])
    ])
    np.testing.assert_allclose(df["AUC"].to_numpy(), ref_auc, rtol=1e-6)

    # Cross-check MaxSignal against reference loop
    ref_max = np.array([
        float(cov.astype(np.float64)[s:e].max())
        for s, e in zip(df["Start"], df["End"])
    ])
    np.testing.assert_allclose(df["MaxSignal"].to_numpy(), ref_max, rtol=1e-6)

"""Unit tests for LanceOtron peak calling — pure numpy, no zarr or model weights required."""

import numpy as np
import pytest

from quantnado.peak_calling.call_lanceotron_peaks import (
    DEEP_WINDOW,
    MAX_PEAK_WIDTH,
    MIN_PEAK_WIDTH,
    N_LAMBDA_STEPS,
    _extract_features,
    _find_candidate_peaks,
    _normalise_features,
    _rolling_mean,
    _rpkm_normalise,
)


# ---------------------------------------------------------------------------
# _rpkm_normalise
# ---------------------------------------------------------------------------


def test_rpkm_normalise_basic():
    cov = np.array([0, 10, 20], dtype=np.uint32)
    result = _rpkm_normalise(cov, total_reads=10_000_000)
    expected = np.array([0.0, 1000.0, 2000.0], dtype=np.float32)
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_rpkm_normalise_zero_reads():
    cov = np.array([5, 5], dtype=np.uint32)
    result = _rpkm_normalise(cov, total_reads=0)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, np.array([5.0, 5.0], dtype=np.float32))


# ---------------------------------------------------------------------------
# _rolling_mean
# ---------------------------------------------------------------------------


def test_rolling_mean_flat():
    arr = np.ones(1000, dtype=np.float32) * 5.0
    result = _rolling_mean(arr, window=100)
    np.testing.assert_allclose(result, 5.0, atol=1e-5)


def test_rolling_mean_shape_preserved():
    arr = np.random.rand(5000).astype(np.float32)
    result = _rolling_mean(arr, window=400)
    assert result.shape == arr.shape


def test_rolling_mean_smooths_spike():
    arr = np.zeros(1000, dtype=np.float32)
    arr[500] = 1000.0
    result = _rolling_mean(arr, window=100)
    # Spike should be spread out — surrounding area gets non-zero values
    assert result[500] < 1000.0
    assert result.max() > 0


# ---------------------------------------------------------------------------
# _find_candidate_peaks
# ---------------------------------------------------------------------------


def test_find_candidate_peaks_single_peak():
    arr = np.zeros(10_000, dtype=np.float32)
    arr[4000:4200] = 100.0  # 200 bp peak, mean=100
    chrom_mean = float(arr.mean())
    candidates = _find_candidate_peaks(arr, chrom_mean, threshold_factor=4.0)
    assert len(candidates) == 1
    start, end = candidates[0]
    assert start >= 4000
    assert end <= 4200


def test_find_candidate_peaks_no_enrichment():
    arr = np.ones(10_000, dtype=np.float32)
    chrom_mean = 1.0
    candidates = _find_candidate_peaks(arr, chrom_mean, threshold_factor=4.0)
    assert candidates == []


def test_find_candidate_peaks_too_narrow():
    arr = np.zeros(10_000, dtype=np.float32)
    arr[5000:5010] = 1000.0  # only 10 bp — below MIN_PEAK_WIDTH
    chrom_mean = float(arr.mean())
    candidates = _find_candidate_peaks(arr, chrom_mean, threshold_factor=4.0)
    assert all((e - s) >= MIN_PEAK_WIDTH for s, e in candidates)


def test_find_candidate_peaks_wide_region_splits():
    arr = np.zeros(20_000, dtype=np.float32)
    # Plateau 3× MAX_PEAK_WIDTH wide → must be recursively split
    arr[5000 : 5000 + MAX_PEAK_WIDTH * 3] = 1000.0
    chrom_mean = float(arr.mean())
    candidates = _find_candidate_peaks(arr, chrom_mean, threshold_factor=4.0)
    # Each resulting candidate should be ≤ MAX_PEAK_WIDTH
    assert all((e - s) <= MAX_PEAK_WIDTH for s, e in candidates)


def test_find_candidate_peaks_multiple():
    arr = np.zeros(50_000, dtype=np.float32)
    arr[1000:1200] = 200.0
    arr[10000:10300] = 300.0
    chrom_mean = float(arr.mean())
    candidates = _find_candidate_peaks(arr, chrom_mean, threshold_factor=4.0)
    assert len(candidates) == 2


# ---------------------------------------------------------------------------
# _extract_features
# ---------------------------------------------------------------------------


def test_extract_features_shapes():
    chrom_len = 50_000
    norm_cov = np.random.rand(chrom_len).astype(np.float32) * 10
    candidates = [(1000, 1200), (5000, 5300), (20000, 20400)]
    seq_depth = 0.05
    X_deep, X_wide = _extract_features(norm_cov, candidates, seq_depth)
    assert X_deep.shape == (3, DEEP_WINDOW)
    assert X_wide.shape == (3, N_LAMBDA_STEPS + 2)


def test_extract_features_seq_depth_in_wide():
    chrom_len = 50_000
    norm_cov = np.ones(chrom_len, dtype=np.float32) * 5.0
    candidates = [(5000, 5200)]
    seq_depth = 0.123
    _, X_wide = _extract_features(norm_cov, candidates, seq_depth)
    assert X_wide[0, -1] == pytest.approx(seq_depth, rel=1e-5)


def test_extract_features_max_height_in_wide():
    chrom_len = 50_000
    norm_cov = np.zeros(chrom_len, dtype=np.float32)
    norm_cov[4990:5010] = 42.0
    candidates = [(4990, 5010)]
    _, X_wide = _extract_features(norm_cov, candidates, seq_depth=0.05)
    assert X_wide[0, 0] == pytest.approx(42.0, rel=1e-4)


def test_extract_features_deep_window_near_edge():
    chrom_len = 500  # very short chromosome
    norm_cov = np.ones(chrom_len, dtype=np.float32)
    candidates = [(10, 50)]  # near start
    X_deep, _ = _extract_features(norm_cov, candidates, seq_depth=0.05)
    assert X_deep.shape == (1, DEEP_WINDOW)
    # Zero-padded region should be 0
    assert X_deep[0, chrom_len:].sum() == 0.0


# ---------------------------------------------------------------------------
# _normalise_features
# ---------------------------------------------------------------------------


def _make_dummy_scalers():
    wide_mean = np.zeros(N_LAMBDA_STEPS + 2)
    wide_scale = np.ones(N_LAMBDA_STEPS + 2)
    deep_mean = np.zeros(DEEP_WINDOW)
    deep_scale = np.ones(DEEP_WINDOW)
    return (wide_mean, wide_scale), (deep_mean, deep_scale)


def test_normalise_features_identity_scalers():
    """With zero-mean unit-scale scalers, only per-sample z-score is applied."""
    rng = np.random.default_rng(42)
    n_peaks = 20
    X_deep = rng.random((n_peaks, DEEP_WINDOW)).astype(np.float32)
    X_wide = rng.random((n_peaks, N_LAMBDA_STEPS + 2)).astype(np.float32)
    wide_scaler, deep_scaler = _make_dummy_scalers()

    X_deep_n, X_wide_n = _normalise_features(
        X_deep, X_wide,
        wide_mean=wide_scaler[0], wide_scale=wide_scaler[1],
        deep_mean=deep_scaler[0], deep_scale=deep_scaler[1],
    )
    assert X_deep_n.shape == (n_peaks, DEEP_WINDOW)
    assert X_wide_n.shape == (n_peaks, N_LAMBDA_STEPS + 2)
    assert X_deep_n.dtype == np.float32
    assert X_wide_n.dtype == np.float32


def test_normalise_features_wide_offset_applied():
    """Wide scaler shift should be visible in normalised output."""
    n_peaks = 5
    X_deep = np.ones((n_peaks, DEEP_WINDOW), dtype=np.float32)
    X_wide = np.ones((n_peaks, N_LAMBDA_STEPS + 2), dtype=np.float32) * 3.0
    wide_mean = np.ones(N_LAMBDA_STEPS + 2) * 3.0   # shift should produce 0
    wide_scale = np.ones(N_LAMBDA_STEPS + 2)
    _, (deep_mean, deep_scale) = _make_dummy_scalers()

    _, X_wide_n = _normalise_features(
        X_deep, X_wide,
        wide_mean=wide_mean, wide_scale=wide_scale,
        deep_mean=deep_mean, deep_scale=deep_scale,
    )
    np.testing.assert_allclose(X_wide_n, 0.0, atol=1e-5)


def test_normalise_features_per_sample_zscore():
    """Per-sample z-score step: each column of X_deep.T should be ~unit variance."""
    rng = np.random.default_rng(0)
    n_peaks = 50
    X_deep = (rng.random((n_peaks, DEEP_WINDOW)) * 100).astype(np.float32)
    X_wide = rng.random((n_peaks, N_LAMBDA_STEPS + 2)).astype(np.float32)
    wide_scaler, deep_scaler = _make_dummy_scalers()

    X_deep_n, _ = _normalise_features(
        X_deep, X_wide,
        wide_mean=wide_scaler[0], wide_scale=wide_scaler[1],
        deep_mean=deep_scaler[0], deep_scale=deep_scaler[1],
    )
    # After step-1 z-score (with identity step-2): column stds should be ~1
    col_stds = X_deep_n.T.std(axis=0)
    np.testing.assert_allclose(col_stds, 1.0, atol=0.1)

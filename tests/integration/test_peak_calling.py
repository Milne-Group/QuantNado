"""Integration tests for quantile-based peak calling."""
import pandas as pd
import pytest

from quantnado.peak_calling.call_quantile_peaks import call_quantile_peaks

pytestmark = pytest.mark.integration


def test_call_quantile_peaks_basic():
    signal = pd.Series([0, 1, 5, 10, 1, 0], name="test_sample")
    chroms = pd.Series(["chr1"] * 6)
    starts = pd.Series([0, 100, 200, 300, 400, 500])
    ends = pd.Series([100, 200, 300, 400, 500, 600])

    peaks = call_quantile_peaks(
        signal=signal,
        chroms=chroms,
        starts=starts,
        ends=ends,
        tilesize=100,
        quantile=0.95,
        blacklist_file=None,
    )

    assert peaks is not None
    assert len(peaks) == 1
    peak_df = pd.DataFrame(peaks)
    assert peak_df.iloc[0]["Start"] == 300
    assert peak_df.iloc[0]["End"] == 400


def test_call_quantile_peaks_no_peaks_above_threshold():
    """When all tiles are equal, no peak should pass a strict quantile."""
    signal = pd.Series([1, 1, 1, 1], name="flat")
    chroms = pd.Series(["chr1"] * 4)
    starts = pd.Series([0, 100, 200, 300])
    ends = pd.Series([100, 200, 300, 400])

    peaks = call_quantile_peaks(
        signal=signal,
        chroms=chroms,
        starts=starts,
        ends=ends,
        tilesize=100,
        quantile=1.0,
        blacklist_file=None,
    )

    # quantile=1.0 → returns None immediately
    assert peaks is None


def test_call_quantile_peaks_multiple_chroms():
    signal = pd.Series([0, 10, 0, 10], name="multi")
    chroms = pd.Series(["chr1", "chr1", "chr2", "chr2"])
    starts = pd.Series([0, 100, 0, 100])
    ends = pd.Series([100, 200, 100, 200])

    peaks = call_quantile_peaks(
        signal=signal,
        chroms=chroms,
        starts=starts,
        ends=ends,
        tilesize=100,
        quantile=0.5,
        blacklist_file=None,
    )
    assert peaks is not None
    assert len(peaks) >= 1


def test_call_quantile_peaks_all_zero_returns_none():
    """All-zero signal should return None (no nonzero values)."""
    signal = pd.Series([0, 0, 0], name="zero")
    chroms = pd.Series(["chr1"] * 3)
    starts = pd.Series([0, 100, 200])
    ends = pd.Series([100, 200, 300])

    peaks = call_quantile_peaks(
        signal=signal,
        chroms=chroms,
        starts=starts,
        ends=ends,
        tilesize=100,
        quantile=0.9,
        blacklist_file=None,
    )
    assert peaks is None

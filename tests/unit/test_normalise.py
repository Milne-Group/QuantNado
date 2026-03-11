"""Unit tests for quantnado/analysis/normalise.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import dask.array as da

from quantnado.analysis.normalise import (
    _resolve_library_sizes,
    _scale_per_sample,
    get_library_sizes,
    get_mean_read_lengths,
    normalise,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SAMPLES = ["s1", "s2", "s3"]
LIB_SIZES = pd.Series(
    {"s1": 1_000_000, "s2": 2_000_000, "s3": 4_000_000}, name="library_size"
)


def _make_dataframe(n_features=5):
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        rng.integers(10, 1000, (n_features, 3)).astype(float),
        columns=SAMPLES,
    )


def _make_xr_dataset(n_ranges=4, with_range_length=True):
    """xr.Dataset similar to reduce() output."""
    rng = np.random.default_rng(1)
    vals = rng.random((n_ranges, 3)).astype(np.float32)
    ds = xr.Dataset(
        {
            "mean": (["ranges", "sample"], vals),
            "sum": (["ranges", "sample"], vals * 2),
            "count": (["ranges", "sample"], np.ones((n_ranges, 3), dtype=np.int64)),
        },
        coords={"sample": SAMPLES},
    )
    if with_range_length:
        ds = ds.assign_coords(range_length=("ranges", np.full(n_ranges, 1000)))
    return ds


def _make_xr_dataarray(n_intervals=4, n_positions=10):
    """xr.DataArray similar to extract() output, dims (interval, position, sample)."""
    rng = np.random.default_rng(2)
    data = rng.random((n_intervals, n_positions, 3)).astype(np.float32)
    return xr.DataArray(
        data,
        dims=("interval", "relative_position", "sample"),
        coords={"sample": SAMPLES},
    )


class _FakeZarrArray:
    """Minimal zarr-array stand-in (supports [:] indexing)."""

    def __init__(self, arr: np.ndarray):
        self._arr = arr

    def __getitem__(self, idx):
        return self._arr[idx]


class _FakeStore:
    """Minimal BamStore-like object for get_library_sizes tests."""

    def __init__(self, reads, completed, sample_names=None):
        self.meta = {"total_reads": _FakeZarrArray(np.asarray(reads))}
        self.completed_mask = np.asarray(completed)
        self.sample_names = sample_names or [f"s{i+1}" for i in range(len(reads))]


# ---------------------------------------------------------------------------
# _resolve_library_sizes
# ---------------------------------------------------------------------------


class TestResolveLibrarySizes:
    def test_dict_converted_to_series(self):
        result = _resolve_library_sizes(None, {"s1": 1e6, "s2": 2e6})
        assert isinstance(result, pd.Series)
        assert result["s1"] == pytest.approx(1e6)

    def test_series_passthrough(self):
        result = _resolve_library_sizes(None, LIB_SIZES)
        assert isinstance(result, pd.Series)
        assert list(result.index) == list(LIB_SIZES.index)

    def test_both_none_raises(self):
        with pytest.raises(ValueError, match="dataset"):
            _resolve_library_sizes(None, None)


# ---------------------------------------------------------------------------
# _scale_per_sample
# ---------------------------------------------------------------------------


class TestScalePerSample:
    def test_correct_values(self):
        scale = _scale_per_sample(LIB_SIZES, ["s1", "s2", "s3"])
        np.testing.assert_allclose(scale, [1.0, 2.0, 4.0])

    def test_subset_ordering(self):
        scale = _scale_per_sample(LIB_SIZES, ["s3", "s1"])
        np.testing.assert_allclose(scale, [4.0, 1.0])

    def test_missing_sample_raises(self):
        with pytest.raises(ValueError, match="missing"):
            _scale_per_sample(LIB_SIZES, ["s1", "s_unknown"])


# ---------------------------------------------------------------------------
# normalise — dispatcher
# ---------------------------------------------------------------------------


class TestNormaliseDispatcher:
    def test_unknown_method_raises(self):
        df = _make_dataframe()
        with pytest.raises(ValueError, match="Unknown normalisation method"):
            normalise(df, library_sizes=LIB_SIZES, method="fpkm")

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError):
            normalise([1, 2, 3], library_sizes=LIB_SIZES, method="cpm")

    def test_method_case_insensitive(self):
        df = _make_dataframe()
        result = normalise(df, library_sizes=LIB_SIZES, method="CPM")
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# pd.DataFrame normalisation
# ---------------------------------------------------------------------------


class TestNormaliseDataFrame:
    def test_cpm_returns_dataframe(self):
        df = _make_dataframe()
        result = normalise(df, library_sizes=LIB_SIZES, method="cpm")
        assert isinstance(result, pd.DataFrame)
        assert result.shape == df.shape

    def test_cpm_divides_by_lib_size_million(self):
        df = _make_dataframe()
        result = normalise(df, library_sizes=LIB_SIZES, method="cpm")
        # s1 lib size = 1e6 → scale = 1.0, so values unchanged
        pd.testing.assert_series_equal(result["s1"], df["s1"] / 1.0, check_names=False)
        pd.testing.assert_series_equal(result["s2"], df["s2"] / 2.0, check_names=False)
        pd.testing.assert_series_equal(result["s3"], df["s3"] / 4.0, check_names=False)

    def test_cpm_does_not_modify_input(self):
        df = _make_dataframe()
        original = df.copy()
        normalise(df, library_sizes=LIB_SIZES, method="cpm")
        pd.testing.assert_frame_equal(df, original)

    def test_rpkm_requires_feature_lengths(self):
        df = _make_dataframe()
        with pytest.raises(ValueError, match="feature_lengths"):
            normalise(df, library_sizes=LIB_SIZES, method="rpkm")

    def test_rpkm_length_mismatch_raises(self):
        df = _make_dataframe(n_features=5)
        with pytest.raises(ValueError, match="feature_lengths length"):
            normalise(df, library_sizes=LIB_SIZES, method="rpkm", feature_lengths=np.ones(3))

    def test_rpkm_equals_cpm_over_length_kb(self):
        df = _make_dataframe()
        lengths_bp = np.full(len(df), 2000)
        result = normalise(df, library_sizes=LIB_SIZES, method="rpkm", feature_lengths=lengths_bp)
        cpm = normalise(df, library_sizes=LIB_SIZES, method="cpm")
        pd.testing.assert_frame_equal(result, cpm / 2.0)

    def test_tpm_requires_feature_lengths(self):
        df = _make_dataframe()
        with pytest.raises(ValueError, match="feature_lengths"):
            normalise(df, method="tpm")

    def test_tpm_length_mismatch_raises(self):
        df = _make_dataframe(n_features=5)
        with pytest.raises(ValueError, match="feature_lengths length"):
            normalise(df, method="tpm", feature_lengths=np.ones(3))

    def test_tpm_columns_sum_to_1e6(self):
        df = _make_dataframe()
        lengths = np.full(len(df), 1000)
        result = normalise(df, method="tpm", feature_lengths=lengths)
        np.testing.assert_allclose(result.sum(axis=0).values, 1e6, rtol=1e-5)

    def test_tpm_no_dataset_needed(self):
        """TPM is self-normalising — dataset and library_sizes are not required."""
        df = _make_dataframe()
        lengths = np.full(len(df), 1000)
        result = normalise(df, method="tpm", feature_lengths=lengths)
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# xr.Dataset normalisation (reduce() output)
# ---------------------------------------------------------------------------


class TestNormaliseXrDataset:
    def test_cpm_returns_dataset(self):
        ds = _make_xr_dataset()
        result = normalise(ds, library_sizes=LIB_SIZES, method="cpm")
        assert isinstance(result, xr.Dataset)

    def test_cpm_data_vars_preserved(self):
        ds = _make_xr_dataset()
        result = normalise(ds, library_sizes=LIB_SIZES, method="cpm")
        assert set(result.data_vars) == set(ds.data_vars)

    def test_cpm_preserves_count_unchanged(self):
        ds = _make_xr_dataset()
        result = normalise(ds, library_sizes=LIB_SIZES, method="cpm")
        np.testing.assert_array_equal(
            result["count"].values, ds["count"].values
        )

    def test_cpm_normalises_mean(self):
        ds = _make_xr_dataset()
        result = normalise(ds, library_sizes=LIB_SIZES, method="cpm")
        scale = np.array([1.0, 2.0, 4.0])  # lib_sizes / 1e6
        expected = ds["mean"].values / scale
        np.testing.assert_allclose(result["mean"].values, expected, rtol=1e-5)

    def test_cpm_attrs_contain_normalised(self):
        ds = _make_xr_dataset()
        result = normalise(ds, library_sizes=LIB_SIZES, method="cpm")
        assert result.attrs.get("normalised") == "cpm"

    def test_cpm_result_is_lazy(self):
        ds = _make_xr_dataset()
        result = normalise(ds, library_sizes=LIB_SIZES, method="cpm")
        assert isinstance(result["mean"].data, da.Array)

    def test_rpkm_uses_range_length_coord(self):
        # range_length = 1000 bp = 1 kb → RPKM == CPM / 1.0
        ds = _make_xr_dataset(with_range_length=True)
        rpkm = normalise(ds, library_sizes=LIB_SIZES, method="rpkm")
        cpm = normalise(ds, library_sizes=LIB_SIZES, method="cpm")
        np.testing.assert_allclose(rpkm["mean"].values, cpm["mean"].values, rtol=1e-4)

    def test_rpkm_uses_feature_lengths_arg_when_no_coord(self):
        ds = _make_xr_dataset(with_range_length=False)
        lengths = np.full(4, 2000)  # 2 kb
        rpkm = normalise(ds, library_sizes=LIB_SIZES, method="rpkm", feature_lengths=lengths)
        cpm = normalise(ds, library_sizes=LIB_SIZES, method="cpm")
        np.testing.assert_allclose(rpkm["mean"].values, cpm["mean"].values / 2.0, rtol=1e-4)

    def test_rpkm_no_lengths_raises(self):
        ds = _make_xr_dataset(with_range_length=False)
        with pytest.raises(ValueError, match="RPKM requires feature lengths"):
            normalise(ds, library_sizes=LIB_SIZES, method="rpkm")


# ---------------------------------------------------------------------------
# xr.DataArray normalisation (extract() output)
# ---------------------------------------------------------------------------


class TestNormaliseXrDataArray:
    def test_cpm_returns_dataarray(self):
        da_in = _make_xr_dataarray()
        result = normalise(da_in, library_sizes=LIB_SIZES, method="cpm")
        assert isinstance(result, xr.DataArray)

    def test_cpm_shape_preserved(self):
        da_in = _make_xr_dataarray()
        result = normalise(da_in, library_sizes=LIB_SIZES, method="cpm")
        assert result.shape == da_in.shape

    def test_cpm_dims_preserved(self):
        da_in = _make_xr_dataarray()
        result = normalise(da_in, library_sizes=LIB_SIZES, method="cpm")
        assert result.dims == da_in.dims

    def test_cpm_values(self):
        da_in = _make_xr_dataarray()
        result = normalise(da_in, library_sizes=LIB_SIZES, method="cpm")
        scale = np.array([1.0, 2.0, 4.0])  # lib_sizes / 1e6
        expected = da_in.values / scale
        np.testing.assert_allclose(result.values, expected, rtol=1e-5)

    def test_cpm_attrs_contain_normalised(self):
        da_in = _make_xr_dataarray()
        result = normalise(da_in, library_sizes=LIB_SIZES, method="cpm")
        assert result.attrs.get("normalised") == "cpm"

    def test_cpm_result_is_lazy(self):
        da_in = _make_xr_dataarray()
        result = normalise(da_in, library_sizes=LIB_SIZES, method="cpm")
        assert isinstance(result.data, da.Array)

    def test_rpkm_returns_dataarray(self):
        da_in = _make_xr_dataarray()
        result = normalise(da_in, library_sizes=LIB_SIZES, method="rpkm")
        assert isinstance(result, xr.DataArray)
        assert result.shape == da_in.shape

    def test_rpkm_without_read_lengths_uses_bin_size(self):
        """Without mean_read_lengths, RPKM = CPM / bin_size_kb."""
        da_in = _make_xr_dataarray()
        cpm = normalise(da_in, library_sizes=LIB_SIZES, method="cpm")
        rpkm = normalise(da_in, library_sizes=LIB_SIZES, method="rpkm")
        coords = da_in.coords["relative_position"].values
        bin_size_kb = abs(float(coords[1] - coords[0])) / 1000.0
        np.testing.assert_allclose(rpkm.values, cpm.values / bin_size_kb, rtol=1e-5)

    def test_rpkm_with_read_lengths_larger_than_bin_size_uses_bin_size_kb(self):
        """When bin_size <= read_length (long reads span the bin), RPKM = CPM / bin_size_kb."""
        # _make_xr_dataarray uses coords 0,1,...,9 so bin_size=1bp; read_length=150 >> bin_size
        da_in = _make_xr_dataarray()
        read_lengths = pd.Series({"s1": 150.0, "s2": 150.0, "s3": 150.0}, name="mean_read_length")
        cpm = normalise(da_in, library_sizes=LIB_SIZES, method="cpm")
        rpkm = normalise(da_in, library_sizes=LIB_SIZES, method="rpkm", mean_read_lengths=read_lengths)
        # min(bin_size=1, read_length=150) = 1 → RPKM = CPM / 0.001
        np.testing.assert_allclose(rpkm.values, cpm.values / 0.001, rtol=1e-5)

    def test_rpkm_with_read_lengths_smaller_than_bin_size_uses_read_length_kb(self):
        """When bin_size > read_length (reads shorter than bin), RPKM = CPM / read_length_kb."""
        # Make a DataArray with bin coords spaced 200bp apart (bin_size=200)
        rng = np.random.default_rng(99)
        data = rng.random((4, 10, 3)).astype(np.float32)
        da_in = xr.DataArray(
            data,
            dims=("interval", "bin", "sample"),
            coords={"sample": SAMPLES, "bin": np.arange(10) * 200},
            attrs={"bin_size": 200},
        )
        read_lengths = pd.Series({"s1": 50.0, "s2": 50.0, "s3": 50.0}, name="mean_read_length")
        cpm = normalise(da_in, library_sizes=LIB_SIZES, method="cpm")
        rpkm = normalise(da_in, library_sizes=LIB_SIZES, method="rpkm", mean_read_lengths=read_lengths)
        # min(bin_size=200, read_length=50) = 50 → RPKM = CPM / 0.05
        np.testing.assert_allclose(rpkm.values, cpm.values / 0.05, rtol=1e-5)

    def test_rpkm_result_is_lazy(self):
        da_in = _make_xr_dataarray()
        result = normalise(da_in, library_sizes=LIB_SIZES, method="rpkm")
        assert isinstance(result.data, da.Array)

    def test_tpm_raises(self):
        da_in = _make_xr_dataarray()
        with pytest.raises(ValueError, match="TPM"):
            normalise(da_in, library_sizes=LIB_SIZES, method="tpm")


# ---------------------------------------------------------------------------
# get_library_sizes
# ---------------------------------------------------------------------------


class TestGetLibrarySizes:
    def test_missing_total_reads_raises(self):
        """Store without total_reads should raise RuntimeError."""

        class _NoReadsStore:
            meta = {}
            sample_names = ["s1"]
            completed_mask = np.array([True])

        with pytest.raises(RuntimeError, match="total_reads"):
            get_library_sizes(_NoReadsStore())

    def test_returns_series_indexed_by_sample_names(self):
        store = _FakeStore(reads=[1_000_000, 2_000_000], completed=[True, True])
        result = get_library_sizes(store)
        assert isinstance(result, pd.Series)
        assert list(result.index) == store.sample_names

    def test_correct_values(self):
        store = _FakeStore(reads=[1_000_000, 3_000_000], completed=[True, True])
        result = get_library_sizes(store)
        assert result["s1"] == pytest.approx(1_000_000)
        assert result["s2"] == pytest.approx(3_000_000)

    def test_incomplete_sample_is_nan(self):
        store = _FakeStore(reads=[1_000_000, 2_000_000], completed=[True, False])
        result = get_library_sizes(store)
        assert result["s1"] == pytest.approx(1_000_000)
        assert np.isnan(result["s2"])

    def test_resolves_from_bam_store_via_dataset(self):
        """get_library_sizes accepts a QuantNado-like object with a .coverage attr."""
        store = _FakeStore(reads=[500_000], completed=[True], sample_names=["sA"])

        class _FakeDataset:
            coverage = store

        result = get_library_sizes(_FakeDataset())
        assert result["sA"] == pytest.approx(500_000)

"""Unit tests for quantnado.dataset.pca (no BamStore dependency)."""
import numpy as np
import pytest
import dask.array as da
import xarray as xr

from quantnado.dataset.pca import run_pca


def _make_dataarray(data: np.ndarray) -> xr.DataArray:
    """Wrap a 2D numpy array in an xr.DataArray with (sample, feature) dims."""
    return xr.DataArray(
        da.from_array(data, chunks=data.shape),
        dims=("sample", "feature"),
    )


class TestRunPCA:
    def test_basic_shape(self):
        data = np.random.default_rng(0).random((5, 10))
        arr = _make_dataarray(data)
        _, transformed = run_pca(arr, n_components=2, random_state=0)
        assert transformed.compute().shape == (5, 2)

    def test_nan_handling_drop(self):
        data = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        arr = _make_dataarray(data)
        pca_obj, transformed = run_pca(arr, n_components=1, nan_handling_strategy="drop", random_state=0)
        result = transformed.compute()
        assert result.shape == (3, 1)
        assert np.all(np.isfinite(result))

    def test_nan_handling_set_to_zero(self):
        data = np.array([[1.0, np.nan], [2.0, 3.0], [4.0, 5.0]])
        arr = _make_dataarray(data)
        _, transformed = run_pca(arr, n_components=1, nan_handling_strategy="set_to_zero", random_state=0)
        result = transformed.compute()
        assert result.shape == (3, 1)
        assert np.all(np.isfinite(result))

    def test_nan_handling_mean_imputation(self):
        data = np.array(
            [[1.0, np.nan, 3.0], [1.0, 2.0, np.nan], [2.0, 2.0, 2.0]]
        )
        arr = _make_dataarray(data)
        _, transformed = run_pca(
            arr,
            n_components=2,
            nan_handling_strategy="mean_value_imputation",
            random_state=0,
        )
        result = transformed.compute()
        assert result.shape == (3, 2)
        assert np.all(np.isfinite(result))

    def test_standardize_centers_output(self):
        data = np.array(
            [[1.0, np.nan, 3.0], [1.0, 2.0, np.nan], [2.0, 2.0, 2.0]]
        )
        arr = _make_dataarray(data)
        _, transformed = run_pca(
            arr,
            n_components=2,
            nan_handling_strategy="mean_value_imputation",
            standardize=True,
            random_state=0,
        )
        result = transformed.compute()
        assert np.allclose(result.mean(axis=0), 0.0, atol=1e-6)

    def test_n_components_stored(self):
        data = np.random.default_rng(0).random((4, 8))
        arr = _make_dataarray(data)
        pca_obj, _ = run_pca(arr, n_components=3, random_state=0)
        assert pca_obj.n_components == 3

    def test_invalid_nan_strategy_raises(self):
        data = np.ones((3, 4))
        arr = _make_dataarray(data)
        with pytest.raises(ValueError, match="nan_handling_strategy"):
            run_pca(arr, n_components=1, nan_handling_strategy="invalid")

    def test_all_nan_column_dropped(self):
        data = np.array([[1.0, np.nan], [2.0, np.nan]])
        arr = _make_dataarray(data)
        _, transformed = run_pca(arr, n_components=1, nan_handling_strategy="drop", random_state=0)
        # Only one valid column remains; n_components=1 is feasible.
        result = transformed.compute()
        assert result.shape == (2, 1)

    def test_subset_size_reduces_features(self):
        rng = np.random.default_rng(42)
        data = rng.random((6, 20))
        arr = _make_dataarray(data)
        _, transformed = run_pca(arr, n_components=2, subset_size=5, random_state=42)
        assert transformed.compute().shape == (6, 2)

    def test_subset_strategy_first(self):
        rng = np.random.default_rng(42)
        data = rng.random((4, 10))
        arr = _make_dataarray(data)
        _, transformed = run_pca(
            arr, n_components=2, subset_size=4, subset_strategy="first", random_state=42
        )
        assert transformed.compute().shape == (4, 2)

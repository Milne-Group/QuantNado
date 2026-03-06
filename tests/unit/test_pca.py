"""Unit tests for quantnado.dataset.pca (no BamStore dependency)."""
import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import numpy as np
import pandas as pd
import pytest
import dask.array as da
import xarray as xr

from quantnado.dataset.pca import (
    _normalise_orientation,
    plot_pca_scatter,
    plot_pca_scree,
    run_pca,
)


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

    def test_invalid_subset_strategy_raises(self):
        data = np.random.default_rng(0).random((4, 10))
        arr = _make_dataarray(data)
        with pytest.raises(ValueError, match="subset_strategy"):
            run_pca(arr, n_components=2, subset_size=5, subset_strategy="bad", random_state=0)

    def test_numpy_standardize(self):
        """Non-dask standardize path (lines 138-142)."""
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        arr = xr.DataArray(
            da.from_array(data),
            dims=("sample", "feature"),
        )
        _, transformed = run_pca(arr, n_components=1, standardize=True, random_state=0)
        assert transformed.compute().shape == (3, 1)


# ---------------------------------------------------------------------------
# _normalise_orientation
# ---------------------------------------------------------------------------

class TestNormaliseOrientation:
    def test_sample_first_axis_unchanged(self):
        arr = xr.DataArray(np.zeros((3, 5)), dims=("sample", "feature"))
        result = _normalise_orientation(arr)
        assert result.dims[0] == "sample"

    def test_sample_second_axis_transposed(self):
        arr = xr.DataArray(np.zeros((5, 3)), dims=("feature", "sample"))
        result = _normalise_orientation(arr)
        assert result.dims[0] == "sample"

    def test_sample_id_dim_name(self):
        arr = xr.DataArray(np.zeros((3, 5)), dims=("sample_id", "feature"))
        result = _normalise_orientation(arr)
        assert result.dims[0] == "sample_id"

    def test_1d_raises(self):
        arr = xr.DataArray(np.zeros(5), dims=("sample",))
        with pytest.raises(ValueError, match="2D"):
            _normalise_orientation(arr)

    def test_unknown_dim_raises(self):
        arr = xr.DataArray(np.zeros((3, 5)), dims=("rows", "cols"))
        with pytest.raises(ValueError, match="sample dimension"):
            _normalise_orientation(arr)


# ---------------------------------------------------------------------------
# plot_pca_scree
# ---------------------------------------------------------------------------

class TestPlotPcaScree:
    def _fit_pca(self):
        data = np.random.default_rng(0).random((5, 10))
        arr = _make_dataarray(data)
        pca_obj, _ = run_pca(arr, n_components=2, random_state=0)
        return pca_obj

    def test_returns_plot(self):
        import matplotlib.pyplot as plt
        pca_obj = self._fit_pca()
        plot = plot_pca_scree(pca_obj)
        assert plot is not None
        plt.close("all")

    def test_saves_file(self, tmp_path):
        import matplotlib.pyplot as plt
        pca_obj = self._fit_pca()
        out = tmp_path / "scree.png"
        plot_pca_scree(pca_obj, filepath=str(out))
        assert out.exists()
        plt.close("all")


# ---------------------------------------------------------------------------
# plot_pca_scatter
# ---------------------------------------------------------------------------

class TestPlotPcaScatter:
    def _fit(self):
        data = np.random.default_rng(0).random((6, 10))
        arr = _make_dataarray(data)
        pca_obj, transformed = run_pca(arr, n_components=2, random_state=0)
        return pca_obj, transformed.compute()

    def test_basic_scatter(self):
        import matplotlib.pyplot as plt
        pca_obj, transformed = self._fit()
        plot = plot_pca_scatter(pca_obj, transformed)
        assert plot is not None
        plt.close("all")

    def test_with_metadata_colour_by(self):
        import matplotlib.pyplot as plt
        pca_obj, transformed = self._fit()
        md = pd.DataFrame({
            "sample_id": [f"s{i}" for i in range(6)],
            "condition": ["A", "B"] * 3,
        })
        plot = plot_pca_scatter(pca_obj, transformed, metadata_df=md, colour_by="condition")
        assert plot is not None
        plt.close("all")

    def test_with_shape_by(self):
        import matplotlib.pyplot as plt
        pca_obj, transformed = self._fit()
        md = pd.DataFrame({
            "sample_id": [f"s{i}" for i in range(6)],
            "group": ["X", "Y"] * 3,
        })
        plot = plot_pca_scatter(pca_obj, transformed, metadata_df=md, shape_by="group")
        assert plot is not None
        plt.close("all")

    def test_with_sample_labels_annotated(self):
        import matplotlib.pyplot as plt
        pca_obj, transformed = self._fit()
        md = pd.DataFrame({
            "sample": [f"s{i}" for i in range(6)],
        })
        plot = plot_pca_scatter(pca_obj, transformed, metadata_df=md)
        assert plot is not None
        plt.close("all")

    def test_saves_file(self, tmp_path):
        import matplotlib.pyplot as plt
        pca_obj, transformed = self._fit()
        out = tmp_path / "scatter.png"
        plot_pca_scatter(pca_obj, transformed, filepath=str(out))
        assert out.exists()
        plt.close("all")

    def test_metadata_length_mismatch_no_labels(self):
        import matplotlib.pyplot as plt
        pca_obj, transformed = self._fit()
        md = pd.DataFrame({"sample_id": ["s1"]})  # wrong length
        plot = plot_pca_scatter(pca_obj, transformed, metadata_df=md)
        assert plot is not None
        plt.close("all")

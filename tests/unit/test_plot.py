"""Unit tests for quantnado/analysis/plot.py."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from quantnado.analysis.plot import (
    _prep_extract,
    _resolve_palette,
    locus_plot,
    metaplot,
    tornadoplot,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_extract(n_intervals=4, n_positions=10, n_samples=2, pos_dim="relative_position", with_strand=False):
    """DataArray with dims (interval, <pos_dim>, sample) as returned by qn.extract()."""
    rng = np.random.default_rng(0)
    data = rng.random((n_intervals, n_positions, n_samples))
    coords = {
        pos_dim: np.linspace(-200, 200, n_positions),
        "sample": [f"s{i+1}" for i in range(n_samples)],
    }
    if with_strand:
        coords["strand"] = ("interval", np.array(["+", "-", "+", "-"][:n_intervals]))
    return xr.DataArray(data, dims=("interval", pos_dim, "sample"), coords=coords)


def _make_locus(n_positions=50, sample_names=None):
    """DataArray with dims (sample, position) for locus_plot inputs."""
    rng = np.random.default_rng(1)
    names = sample_names or ["s1", "s2"]
    data = rng.random((len(names), n_positions))
    return xr.DataArray(
        data,
        dims=("sample", "position"),
        coords={"sample": names, "position": np.arange(n_positions)},
    )


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# _resolve_palette
# ---------------------------------------------------------------------------


class TestResolvePalette:
    def test_none_returns_n_colors(self):
        colors = _resolve_palette(None, 3)
        assert len(colors) == 3

    def test_list_palette_cycles(self):
        colors = _resolve_palette(["red", "blue"], 4)
        assert colors == ["red", "blue", "red", "blue"]

    def test_dict_palette_with_labels(self):
        colors = _resolve_palette({"s1": "red", "s2": "blue"}, 2, labels=["s1", "s2"])
        assert colors == ["red", "blue"]

    def test_dict_palette_missing_label_falls_back(self):
        colors = _resolve_palette({"s1": "red"}, 2, labels=["s1", "s3"])
        assert colors[0] == "red"
        assert colors[1] is not None

    def test_string_cmap(self):
        colors = _resolve_palette("viridis", 5)
        assert len(colors) == 5
        assert all(len(c) == 4 for c in colors)  # RGBA tuples


# ---------------------------------------------------------------------------
# _prep_extract
# ---------------------------------------------------------------------------


class TestPrepExtract:
    def test_valid_dims_passthrough(self):
        da = _make_extract()
        result, x, pos_dim = _prep_extract(da, flip_minus_strand=False)
        assert result.dims == ("interval", "relative_position", "sample")
        assert pos_dim == "relative_position"

    def test_transposes_to_canonical_order(self):
        da = _make_extract().transpose("sample", "relative_position", "interval")
        result, _, _ = _prep_extract(da, flip_minus_strand=False)
        assert result.dims[0] == "interval"

    def test_invalid_dims_raises(self):
        bad = xr.DataArray(np.ones((3, 4)), dims=("x", "y"))
        with pytest.raises(ValueError, match="dims"):
            _prep_extract(bad, flip_minus_strand=False)

    def test_bin_dim_accepted(self):
        da = _make_extract(pos_dim="bin")
        result, _, pos_dim = _prep_extract(da, flip_minus_strand=False)
        assert pos_dim == "bin"

    def test_strand_flip_reverses_minus(self):
        da = _make_extract(n_intervals=2, with_strand=True)
        original = da.values.copy()
        result, _, _ = _prep_extract(da, flip_minus_strand=True)
        # minus-strand row (index 1) should be reversed
        np.testing.assert_array_equal(result.values[1], original[1, ::-1, :])
        # plus-strand row unchanged
        np.testing.assert_array_equal(result.values[0], original[0])

    def test_no_flip_when_disabled(self):
        da = _make_extract(n_intervals=2, with_strand=True)
        original = da.values.copy()
        result, _, _ = _prep_extract(da, flip_minus_strand=False)
        np.testing.assert_array_equal(result.values, original)


# ---------------------------------------------------------------------------
# metaplot
# ---------------------------------------------------------------------------


class TestMetaplot:
    def test_basic_returns_axes(self):
        da = _make_extract()
        ax = metaplot(da)
        assert hasattr(ax, "get_lines")

    def test_invalid_dims_raises(self):
        bad = xr.DataArray(np.ones((3, 4)), dims=("x", "y"))
        with pytest.raises(ValueError, match="dims"):
            metaplot(bad)

    def test_invalid_modality_raises(self):
        da = _make_extract()
        with pytest.raises(ValueError, match="modality"):
            metaplot(da, modality="bogus")

    def test_modality_sets_ylabel(self):
        da = _make_extract()
        ax = metaplot(da, modality="coverage")
        assert "Coverage" in ax.get_ylabel()

    def test_modality_methylation(self):
        da = _make_extract()
        ax = metaplot(da, modality="methylation")
        assert ax is not None

    def test_samples_subset(self):
        da = _make_extract(n_samples=3)
        ax = metaplot(da, samples=["s1"])
        lines = [l for l in ax.get_lines() if l.get_label() == "s1"]
        assert len(lines) >= 1

    def test_missing_sample_raises(self):
        da = _make_extract()
        with pytest.raises(ValueError, match="not found"):
            metaplot(da, samples=["nonexistent"])

    def test_groups(self):
        da = _make_extract(n_samples=3)
        ax = metaplot(da, groups={"g1": ["s1", "s2"], "g2": ["s3"]})
        assert ax is not None

    def test_groups_missing_sample_raises(self):
        da = _make_extract()
        with pytest.raises(ValueError, match="not found"):
            metaplot(da, groups={"g1": ["s1", "nope"]})

    def test_error_stat_none(self):
        da = _make_extract()
        ax = metaplot(da, error_stat=None)
        assert ax is not None

    def test_error_stat_std(self):
        da = _make_extract()
        ax = metaplot(da, error_stat="std")
        assert ax is not None

    def test_reference_point_none(self):
        da = _make_extract()
        ax = metaplot(da, reference_point=None)
        assert ax is not None

    def test_filepath_saves(self, tmp_path):
        da = _make_extract()
        out = tmp_path / "metaplot.png"
        metaplot(da, filepath=str(out))
        assert out.exists()

    def test_existing_axes_used(self):
        da = _make_extract()
        _, ax_in = plt.subplots()
        ax_out = metaplot(da, ax=ax_in)
        assert ax_out is ax_in

    def test_strand_flip_in_metaplot(self):
        da = _make_extract(n_intervals=2, with_strand=True)
        ax = metaplot(da, flip_minus_strand=True)
        assert ax is not None


# ---------------------------------------------------------------------------
# tornadoplot
# ---------------------------------------------------------------------------


class TestTornadoplot:
    def test_basic_returns_axes_list(self):
        da = _make_extract()
        axes = tornadoplot(da)
        assert isinstance(axes, list)
        assert len(axes) == 2  # 2 samples

    def test_single_sample(self):
        da = _make_extract(n_samples=1)
        axes = tornadoplot(da)
        assert len(axes) == 1

    def test_invalid_dims_raises(self):
        bad = xr.DataArray(np.ones((3, 4)), dims=("x", "y"))
        with pytest.raises(ValueError, match="dims"):
            tornadoplot(bad)

    def test_invalid_modality_raises(self):
        da = _make_extract()
        with pytest.raises(ValueError, match="modality"):
            tornadoplot(da, modality="bogus")

    def test_modality_methylation_sets_vmin_vmax(self):
        da = _make_extract()
        axes = tornadoplot(da, modality="methylation")
        assert axes is not None

    def test_samples_subset(self):
        da = _make_extract(n_samples=3)
        axes = tornadoplot(da, samples=["s1"])
        assert len(axes) == 1

    def test_missing_sample_raises(self):
        da = _make_extract()
        with pytest.raises(ValueError, match="not found"):
            tornadoplot(da, samples=["nope"])

    def test_groups(self):
        da = _make_extract(n_samples=3)
        axes = tornadoplot(da, groups={"g1": ["s1", "s2"], "g2": ["s3"]})
        assert len(axes) == 2

    def test_groups_missing_sample_raises(self):
        da = _make_extract()
        with pytest.raises(ValueError, match="not found"):
            tornadoplot(da, groups={"g1": ["s1", "nope"]})

    def test_sort_by_mean(self):
        da = _make_extract()
        axes = tornadoplot(da, sort_by="mean")
        assert axes is not None

    def test_sort_by_max(self):
        da = _make_extract()
        axes = tornadoplot(da, sort_by="max")
        assert axes is not None

    def test_sort_by_none(self):
        da = _make_extract()
        axes = tornadoplot(da, sort_by=None)
        assert axes is not None

    def test_invalid_sort_by_raises(self):
        da = _make_extract()
        with pytest.raises(ValueError, match="sort_by"):
            tornadoplot(da, sort_by="median")

    def test_sample_name_aliases(self):
        da = _make_extract(n_samples=2)
        axes = tornadoplot(da, samples=["s1", "s2"], sample_names=["alias1", "alias2"])
        assert axes[0].get_title() == "alias1"

    def test_sample_names_length_mismatch_raises(self):
        da = _make_extract(n_samples=2)
        with pytest.raises(ValueError, match="sample_names length"):
            tornadoplot(da, samples=["s1", "s2"], sample_names=["only_one"])

    def test_filepath_saves(self, tmp_path):
        da = _make_extract()
        out = tmp_path / "tornado.png"
        tornadoplot(da, filepath=str(out))
        assert out.exists()

    def test_vmin_vmax_explicit(self):
        da = _make_extract()
        axes = tornadoplot(da, vmin=0.0, vmax=1.0)
        assert axes is not None


# ---------------------------------------------------------------------------
# locus_plot
# ---------------------------------------------------------------------------


class TestLocusPlot:
    def test_coverage_modality(self):
        cov = _make_locus(sample_names=["s1"])
        axes = locus_plot(
            "chr1:0-50",
            sample_names=["s1"],
            modality=["coverage"],
            coverage=cov,
        )
        assert len(axes) == 1

    def test_methylation_modality(self):
        meth = _make_locus(sample_names=["s1"])
        axes = locus_plot(
            "chr1:0-50",
            sample_names=["s1"],
            modality=["methylation"],
            methylation=meth,
        )
        assert len(axes) == 1

    def test_variant_modality(self):
        ref = _make_locus(sample_names=["s1"])
        alt = _make_locus(sample_names=["s1"])
        axes = locus_plot(
            "chr1:0-50",
            sample_names=["s1"],
            modality=["variant"],
            allele_depth_ref=ref,
            allele_depth_alt=alt,
        )
        assert len(axes) == 1

    def test_variant_with_genotype(self):
        ref = _make_locus(sample_names=["s1"])
        alt = _make_locus(sample_names=["s1"])
        gt_data = np.random.choice([0, 1, 2, -1], size=(1, 50))
        gt = xr.DataArray(
            gt_data,
            dims=("sample", "position"),
            coords={"sample": ["s1"], "position": np.arange(50)},
        )
        axes = locus_plot(
            "chr1:0-50",
            sample_names=["s1"],
            modality=["variant"],
            allele_depth_ref=ref,
            allele_depth_alt=alt,
            genotype=gt,
        )
        assert len(axes) == 1

    def test_multi_track_mixed_modality(self):
        cov = _make_locus(sample_names=["s1", "s2"])
        meth = _make_locus(sample_names=["s1", "s2"])
        axes = locus_plot(
            "chr1:0-50",
            sample_names=["s1", "s2"],
            modality=["coverage", "methylation"],
            coverage=cov,
            methylation=meth,
        )
        assert len(axes) == 2

    def test_mismatched_lengths_raises(self):
        cov = _make_locus(sample_names=["s1"])
        with pytest.raises(ValueError, match="must match"):
            locus_plot(
                "chr1:0-50",
                sample_names=["s1", "s2"],
                modality=["coverage"],
                coverage=cov,
            )

    def test_missing_coverage_raises(self):
        with pytest.raises(ValueError, match="coverage"):
            locus_plot("chr1:0-50", sample_names=["s1"], modality=["coverage"])

    def test_missing_methylation_raises(self):
        with pytest.raises(ValueError, match="methylation"):
            locus_plot("chr1:0-50", sample_names=["s1"], modality=["methylation"])

    def test_missing_variant_raises(self):
        with pytest.raises(ValueError, match="allele_depth"):
            locus_plot("chr1:0-50", sample_names=["s1"], modality=["variant"])

    def test_unknown_modality_raises(self):
        cov = _make_locus(sample_names=["s1"])
        with pytest.raises(ValueError, match="Unknown modality"):
            locus_plot("chr1:0-50", sample_names=["s1"], modality=["bogus"], coverage=cov)

    def test_filepath_saves(self, tmp_path):
        cov = _make_locus(sample_names=["s1"])
        out = tmp_path / "locus.png"
        locus_plot(
            "chr1:0-50",
            sample_names=["s1"],
            modality=["coverage"],
            coverage=cov,
            filepath=str(out),
        )
        assert out.exists()

    def test_custom_title(self):
        cov = _make_locus(sample_names=["s1"])
        axes = locus_plot(
            "chr1:0-50",
            sample_names=["s1"],
            modality=["coverage"],
            coverage=cov,
            title="My locus",
        )
        assert axes is not None


# ---------------------------------------------------------------------------
# Dask array tests
# ---------------------------------------------------------------------------


class TestDaskArrays:
    """Test that dask-backed arrays are handled correctly."""

    def test_metaplot_with_dask_array(self):
        pytest.importorskip("dask.array")
        import dask.array as da

        da_data = da.from_delayed(
            _make_extract().data,
            shape=(4, 10, 2),
            dtype=float,
        )
        data = _make_extract()
        data.data = da_data
        ax = metaplot(data)
        assert ax is not None

    def test_tornadoplot_with_dask_array(self):
        pytest.importorskip("dask.array")
        import dask.array as da

        da_data = da.from_delayed(
            _make_extract().data,
            shape=(4, 10, 2),
            dtype=float,
        )
        data = _make_extract()
        data.data = da_data
        axes = tornadoplot(data)
        assert len(axes) == 2

    def test_locus_plot_with_dask_array(self):
        pytest.importorskip("dask.array")
        import dask.array as da

        locus_data = _make_locus(sample_names=["s1"])
        da_data = da.from_delayed(locus_data.data, shape=(1, 50), dtype=float)
        locus_data.data = da_data
        axes = locus_plot(
            "chr1:0-50",
            sample_names=["s1"],
            modality=["coverage"],
            coverage=locus_data,
        )
        assert len(axes) == 1


# ---------------------------------------------------------------------------
# Palette and styling tests
# ---------------------------------------------------------------------------


class TestPaletteAndStyling:
    """Test palette resolution and styling options."""

    def test_metaplot_with_dict_palette_and_groups(self):
        da = _make_extract(n_samples=3)
        palette = {"g1": "red", "g2": "blue"}
        ax = metaplot(
            da,
            groups={"g1": ["s1", "s2"], "g2": ["s3"]},
            palette=palette,
        )
        assert ax is not None

    def test_tornadoplot_with_list_palette(self):
        da = _make_extract(n_samples=3)
        axes = tornadoplot(da, samples=["s1", "s2", "s3"], palette=["red", "green"])
        assert len(axes) == 3

    def test_tornadoplot_with_cmap_palette(self):
        da = _make_extract(n_samples=2)
        axes = tornadoplot(da, palette="viridis")
        assert len(axes) == 2

    def test_locus_plot_with_palette_dict(self):
        cov = _make_locus(sample_names=["s1", "s2"])
        axes = locus_plot(
            "chr1:0-50",
            sample_names=["s1", "s2"],
            modality=["coverage", "coverage"],
            coverage=cov,
            palette={"s1": "red", "s2": "blue"},
        )
        assert len(axes) == 2

    def test_locus_plot_with_palette_cmap(self):
        cov = _make_locus(sample_names=["s1", "s2"])
        axes = locus_plot(
            "chr1:0-50",
            sample_names=["s1", "s2"],
            modality=["coverage", "coverage"],
            coverage=cov,
            palette="coolwarm",
        )
        assert len(axes) == 2

    def test_metaplot_custom_figsize(self):
        da = _make_extract()
        ax = metaplot(da, figsize=(12, 8))
        assert ax is not None

    def test_tornadoplot_custom_figsize(self):
        da = _make_extract()
        axes = tornadoplot(da, figsize=(15, 10))
        assert len(axes) == 2

    def test_locus_plot_custom_figsize(self):
        cov = _make_locus(sample_names=["s1"])
        axes = locus_plot(
            "chr1:0-50",
            sample_names=["s1"],
            modality=["coverage"],
            coverage=cov,
            figsize=(10, 5),
        )
        assert len(axes) == 1


# ---------------------------------------------------------------------------
# Modality-specific tests
# ---------------------------------------------------------------------------


class TestModalityDefaults:
    """Test that modality defaults are applied correctly."""

    def test_metaplot_coverage_modality(self):
        da = _make_extract()
        ax = metaplot(da, modality="coverage")
        assert "Coverage" in ax.get_ylabel()

    def test_metaplot_variant_modality(self):
        da = _make_extract()
        ax = metaplot(da, modality="variant")
        assert ax is not None

    def test_tornadoplot_coverage_modality(self):
        da = _make_extract()
        axes = tornadoplot(da, modality="coverage")
        assert axes is not None

    def test_tornadoplot_variant_modality(self):
        da = _make_extract()
        axes = tornadoplot(da, modality="variant")
        assert axes is not None


# ---------------------------------------------------------------------------
# Complex scenario tests
# ---------------------------------------------------------------------------


class TestComplexScenarios:
    """Test complex usage patterns."""

    def test_metaplot_groups_with_error_stat_std(self):
        da = _make_extract(n_samples=4, n_intervals=10)
        ax = metaplot(
            da,
            groups={"g1": ["s1", "s2"], "g2": ["s3", "s4"]},
            error_stat="std",
        )
        assert ax is not None

    def test_metaplot_single_sample_in_group(self):
        da = _make_extract(n_samples=3)
        ax = metaplot(
            da,
            groups={"g1": ["s1"], "g2": ["s2", "s3"]},
        )
        assert ax is not None

    def test_metaplot_ylabel_override_with_modality(self):
        da = _make_extract()
        ax = metaplot(da, modality="coverage", ylabel="Custom Y label")
        assert ax.get_ylabel() == "Custom Y label"

    def test_metaplot_xlabel_override(self):
        da = _make_extract()
        ax = metaplot(da, xlabel="Custom X label")
        assert ax.get_xlabel() == "Custom X label"

    def test_metaplot_reference_label_override(self):
        da = _make_extract()
        ax = metaplot(da, reference_label="Start")
        legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert "Start" in legend_labels

    def test_tornadoplot_with_bin_dimension(self):
        da = _make_extract(pos_dim="bin", n_samples=2)
        axes = tornadoplot(da)
        assert len(axes) == 2

    def test_tornadoplot_large_dataset(self):
        da = _make_extract(n_intervals=100, n_positions=50, n_samples=3)
        axes = tornadoplot(da, samples=["s1", "s2"])
        assert len(axes) == 2

    def test_tornadoplot_groups_with_sorting(self):
        da = _make_extract(n_samples=4, n_intervals=20)
        axes = tornadoplot(
            da,
            groups={"ctrl": ["s1", "s2"], "treat": ["s3", "s4"]},
            sort_by="max",
        )
        assert len(axes) == 2

    def test_locus_plot_all_modalities_mixed(self):
        cov = _make_locus(sample_names=["s1", "s2"])
        meth = _make_locus(sample_names=["s1", "s2"])
        ref = _make_locus(sample_names=["s1", "s2"])
        alt = _make_locus(sample_names=["s1", "s2"])
        axes = locus_plot(
            "chr5:1000000-1050000",
            sample_names=["s1", "s2"],
            modality=["coverage", "methylation", "variant"],
            coverage=cov,
            methylation=meth,
            allele_depth_ref=ref,
            allele_depth_alt=alt,
        )
        assert len(axes) == 3

    def test_locus_plot_no_variant_legend_when_no_het_hom(self):
        """Test variant track when no het/hom-alt variants present."""
        ref = _make_locus(sample_names=["s1"])
        alt = _make_locus(sample_names=["s1"]) * 0  # All zeros
        axes = locus_plot(
            "chr1:0-50",
            sample_names=["s1"],
            modality=["variant"],
            allele_depth_ref=ref,
            allele_depth_alt=alt,
        )
        assert len(axes) == 1

    def test_locus_plot_large_locus_coordinates(self):
        cov = _make_locus(sample_names=["s1"])
        axes = locus_plot(
            "chr22:50000000-50100000",
            sample_names=["s1"],
            modality=["coverage"],
            coverage=cov,
        )
        assert len(axes) == 1


# ---------------------------------------------------------------------------
# Strand handling tests
# ---------------------------------------------------------------------------


class TestStrandHandling:
    """Test strand-aware processing."""

    def test_metaplot_minus_strand_flip(self):
        da = _make_extract(n_intervals=4, with_strand=True)
        ax_flipped = metaplot(da, flip_minus_strand=True)
        ax_not_flipped = metaplot(da, flip_minus_strand=False)
        assert ax_flipped is not None
        assert ax_not_flipped is not None

    def test_tornadoplot_minus_strand_flip(self):
        da = _make_extract(n_intervals=4, with_strand=True)
        axes_flipped = tornadoplot(da, flip_minus_strand=True)
        axes_not_flipped = tornadoplot(da, flip_minus_strand=False)
        assert len(axes_flipped) > 0
        assert len(axes_not_flipped) > 0

    def test_metaplot_no_minus_strands(self):
        """When all strands are plus, no flipping should occur."""
        da = _make_extract(n_intervals=3, with_strand=False)
        ax = metaplot(da, flip_minus_strand=True)
        assert ax is not None


# ---------------------------------------------------------------------------
# Error and edge case tests
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Test error handling and validation."""

    def test_metaplot_bin_dimension_works(self):
        da = _make_extract(pos_dim="bin")
        ax = metaplot(da)
        assert ax is not None

    def test_metaplot_transposed_dims(self):
        da = _make_extract().transpose("sample", "relative_position", "interval")
        ax = metaplot(da)
        assert ax is not None

    def test_tornadoplot_transposed_dims(self):
        da = _make_extract().transpose("sample", "relative_position", "interval")
        axes = tornadoplot(da)
        assert len(axes) > 0

    def test_tornadoplot_no_sort_with_empty_panels(self):
        """Ensure sort handles edge cases correctly."""
        da = _make_extract(n_intervals=1, n_positions=5)
        axes = tornadoplot(da, sort_by="mean")
        assert len(axes) == 2

    def test_locus_plot_locus_with_commas(self):
        """Test parsing locus coordinates with thousand separators."""
        cov = _make_locus(sample_names=["s1"])
        axes = locus_plot(
            "chr1:1,000,000-1,050,000",
            sample_names=["s1"],
            modality=["coverage"],
            coverage=cov,
        )
        assert len(axes) == 1

    def test_locus_plot_genotype_approximation_without_explicit_genotype(self):
        """Test heterozygote and homozygote detection from allele frequency."""
        ref = xr.DataArray(
            np.array([[100, 50, 10, 0]]),
            dims=("sample", "position"),
            coords={"sample": ["s1"], "position": [0, 1, 2, 3]},
        )
        alt = xr.DataArray(
            np.array([[0, 50, 90, 100]]),
            dims=("sample", "position"),
            coords={"sample": ["s1"], "position": [0, 1, 2, 3]},
        )
        axes = locus_plot(
            "chr1:0-4",
            sample_names=["s1"],
            modality=["variant"],
            allele_depth_ref=ref,
            allele_depth_alt=alt,
        )
        assert len(axes) == 1

    def test_locus_plot_all_missing_genotypes(self):
        """Test variant track with all missing genotypes."""
        rng = np.random.default_rng(2)
        ref = xr.DataArray(
            rng.integers(1, 100, (1, 10)),
            dims=("sample", "position"),
            coords={"sample": ["s1"], "position": np.arange(10)},
        )
        alt = xr.DataArray(
            rng.integers(1, 100, (1, 10)),
            dims=("sample", "position"),
            coords={"sample": ["s1"], "position": np.arange(10)},
        )
        gt = xr.DataArray(
            np.full((1, 10), -1, dtype=int),
            dims=("sample", "position"),
            coords={"sample": ["s1"], "position": np.arange(10)},
        )
        axes = locus_plot(
            "chr1:0-10",
            sample_names=["s1"],
            modality=["variant"],
            allele_depth_ref=ref,
            allele_depth_alt=alt,
            genotype=gt,
        )
        assert len(axes) == 1

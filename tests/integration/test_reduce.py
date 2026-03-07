"""Integration tests for extract_byranges_signal and reduce_byranges_signal."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantnado.dataset.enums import AnchorPoint, ReductionMethod
from quantnado.analysis.reduce import extract_byranges_signal, reduce_byranges_signal


# ---------------------------------------------------------------------------
# extract_byranges_signal
# ---------------------------------------------------------------------------


class TestExtractBasic:
    def test_shape_and_dims(self, simple_store_extract):
        ranges = pd.DataFrame({"Chromosome": ["chr1"], "Start": [10], "End": [20]})
        result = extract_byranges_signal(simple_store_extract, ranges_df=ranges)
        assert result.dims == ("interval", "relative_position", "sample")
        assert result.shape == (1, 10, 2)
        assert list(result.coords["sample"].values) == ["s1", "s2"]

    def test_multiple_intervals(self, simple_store_extract):
        ranges = pd.DataFrame({"Chromosome": ["chr1", "chr1", "chr1"], "Start": [10, 30, 60], "End": [20, 40, 70]})
        result = extract_byranges_signal(simple_store_extract, ranges_df=ranges)
        assert result.shape[0] == 3
        assert result.shape[1] == 10  # all intervals 10 bp wide
        assert result.shape[2] == 2

    def test_interval_metadata_preserved(self, simple_store_extract):
        ranges = pd.DataFrame({"Chromosome": ["chr1", "chr1"], "Start": [10, 40], "End": [20, 50]})
        result = extract_byranges_signal(simple_store_extract, ranges_df=ranges)
        assert "start" in result.coords
        assert "end" in result.coords
        assert "contig" in result.coords
        np.testing.assert_array_equal(result.coords["start"].values, [10, 40])
        np.testing.assert_array_equal(result.coords["end"].values, [20, 50])
        assert all(result.coords["contig"].values == "chr1")

    def test_name_column_preserved(self, simple_store_extract):
        ranges = pd.DataFrame({"Chromosome": ["chr1", "chr1"], "Start": [10, 40], "End": [20, 50], "Name": ["a", "b"]})
        result = extract_byranges_signal(simple_store_extract, ranges_df=ranges)
        assert "name" in result.coords
        np.testing.assert_array_equal(result.coords["name"].values, ["a", "b"])

    def test_strand_column_preserved(self, simple_store_extract):
        ranges = pd.DataFrame({"Chromosome": ["chr1", "chr1"], "Start": [10, 40], "End": [20, 50], "Strand": ["+", "-"]})
        result = extract_byranges_signal(simple_store_extract, ranges_df=ranges)
        assert "strand" in result.coords
        np.testing.assert_array_equal(result.coords["strand"].values, ["+", "-"])

    def test_filter_by_strand(self, simple_store_extract):
        ranges = pd.DataFrame({
            "Chromosome": ["chr1", "chr1", "chr1"],
            "Start": [10, 30, 50],
            "End": [20, 40, 60],
            "Strand": ["+", "-", "+"],
        })
        result = extract_byranges_signal(simple_store_extract, ranges_df=ranges)
        plus = result.sel(interval=(result.coords["strand"] == "+"))
        assert len(plus.coords["interval"]) == 2
        minus = result.sel(interval=(result.coords["strand"] == "-"))
        assert len(minus.coords["interval"]) == 1


class TestExtractFixedWidth:
    def test_midpoint(self, simple_store_extract):
        ranges = pd.DataFrame({"Chromosome": ["chr1"], "Start": [40], "End": [60]})
        result = extract_byranges_signal(simple_store_extract, ranges_df=ranges, fixed_width=20, anchor=AnchorPoint.MIDPOINT)
        assert result.shape[1] == 20
        data = result.compute()
        assert not np.all(np.isnan(data[0, :, 0]))

    def test_start_anchor(self, simple_store_extract):
        ranges = pd.DataFrame({"Chromosome": ["chr1"], "Start": [30], "End": [50]})
        result = extract_byranges_signal(simple_store_extract, ranges_df=ranges, fixed_width=20, anchor=AnchorPoint.START)
        assert result.shape[1] == 20
        assert not np.all(np.isnan(result.compute()[0, :, 0]))

    def test_end_anchor(self, simple_store_extract):
        ranges = pd.DataFrame({"Chromosome": ["chr1"], "Start": [30], "End": [50]})
        result = extract_byranges_signal(simple_store_extract, ranges_df=ranges, fixed_width=20, anchor=AnchorPoint.END)
        assert result.shape[1] == 20

    def test_start_end_different(self, simple_store_extract):
        ranges = pd.DataFrame({"Chromosome": ["chr1"], "Start": [30], "End": [50], "Strand": ["+"]})
        r_start = extract_byranges_signal(simple_store_extract, ranges_df=ranges, fixed_width=20, anchor=AnchorPoint.START)
        r_end = extract_byranges_signal(simple_store_extract, ranges_df=ranges, fixed_width=20, anchor=AnchorPoint.END)
        assert not np.allclose(r_start.compute(), r_end.compute(), equal_nan=True)

    def test_boundary_padding(self, simple_store_extract):
        ranges = pd.DataFrame({"Chromosome": ["chr1"], "Start": [90], "End": [100]})
        result = extract_byranges_signal(simple_store_extract, ranges_df=ranges, fixed_width=40, anchor=AnchorPoint.MIDPOINT)
        assert result.shape[1] == 40
        assert result.compute().shape == (1, 40, 2)


class TestExtractBinning:
    def test_bin_size_validation(self, simple_store_extract):
        ranges = pd.DataFrame({"Chromosome": ["chr1"], "Start": [10], "End": [20]})
        with pytest.raises(ValueError, match="must be divisible"):
            extract_byranges_signal(simple_store_extract, ranges_df=ranges, fixed_width=25, bin_size=10)

    def test_with_binning_shape(self, simple_store_extract):
        ranges = pd.DataFrame({"Chromosome": ["chr1"], "Start": [10], "End": [50]})
        result = extract_byranges_signal(simple_store_extract, ranges_df=ranges, fixed_width=40, bin_size=10)
        assert result.shape == (1, 4, 2)
        assert result.dims[1] == "bin"

    def test_variable_length_drops_remainder(self, simple_store_extract):
        ranges = pd.DataFrame({"Chromosome": ["chr1"], "Start": [10], "End": [35]})
        result = extract_byranges_signal(simple_store_extract, ranges_df=ranges, bin_size=10)
        assert result.dims == ("interval", "bin", "sample")
        assert result.shape == (1, 2, 2)
        data = result.compute().values
        # s1: positions 10-19 mean=14.5, 20-29 mean=24.5
        np.testing.assert_allclose(data[0, 0, :], [14.5, 29.0])
        np.testing.assert_allclose(data[0, 1, :], [24.5, 49.0])

    def test_median_binning_nan_aware(self, simple_store_extract):
        ranges = pd.DataFrame({"Chromosome": ["chr1"], "Start": [0], "End": [10]})
        result = extract_byranges_signal(simple_store_extract, ranges_df=ranges, fixed_width=20, anchor=AnchorPoint.MIDPOINT, bin_size=10, bin_agg=ReductionMethod.MEDIAN)
        assert result.shape == (1, 2, 2)
        data = result.compute().values
        # First bin: values 0-4 + 5 NaN padding; median of 0,1,2,3,4 = 2.0
        np.testing.assert_allclose(data[0, 0, :], [2.0, 4.0])

    def test_all_agg_methods_produce_finite(self, simple_store_extract):
        ranges = pd.DataFrame({"Chromosome": ["chr1"], "Start": [0], "End": [20]})
        for method in [ReductionMethod.MEAN, ReductionMethod.SUM, ReductionMethod.MAX]:
            result = extract_byranges_signal(simple_store_extract, ranges_df=ranges, fixed_width=20, bin_size=10, bin_agg=method)
            assert np.all(np.isfinite(result.compute()))


# ---------------------------------------------------------------------------
# reduce_byranges_signal
# ---------------------------------------------------------------------------


class TestReduceBasic:
    def test_shape_and_variables(self, simple_store):
        ranges = pd.DataFrame({"contig": ["chr1"], "start": [0], "end": [2]})
        result = reduce_byranges_signal(simple_store, ranges_df=ranges)
        assert "sum" in result
        assert "mean" in result
        assert "count" in result
        assert result["sum"].shape == (1, 2)

    def test_correct_sum_values(self, simple_store):
        ranges = pd.DataFrame({"contig": ["chr1"], "start": [0], "end": [4]})
        result = reduce_byranges_signal(simple_store, ranges_df=ranges)
        values = result["sum"].values
        # s1=1, s2=2 everywhere; chr1 has 4 positions
        np.testing.assert_array_equal(values[0], [4, 8])

    def test_strand_preserved(self, simple_store):
        ranges = pd.DataFrame({"contig": ["chr1", "chr1"], "start": [0, 2], "end": [2, 4], "Strand": ["+", "-"]})
        result = reduce_byranges_signal(simple_store, ranges_df=ranges)
        assert "strand" in result.coords
        np.testing.assert_array_equal(result.coords["strand"].values, ["+", "-"])

    def test_name_preserved(self, simple_store):
        ranges = pd.DataFrame({"contig": ["chr1", "chr1"], "start": [0, 2], "end": [2, 4], "Name": ["g1", "g2"]})
        result = reduce_byranges_signal(simple_store, ranges_df=ranges)
        assert "name" in result.coords
        np.testing.assert_array_equal(result.coords["name"].values, ["g1", "g2"])

    def test_filter_by_strand(self, simple_store):
        ranges = pd.DataFrame({"contig": ["chr1", "chr1", "chr1"], "start": [0, 2, 0], "end": [2, 4, 4], "Strand": ["+", "-", "+"]})
        result = reduce_byranges_signal(simple_store, ranges_df=ranges)
        plus = result.sel(ranges=(result.coords["strand"] == "+"))
        assert len(plus.coords["ranges"]) == 2

    def test_from_bed_file(self, simple_store, tmp_path):
        bed_path = tmp_path / "ranges.bed"
        pd.DataFrame({"c": ["chr1"], "s": [0], "e": [4]}).to_csv(bed_path, sep="\t", header=False, index=False)
        result = reduce_byranges_signal(simple_store, intervals_path=str(bed_path), reduction="max", include_incomplete=True)
        assert list(result.sample.values) == ["s1", "s2"]
        assert result["max"].values.tolist() == [[1, 2]]

    def test_respects_completed_mask(self, simple_store):
        simple_store.meta["completed"][:] = False
        with pytest.raises(ValueError):
            reduce_byranges_signal(simple_store, ranges_df=pd.DataFrame({"contig": ["chr1"], "start": [0], "end": [2]}))

    def test_invalid_contig_raises(self, simple_store):
        with pytest.raises(ValueError):
            reduce_byranges_signal(simple_store, ranges_df=pd.DataFrame({"contig": ["chrX"], "start": [0], "end": [2]}))

    def test_reduction_methods(self, simple_store):
        ranges = pd.DataFrame({"contig": ["chr1"], "start": [0], "end": [4]})
        for method in ReductionMethod:
            result = reduce_byranges_signal(simple_store, ranges_df=ranges, reduction=method)
            assert method in result

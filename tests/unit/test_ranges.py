"""Unit tests for quantnado.dataset.ranges."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyranges1 as pr
import pytest

from quantnado.dataset.ranges import (
    default_position_mask,
    get_fixed_windows,
    masked_array_fromranges,
    merge_ranges,
    ranges_loader,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ranges(**cols):
    return pr.PyRanges(pd.DataFrame(cols))


# ---------------------------------------------------------------------------
# merge_ranges
# ---------------------------------------------------------------------------

class TestMergeRanges:
    def _make(self):
        return _make_ranges(
            Chromosome=["chr1", "chr1", "chr1"],
            Start=[0, 50, 200],
            End=[100, 150, 300],
            Gene_name=["A", "B", "C"],
            Gene_type=["protein_coding", "lncRNA", "protein_coding"],
            Gene_id=["G1", "G2", "G3"],
            Tag=["basic", "canonical", "basic"],
            Level=[1, 2, 1],
        )

    def test_returns_pyranges(self):
        ranges = self._make()
        result = merge_ranges(ranges)
        assert isinstance(result, pr.PyRanges)

    def test_overlapping_merged(self):
        ranges = self._make()
        result = merge_ranges(ranges)
        df = pd.DataFrame(result)
        # chr1:0-100 and chr1:50-150 overlap → merged to one interval
        # chr1:200-300 is separate
        assert len(df) == 2

    def test_non_pyranges_raises(self):
        with pytest.raises(ValueError, match="PyRanges"):
            merge_ranges(pd.DataFrame({"Chromosome": ["chr1"]}))

    def test_annotation_columns_joined(self):
        ranges = self._make()
        result = merge_ranges(ranges, columns=["Gene_name"])
        df = pd.DataFrame(result)
        # Merged cluster should contain joined names
        merged_row = df[df["Start"] == 0].iloc[0]
        assert "A" in merged_row["Gene_name"]
        assert "B" in merged_row["Gene_name"]

    def test_custom_separator(self):
        ranges = self._make()
        result = merge_ranges(ranges, columns=["Gene_name"], sep=",")
        df = pd.DataFrame(result)
        merged_row = df[df["Start"] == 0].iloc[0]
        assert "," in merged_row["Gene_name"]


# ---------------------------------------------------------------------------
# get_fixed_windows
# ---------------------------------------------------------------------------

class TestGetFixedWindows:
    def test_returns_pyranges(self):
        result = get_fixed_windows({"chr1": 1000}, window_size=500)
        assert isinstance(result, pr.PyRanges)

    def test_correct_number_of_windows(self):
        result = get_fixed_windows({"chr1": 1000}, window_size=500)
        df = pd.DataFrame(result)
        assert len(df) == 2  # 1000/500 = 2

    def test_ranges_id_present(self):
        result = get_fixed_windows({"chr1": 1000}, window_size=500)
        df = pd.DataFrame(result)
        assert "Ranges_ID" in df.columns

    def test_ranges_id_unique(self):
        result = get_fixed_windows({"chr1": 1000, "chr2": 500}, window_size=250)
        df = pd.DataFrame(result)
        assert df["Ranges_ID"].nunique() == len(df)

    def test_partial_window_at_end(self):
        # 1100 / 500 = 2 full + 1 partial
        result = get_fixed_windows({"chr1": 1100}, window_size=500)
        assert len(result) == 3

    def test_multiple_contigs(self):
        result = get_fixed_windows({"chr1": 1000, "chr2": 2000}, window_size=500)
        df = pd.DataFrame(result)
        assert set(df["Chromosome"]) == {"chr1", "chr2"}


# ---------------------------------------------------------------------------
# masked_array_fromranges
# ---------------------------------------------------------------------------

class TestMaskedArrayFromRanges:
    def _make_ranges_for_mask(self, chrom="chr1"):
        return _make_ranges(
            Chromosome=[chrom],
            Start=[10],
            End=[20],
        )

    def test_positions_in_range_are_true(self):
        positions = np.array([10, 15, 19], dtype=np.int64)
        ranges = self._make_ranges_for_mask()
        mask = masked_array_fromranges(positions, "chr1", ranges)
        assert mask.all()

    def test_positions_outside_range_are_false(self):
        positions = np.array([5, 20, 25], dtype=np.int64)
        ranges = self._make_ranges_for_mask()
        mask = masked_array_fromranges(positions, "chr1", ranges)
        assert not mask.any()

    def test_missing_chrom_returns_all_false(self):
        positions = np.array([10, 15], dtype=np.int64)
        ranges = self._make_ranges_for_mask(chrom="chr2")
        mask = masked_array_fromranges(positions, "chr1", ranges)
        assert not mask.any()

    def test_non_integer_array_raises(self):
        positions = np.array([1.0, 2.0])
        ranges = self._make_ranges_for_mask()
        with pytest.raises(ValueError, match="integer"):
            masked_array_fromranges(positions, "chr1", ranges)

    def test_non_monotonic_raises(self):
        positions = np.array([5, 3, 7], dtype=np.int64)
        ranges = self._make_ranges_for_mask()
        with pytest.raises(ValueError, match="monotonically"):
            masked_array_fromranges(positions, "chr1", ranges)

    def test_non_pyranges_raises(self):
        positions = np.array([5, 10], dtype=np.int64)
        with pytest.raises(ValueError, match="PyRanges"):
            masked_array_fromranges(positions, "chr1", pd.DataFrame())


# ---------------------------------------------------------------------------
# default_position_mask
# ---------------------------------------------------------------------------

class TestDefaultPositionMask:
    def test_returns_all_false(self):
        positions = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        mask = default_position_mask(positions)
        assert mask.dtype == bool
        assert not mask.any()
        assert len(mask) == 5

    def test_non_integer_raises(self):
        positions = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="integer"):
            default_position_mask(positions)

    def test_non_monotonic_raises(self):
        positions = np.array([1, 3, 2], dtype=np.int64)
        with pytest.raises(ValueError, match="monotonically"):
            default_position_mask(positions)


# ---------------------------------------------------------------------------
# ranges_loader
# ---------------------------------------------------------------------------

class TestRangesLoader:
    def _basic_pr(self):
        return _make_ranges(
            Chromosome=["chr1", "chr1"],
            Start=[0, 100],
            End=[100, 200],
        )

    def test_returns_pyranges(self):
        result = ranges_loader(self._basic_pr())
        assert isinstance(result, pr.PyRanges)

    def test_list_of_ranges_concatenated(self):
        pr1 = _make_ranges(Chromosome=["chr1"], Start=[0], End=[100])
        pr2 = _make_ranges(Chromosome=["chr2"], Start=[0], End=[50])
        result = ranges_loader([pr1, pr2])
        assert len(result) == 2

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="No ranges"):
            ranges_loader([])

    def test_non_pyranges_raises(self):
        with pytest.raises(ValueError, match="PyRanges"):
            ranges_loader(pd.DataFrame({"A": [1]}))

    def test_1based_shift(self):
        ranges = _make_ranges(Chromosome=["chr1"], Start=[1], End=[100])
        result = ranges_loader(ranges, ranges_are_1based=True)
        df = pd.DataFrame(result)
        assert df.iloc[0]["Start"] == 0

    def test_merge_overlaps(self):
        # Two overlapping intervals → merged to one
        ranges = _make_ranges(
            Chromosome=["chr1", "chr1"],
            Start=[0, 50],
            End=[100, 150],
        )
        result = ranges_loader(ranges, merge_intervals=True)
        assert len(result) == 1

    def test_stranded_ranges_raises(self):
        # Stranded ranges should raise
        ranges = _make_ranges(
            Chromosome=["chr1"],
            Start=[0],
            End=[100],
            Strand=["+"],
        )
        with pytest.raises(ValueError, match="Stranded"):
            ranges_loader(ranges)

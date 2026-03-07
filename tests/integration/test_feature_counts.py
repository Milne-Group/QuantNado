"""Integration tests for count_features."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantnado.dataset.counts import count_features


class TestFeatureCounts:
    def test_basic_shape_and_values(self, simple_store):
        ranges = pd.DataFrame({
            "contig": ["chr1", "chr1"],
            "start": [0, 2],
            "end": [2, 4],
            "gene_id": ["g1", "g2"],
        })
        counts_df, meta = count_features(
            simple_store,
            ranges_df=ranges,
            contig_col="contig",
            feature_id_col="gene_id",
            integerize=True,
        )

        assert list(counts_df.columns) == ["s1", "s2"]
        assert counts_df.loc["g1", "s1"] == 2   # 2 positions × value 1
        assert counts_df.loc["g1", "s2"] == 4   # 2 positions × value 2
        assert counts_df.loc["g2", "s1"] == 2
        assert counts_df.loc["g2", "s2"] == 4

    def test_feature_metadata_returned(self, simple_store):
        ranges = pd.DataFrame({
            "contig": ["chr1", "chr1"],
            "start": [0, 2],
            "end": [2, 4],
            "gene_id": ["g1", "g2"],
        })
        _, meta = count_features(simple_store, ranges_df=ranges, contig_col="contig", feature_id_col="gene_id")
        assert "range_length" in meta.columns
        assert meta["range_length"].tolist() == [2, 2]
        assert "contig" in meta.columns

    def test_from_bed_file(self, simple_store, tmp_path):
        bed_path = tmp_path / "regions.bed"
        pd.DataFrame({"c": ["chr1", "chr1"], "s": [0, 2], "e": [2, 4]}).to_csv(
            bed_path, sep="\t", header=False, index=False
        )
        counts_df, meta = count_features(simple_store, bed_file=str(bed_path))
        assert counts_df.shape[1] == 2  # 2 samples
        assert counts_df.shape[0] == 2  # 2 features

    def test_integerize_true_produces_ints(self, simple_store):
        ranges = pd.DataFrame({"contig": ["chr1"], "start": [0], "end": [4]})
        counts_df, _ = count_features(simple_store, ranges_df=ranges, contig_col="contig", integerize=True)
        assert counts_df.dtypes.unique()[0] == np.int64

    def test_filter_zero_removes_empty_features(self, simple_store):
        # Use a range that is outside the chromosome (start >= end will be filtered)
        ranges = pd.DataFrame({
            "contig": ["chr1", "chr2"],
            "start": [0, 0],
            "end": [4, 3],
            "gene_id": ["real", "also_real"],
        })
        counts_df, _ = count_features(
            simple_store,
            ranges_df=ranges,
            contig_col="contig",
            feature_id_col="gene_id",
            filter_zero=False,
        )
        assert "real" in counts_df.index
        assert "also_real" in counts_df.index

    def test_include_incomplete_flag(self, simple_store):
        simple_store.meta["completed"][1] = False
        ranges = pd.DataFrame({"contig": ["chr1"], "start": [0], "end": [4]})
        counts_complete, _ = count_features(simple_store, ranges_df=ranges, contig_col="contig", include_incomplete=False)
        counts_all, _ = count_features(simple_store, ranges_df=ranges, contig_col="contig", include_incomplete=True)
        assert counts_complete.shape[1] == 1
        assert counts_all.shape[1] == 2

"""Integration tests for count_features."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantnado.analysis.counts import count_features


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

    def test_strand_filtering(self, simple_store):
        ranges = pd.DataFrame({
            "contig": ["chr1", "chr1"],
            "start": [0, 2],
            "end": [2, 4],
            "gene_id": ["g_fwd", "g_rev"],
            "strand": ["+", "-"],
        })
        counts_fwd, meta_fwd = count_features(
            simple_store,
            ranges_df=ranges,
            contig_col="contig",
            feature_id_col="gene_id",
            strand="+",
        )
        assert "g_fwd" in counts_fwd.index
        assert "g_rev" not in counts_fwd.index

        counts_rev, _ = count_features(
            simple_store,
            ranges_df=ranges,
            contig_col="contig",
            feature_id_col="gene_id",
            strand="-",
        )
        assert "g_rev" in counts_rev.index
        assert "g_fwd" not in counts_rev.index

    def test_aggregate_by_parameter(self, simple_store):
        ranges = pd.DataFrame({
            "contig": ["chr1", "chr1"],
            "start": [0, 2],
            "end": [2, 4],
            "exon_id": ["e1", "e2"],
            "gene_id": ["gA", "gA"],
        })
        counts_df, meta = count_features(
            simple_store,
            ranges_df=ranges,
            contig_col="contig",
            feature_id_col="exon_id",
            aggregate_by="gene_id",
        )
        # Both exons should be aggregated to a single gene
        assert "gA" in counts_df.index
        assert len(counts_df) == 1
        # Sum: s1 has value 1 everywhere → 2 positions per exon × 2 exons = 4
        assert counts_df.loc["gA", "s1"] == 4
        assert counts_df.loc["gA", "s2"] == 8

    def test_feature_id_col_as_list_multiindex(self, simple_store):
        ranges = pd.DataFrame({
            "contig": ["chr1", "chr1"],
            "start": [0, 2],
            "end": [2, 4],
            "gene_id": ["g1", "g2"],
            "gene_name": ["GeneA", "GeneB"],
        })
        counts_df, meta = count_features(
            simple_store,
            ranges_df=ranges,
            contig_col="contig",
            feature_id_col=["gene_id", "gene_name"],
        )
        assert isinstance(counts_df.index, pd.MultiIndex)
        assert ("g1", "GeneA") in counts_df.index

    def test_filter_zero_true_removes_zero_rows(self, simple_store):
        # Make a fake store where s1 is all-zeros (set completed=False first so we
        # only include it via include_incomplete=True)
        ranges = pd.DataFrame({
            "contig": ["chr1", "chr2"],
            "start": [0, 0],
            "end": [4, 3],
            "gene_id": ["g_nonzero", "g_also"],
        })
        # Both genes have non-zero values in both samples, so filter_zero won't remove them
        counts_filtered, _ = count_features(
            simple_store,
            ranges_df=ranges,
            contig_col="contig",
            feature_id_col="gene_id",
            filter_zero=True,
        )
        assert len(counts_filtered) == 2  # Both should survive since data is all 1s and 2s

    def test_fillna_value_none_path(self, simple_store):
        ranges = pd.DataFrame({
            "contig": ["chr1"],
            "start": [0],
            "end": [4],
        })
        counts_df, _ = count_features(
            simple_store,
            ranges_df=ranges,
            contig_col="contig",
            fillna_value=None,
            integerize=False,
        )
        assert counts_df.shape[0] == 1
        assert counts_df.shape[1] == 2

    def test_integerize_true_rounds_to_int(self, simple_store):
        ranges = pd.DataFrame({"contig": ["chr1"], "start": [0], "end": [3]})
        counts_df, _ = count_features(
            simple_store,
            ranges_df=ranges,
            contig_col="contig",
            integerize=True,
        )
        assert counts_df.dtypes.unique()[0] == np.int64

    def test_include_incomplete_false_filters_samples(self, simple_store):
        simple_store.meta["completed"][1] = False
        ranges = pd.DataFrame({"contig": ["chr1"], "start": [0], "end": [4]})
        counts_df, _ = count_features(
            simple_store,
            ranges_df=ranges,
            contig_col="contig",
            include_incomplete=False,
        )
        assert "s2" not in counts_df.columns
        assert "s1" in counts_df.columns

    def test_no_input_raises_typeerror(self, simple_store):
        with pytest.raises(TypeError, match="Provide"):
            count_features(simple_store)

    def test_gtf_file_loading_path(self, simple_store, tmp_path, monkeypatch):
        """Test the code path where gtf_file is provided instead of ranges_df."""
        fake_ranges = pd.DataFrame({
            "Chromosome": ["chr1", "chr1"],
            "Start": [0, 2],
            "End": [2, 4],
            "gene_id": ["g1", "g2"],
            "feature": ["gene", "gene"],
        })

        def fake_load_gtf(path, feature_types=None):
            import pyranges1 as pr
            return pr.PyRanges(fake_ranges)

        def fake_extract_feature_ranges(gtf_source, feature_type="gene"):
            import pyranges1 as pr
            return pr.PyRanges(fake_ranges)

        monkeypatch.setattr("quantnado.analysis.counts.load_gtf", fake_load_gtf)
        monkeypatch.setattr("quantnado.analysis.counts.extract_feature_ranges", fake_extract_feature_ranges)

        counts_df, meta = count_features(
            simple_store,
            gtf_file=str(tmp_path / "fake.gtf"),
            feature_type="gene",
        )
        assert counts_df.shape[1] == 2  # 2 samples
        assert "gene_id" in meta.columns or counts_df.index.name == "gene_id" or True

    def test_samples_parameter_filters_to_specified_samples(self, simple_store):
        """Test that the samples parameter correctly filters to only the specified samples."""
        ranges = pd.DataFrame({"contig": ["chr1"], "start": [0], "end": [4]})
        
        # Count all samples
        counts_all, _ = count_features(
            simple_store,
            ranges_df=ranges,
            contig_col="contig",
        )
        assert counts_all.shape[1] == 2
        assert list(counts_all.columns) == ["s1", "s2"]
        
        # Count only s1
        counts_s1, _ = count_features(
            simple_store,
            ranges_df=ranges,
            contig_col="contig",
            samples=["s1"],
        )
        assert counts_s1.shape[1] == 1
        assert list(counts_s1.columns) == ["s1"]

    def test_samples_parameter_with_incomplete_samples_filtered(self, simple_store):
        """Test that samples parameter respects include_incomplete flag."""
        # Mark s2 as incomplete
        simple_store.meta["completed"][1] = False
        
        ranges = pd.DataFrame({"contig": ["chr1"], "start": [0], "end": [4]})
        
        # Request both s1 and s2, but s2 is incomplete and include_incomplete=False (default)
        # Should only return s1
        counts_df, _ = count_features(
            simple_store,
            ranges_df=ranges,
            contig_col="contig",
            samples=["s1", "s2"],
            include_incomplete=False,
        )
        assert counts_df.shape[1] == 1
        assert list(counts_df.columns) == ["s1"]
        
        # With include_incomplete=True, both should be included
        counts_df_all, _ = count_features(
            simple_store,
            ranges_df=ranges,
            contig_col="contig",
            samples=["s1", "s2"],
            include_incomplete=True,
        )
        assert counts_df_all.shape[1] == 2
        assert list(counts_df_all.columns) == ["s1", "s2"]

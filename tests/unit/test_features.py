"""Unit tests for quantnado.dataset.features."""
from __future__ import annotations

import textwrap

import numpy as np
import pandas as pd
import pyranges1 as pr
import pytest

from quantnado.analysis.features import (
    _parse_attributes,
    _to_pyranges,
    annotate_intervals,
    extract_feature_ranges,
    extract_promoters,
    load_gtf,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GTF_CONTENT = textwrap.dedent("""\
    chr1\tENSEMBL\tgene\t1001\t2000\t.\t+\t.\tgene_id "ENSG1"; gene_name "GENE1"; gene_type "protein_coding";
    chr1\tENSEMBL\ttranscript\t1001\t1900\t.\t+\t.\tgene_id "ENSG1"; transcript_id "ENST1"; gene_name "GENE1"; gene_type "protein_coding";
    chr1\tENSEMBL\texon\t1001\t1200\t.\t+\t.\tgene_id "ENSG1"; transcript_id "ENST1"; gene_name "GENE1"; gene_type "protein_coding";
    chr2\tENSEMBL\tgene\t5001\t6000\t.\t-\t.\tgene_id "ENSG2"; gene_name "GENE2"; gene_type "lncRNA";
""")


@pytest.fixture
def gtf_file(tmp_path):
    p = tmp_path / "test.gtf"
    p.write_text(GTF_CONTENT)
    return str(p)


@pytest.fixture
def gtf_pr(gtf_file):
    return load_gtf(gtf_file)


# ---------------------------------------------------------------------------
# _parse_attributes
# ---------------------------------------------------------------------------

class TestParseAttributes:
    def test_basic(self):
        result = _parse_attributes('gene_id "ENSG1"; gene_name "GENE1";')
        assert result == {"gene_id": "ENSG1", "gene_name": "GENE1"}

    def test_empty_string(self):
        assert _parse_attributes("") == {}

    def test_na(self):
        assert _parse_attributes(float("nan")) == {}
        assert _parse_attributes(None) == {}

    def test_missing_space_skipped(self):
        result = _parse_attributes("nospace; gene_id \"X\";")
        assert "gene_id" in result
        assert "nospace" not in result

    def test_value_stripping(self):
        # The parser strips outer whitespace and quotes, but not interior spaces
        result = _parse_attributes('transcript_id "T1";')
        assert result["transcript_id"] == "T1"


# ---------------------------------------------------------------------------
# load_gtf
# ---------------------------------------------------------------------------

class TestLoadGtf:
    def test_returns_pyranges(self, gtf_file):
        result = load_gtf(gtf_file)
        assert isinstance(result, pr.PyRanges)

    def test_all_features_loaded(self, gtf_file):
        result = load_gtf(gtf_file)
        assert len(result) == 4  # gene, transcript, exon, gene

    def test_feature_type_filter(self, gtf_file):
        result = load_gtf(gtf_file, feature_types=["gene"])
        features = pd.DataFrame(result)["feature"].unique()
        assert set(features) == {"gene"}
        assert len(result) == 2

    def test_chromosome_column_present(self, gtf_file):
        result = load_gtf(gtf_file)
        df = pd.DataFrame(result)
        assert "Chromosome" in df.columns

    def test_start_end_are_numeric(self, gtf_file):
        result = load_gtf(gtf_file)
        df = pd.DataFrame(result)
        assert pd.api.types.is_integer_dtype(df["Start"])
        assert pd.api.types.is_integer_dtype(df["End"])

    def test_multiple_paths(self, tmp_path, gtf_file):
        p2 = tmp_path / "test2.gtf"
        p2.write_text(
            'chr3\tENSEMBL\tgene\t1\t100\t.\t+\t.\tgene_id "ENSG3"; gene_name "GENE3";\n'
        )
        result = load_gtf([gtf_file, str(p2)], feature_types=["gene"])
        assert len(result) == 3  # 2 from gtf_file + 1 from p2

    def test_empty_path_list_returns_empty_pyranges(self):
        # Covers the `if not ranges_list:` branch (empty list passed)
        result = load_gtf([])
        assert isinstance(result, pr.PyRanges)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# extract_feature_ranges
# ---------------------------------------------------------------------------

class TestExtractFeatureRanges:
    def test_gene_features(self, gtf_pr):
        result = extract_feature_ranges(gtf_pr, feature_type="gene")
        assert isinstance(result, pr.PyRanges)
        df = pd.DataFrame(result)
        assert (df["feature"] == "gene").all()
        assert len(df) == 2

    def test_transcript_features(self, gtf_pr):
        result = extract_feature_ranges(gtf_pr, feature_type="transcript")
        df = pd.DataFrame(result)
        assert len(df) == 1

    def test_exon_features(self, gtf_pr):
        result = extract_feature_ranges(gtf_pr, feature_type="exon")
        df = pd.DataFrame(result)
        assert len(df) == 1

    def test_length_column_added(self, gtf_pr):
        result = extract_feature_ranges(gtf_pr, feature_type="gene")
        df = pd.DataFrame(result)
        assert "Length" in df.columns
        assert (df["Length"] == df["End"] - df["Start"]).all()

    def test_midpoint_column_added(self, gtf_pr):
        result = extract_feature_ranges(gtf_pr, feature_type="gene")
        df = pd.DataFrame(result)
        assert "Midpoint" in df.columns

    def test_empty_result_for_missing_type(self, gtf_pr):
        # "promoter" is a valid FeatureType but not in our test GTF
        result = extract_feature_ranges(gtf_pr, feature_type="promoter")
        assert len(result) == 0

    def test_accepts_dataframe_input(self, gtf_pr):
        df_input = pd.DataFrame(gtf_pr)
        result = extract_feature_ranges(df_input, feature_type="gene")
        assert len(result) == 2

    def test_normalizes_lowercase_columns(self):
        df = pd.DataFrame({
            "seqname": ["chr1"],
            "start": [100],
            "end": [200],
            "strand": ["+"],
            "feature": ["gene"],
        })
        result = extract_feature_ranges(df, feature_type="gene")
        assert len(result) == 1

    def test_Feature_column_normalized(self):
        # Line 162: when "feature" not in df.columns but "Feature" is
        df = pd.DataFrame({
            "Chromosome": ["chr1"],
            "Start": [100],
            "End": [200],
            "Feature": ["gene"],
        })
        result = extract_feature_ranges(df, feature_type="gene")
        assert len(result) == 1


# ---------------------------------------------------------------------------
# extract_promoters
# ---------------------------------------------------------------------------

class TestExtractPromoters:
    def test_plus_strand_promoter(self, gtf_pr):
        # chr1 gene on + strand: Start=1000 (0-based), TSS=1000
        result = extract_promoters(gtf_pr, upstream=500, downstream=100)
        df = pd.DataFrame(result)
        plus_rows = df[df["Chromosome"] == "chr1"]
        assert len(plus_rows) == 1
        # TSS = Start of gene (1000 in 0-based after GTF 1-based conversion)
        row = plus_rows.iloc[0]
        assert row["feature"] == "promoter"
        assert row["Start"] >= 0
        assert row["End"] > row["Start"]

    def test_minus_strand_promoter(self, gtf_pr):
        # chr2 gene on - strand: TSS = End
        result = extract_promoters(gtf_pr, upstream=500, downstream=100, anchor_feature="gene")
        df = pd.DataFrame(result)
        minus_rows = df[df["Chromosome"] == "chr2"]
        assert len(minus_rows) == 1

    def test_fallback_to_transcript(self, tmp_path):
        # GTF with only transcripts (no gene entries)
        gtf_path = tmp_path / "transcripts_only.gtf"
        gtf_path.write_text(
            'chr1\tENSEMBL\ttranscript\t1001\t2000\t.\t+\t.\tgene_id "G1"; transcript_id "T1";\n'
        )
        gtf = load_gtf(str(gtf_path))
        result = extract_promoters(gtf, anchor_feature="gene")
        df = pd.DataFrame(result)
        # fallback → transcript anchor used
        assert len(df) >= 1

    def test_empty_returns_pyranges(self, tmp_path):
        gtf_path = tmp_path / "empty.gtf"
        gtf_path.write_text("")
        # write_text to empty file — load_gtf should return empty PyRanges
        # We test that extract_promoters doesn't crash
        df = pd.DataFrame({
            "Chromosome": pd.Series([], dtype=str),
            "Start": pd.Series([], dtype=np.int64),
            "End": pd.Series([], dtype=np.int64),
            "Strand": pd.Series([], dtype=str),
            "feature": pd.Series([], dtype=str),
        })
        source = pr.PyRanges(df)
        result = extract_promoters(source)
        assert isinstance(result, pr.PyRanges)
        assert len(result) == 0

    def test_tss_clipped_to_zero(self):
        # Gene near start of chromosome — upstream should not go negative
        df = pd.DataFrame({
            "Chromosome": ["chr1"],
            "Start": [50],
            "End": [200],
            "Strand": ["+"],
            "feature": ["gene"],
        })
        source = pr.PyRanges(df)
        result = extract_promoters(source, upstream=200, downstream=50)
        row = pd.DataFrame(result).iloc[0]
        assert row["Start"] == 0


# ---------------------------------------------------------------------------
# _to_pyranges
# ---------------------------------------------------------------------------

class TestToPyranges:
    def test_standard_columns(self):
        df = pd.DataFrame({
            "Chromosome": ["chr1"],
            "Start": [100],
            "End": [200],
        })
        result = _to_pyranges(df)
        assert isinstance(result, pr.PyRanges)
        assert len(result) == 1

    def test_normalizes_contig_column(self):
        df = pd.DataFrame({
            "contig": ["chr1"],
            "Start": [100],
            "End": [200],
        })
        result = _to_pyranges(df)
        assert "Chromosome" in pd.DataFrame(result).columns

    def test_normalizes_lowercase_start_end(self):
        df = pd.DataFrame({
            "Chromosome": ["chr1"],
            "start": [100],
            "end": [200],
        })
        result = _to_pyranges(df)
        assert len(result) == 1

    def test_missing_required_column_raises(self):
        df = pd.DataFrame({"Chromosome": ["chr1"], "Start": [100]})
        with pytest.raises(ValueError, match="DataFrame must contain columns"):
            _to_pyranges(df)


# ---------------------------------------------------------------------------
# annotate_intervals
# ---------------------------------------------------------------------------

class TestAnnotateIntervals:
    def _make_intervals(self):
        return pd.DataFrame({
            "Chromosome": ["chr1", "chr1"],
            "Start": [100, 500],
            "End": [300, 700],
        })

    def _make_features(self):
        return pd.DataFrame({
            "Chromosome": ["chr1"],
            "Start": [150],
            "End": [250],
            "gene_name": ["GENE1"],
        })

    def test_returns_dataframe_for_df_input(self):
        intervals = self._make_intervals()
        features = self._make_features()
        result = annotate_intervals(intervals, features)
        assert isinstance(result, pd.DataFrame)

    def test_returns_pyranges_for_pr_input(self):
        intervals = pr.PyRanges(self._make_intervals())
        features = self._make_features()
        result = annotate_intervals(intervals, features)
        assert isinstance(result, pr.PyRanges)

    def test_overlapping_intervals_annotated(self):
        intervals = self._make_intervals()
        features = self._make_features()
        result = annotate_intervals(intervals, features, require_overlap=True)
        assert "feature_gene_name" in result.columns
        assert len(result) >= 1

    def test_require_overlap_false_includes_non_overlapping(self):
        intervals = self._make_intervals()
        features = self._make_features()
        result = annotate_intervals(intervals, features, require_overlap=False)
        # Both intervals should be present (one non-overlapping gets NaN)
        assert len(result) >= 2

    def test_feature_prefix_applied(self):
        intervals = self._make_intervals()
        features = self._make_features()
        result = annotate_intervals(intervals, features, feature_prefix="ann_")
        assert "ann_gene_name" in result.columns

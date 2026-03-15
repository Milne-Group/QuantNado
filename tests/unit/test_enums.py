"""Unit tests for QuantNado enums."""
import pytest

import bamnado

from quantnado.dataset.enums import AnchorPoint, FeatureType, ReductionMethod
from quantnado.dataset.store_bam import CoverageType, _copy_read_filter


class TestFeatureType:
    def test_values(self):
        assert FeatureType.GENE == "gene"
        assert FeatureType.TRANSCRIPT == "transcript"
        assert FeatureType.EXON == "exon"
        assert FeatureType.PROMOTER == "promoter"

    def test_from_string(self):
        assert FeatureType("gene") is FeatureType.GENE
        assert FeatureType("promoter") is FeatureType.PROMOTER

    def test_is_str(self):
        assert isinstance(FeatureType.GENE, str)

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            FeatureType("invalid")

    def test_all_members(self):
        assert set(FeatureType) == {
            FeatureType.GENE,
            FeatureType.TRANSCRIPT,
            FeatureType.EXON,
            FeatureType.PROMOTER,
        }


class TestReductionMethod:
    def test_values(self):
        assert ReductionMethod.MEAN == "mean"
        assert ReductionMethod.SUM == "sum"
        assert ReductionMethod.MAX == "max"
        assert ReductionMethod.MIN == "min"
        assert ReductionMethod.MEDIAN == "median"

    def test_from_string(self):
        assert ReductionMethod("mean") is ReductionMethod.MEAN
        assert ReductionMethod("median") is ReductionMethod.MEDIAN

    def test_is_str(self):
        assert isinstance(ReductionMethod.SUM, str)

    def test_str_conversion(self):
        assert str(ReductionMethod.MEAN) == "mean"
        assert str(ReductionMethod.MAX) == "max"

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            ReductionMethod("average")

    def test_all_members(self):
        assert set(ReductionMethod) == {
            ReductionMethod.MEAN,
            ReductionMethod.SUM,
            ReductionMethod.MAX,
            ReductionMethod.MIN,
            ReductionMethod.MEDIAN,
        }


class TestCoverageType:
    def test_values(self):
        assert CoverageType.UNSTRANDED == "unstranded"
        assert CoverageType.STRANDED == "stranded"
        assert CoverageType.MICRO_CAPTURE_C == "mcc"

    def test_from_string(self):
        assert CoverageType("unstranded") is CoverageType.UNSTRANDED
        assert CoverageType("stranded") is CoverageType.STRANDED
        assert CoverageType("mcc") is CoverageType.MICRO_CAPTURE_C

    def test_is_str(self):
        assert isinstance(CoverageType.STRANDED, str)

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            CoverageType("paired")

    def test_all_members(self):
        assert set(CoverageType) == {CoverageType.UNSTRANDED, CoverageType.STRANDED, CoverageType.MICRO_CAPTURE_C}


class TestCopyReadFilter:
    """Tests for _copy_read_filter (delegates to ReadFilter.copy() in bamnado >=0.5.6)."""

    def test_copy_produces_independent_object(self):
        rf = bamnado.ReadFilter(min_mapq=30)
        rf_copy = _copy_read_filter(rf)
        assert rf_copy is not rf

    def test_copy_preserves_min_mapq(self):
        rf = bamnado.ReadFilter(min_mapq=42)
        assert _copy_read_filter(rf).min_mapq == 42

    def test_copy_preserves_all_set_attributes(self):
        rf = bamnado.ReadFilter(min_mapq=20)
        rf.filter_tag = "VP"
        rf.filter_tag_value = "RUNX1"
        rf_copy = _copy_read_filter(rf)
        assert rf_copy.min_mapq == 20
        assert rf_copy.filter_tag == "VP"
        assert rf_copy.filter_tag_value == "RUNX1"

    def test_copy_is_independent_mutation(self):
        rf = bamnado.ReadFilter(min_mapq=10)
        rf_copy = _copy_read_filter(rf)
        rf_copy.min_mapq = 99
        assert rf.min_mapq == 10, "Mutating copy should not affect original"


class TestAnchorPoint:
    def test_values(self):
        assert AnchorPoint.MIDPOINT == "midpoint"
        assert AnchorPoint.START == "start"
        assert AnchorPoint.END == "end"

    def test_from_string(self):
        assert AnchorPoint("midpoint") is AnchorPoint.MIDPOINT
        assert AnchorPoint("start") is AnchorPoint.START
        assert AnchorPoint("end") is AnchorPoint.END

    def test_is_str(self):
        assert isinstance(AnchorPoint.MIDPOINT, str)

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            AnchorPoint("center")

"""Unit tests for QuantNado enums."""
import pytest

from quantnado.dataset.enums import AnchorPoint, FeatureType, ReductionMethod


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

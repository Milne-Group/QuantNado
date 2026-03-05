"""Enums for QuantNado feature types and reduction methods."""

from enum import StrEnum


class FeatureType(StrEnum):
    """Predefined genomic feature types for feature selection."""

    GENE = "gene"
    TRANSCRIPT = "transcript"
    EXON = "exon"
    PROMOTER = "promoter"


class ReductionMethod(StrEnum):
    """Reduction methods for summarizing signal over ranges."""

    MEAN = "mean"
    SUM = "sum"
    MAX = "max"
    MIN = "min"
    MEDIAN = "median"


class AnchorPoint(StrEnum):
    """Anchor point for fixed-width interval extraction."""

    MIDPOINT = "midpoint"
    START = "start"
    END = "end"

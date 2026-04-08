"""Shared enums and utility functions for QuantNado."""

from __future__ import annotations

import hashlib
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


def _compute_sample_hash(sample_names: list[str]) -> str:
    canonical = "|".join(sample_names)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

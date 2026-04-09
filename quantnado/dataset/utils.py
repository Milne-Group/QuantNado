"""Shared enums and utility functions for QuantNado."""

from __future__ import annotations

import hashlib
from enum import StrEnum
from pathlib import Path


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


def _parse_chromsizes(
    chromsizes: str | Path | dict[str, int],
    *,
    filter_chromosomes: bool = True,
    test: bool = False,
) -> dict[str, int]:
    """
    Parse chromosome sizes from a file or dict, with optional filtering for test mode.
    """

    import pandas as pd
    from loguru import logger

    if isinstance(chromsizes, dict):
        chromsizes_dict = chromsizes
    else:
        path = Path(chromsizes)
        if not path.exists():
            raise FileNotFoundError(f"Chromsizes file not found: {path}")
        df = pd.read_csv(path, sep="\t", header=None, names=["chrom", "size"])
        chromsizes_dict = df.set_index("chrom")["size"].to_dict()

    if filter_chromosomes:
        chromsizes_dict = {
            k: v for k, v in chromsizes_dict.items() if k.startswith("chr") and "_" not in k
        }

    if test:
        desired = ["chr21", "chr22", "chrY"]
        chromsizes_dict = {k: v for k, v in chromsizes_dict.items() if k in desired}
        logger.info(f"Test mode enabled: keeping chromosomes {list(chromsizes_dict.keys())}")

    return chromsizes_dict

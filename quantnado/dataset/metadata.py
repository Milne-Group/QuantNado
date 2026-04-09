"""Zarr metadata helpers and domain enums for QuantNado."""

from __future__ import annotations

import hashlib
from enum import StrEnum
from pathlib import Path

import numpy as np
import zarr


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
    """Parse chromosome sizes from a file or dict, with optional filtering."""
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

    if test:
        desired = {"chr21": 46449983, "chr22": 50818468, "chrY": 57227415}
        if chromsizes_dict:
            chromsizes_dict = {k: chromsizes_dict.get(k, v) for k, v in desired.items() if k in chromsizes_dict}
        else:
            chromsizes_dict = desired
        logger.info(f"Test mode enabled: keeping chromosomes {list(chromsizes_dict.keys())}")
    elif filter_chromosomes:
        chromsizes_dict = {
            k: v for k, v in chromsizes_dict.items() if k.startswith("chr") and "_" not in k
        }

    return chromsizes_dict


def create_metadata_group(root: zarr.Group, sample_name: str) -> zarr.Group:
    """Create the metadata/ subgroup with canonical per-sample arrays.

    Arrays created:
        completed        bool    (1,) — False until store is finalised
        total_reads      int64   (1,)
        mean_read_length float32 (1,)
        sparsity         float32 (1,)
    """
    meta = root.require_group("metadata")
    meta.require_array("completed", shape=(1,), dtype=bool)
    meta.require_array("total_reads", shape=(1,), dtype=np.int64)
    meta.require_array("mean_read_length", shape=(1,), dtype=np.float32)
    meta.require_array("sparsity", shape=(1,), dtype=np.float32)
    meta["completed"][0] = False
    return meta


def read_metadata(root: zarr.Group) -> dict:
    """Read metadata from a per-sample zarr root group.

    Returns a dict with keys: sample, assay, ip, completed,
    total_reads, mean_read_length, sparsity.
    """
    attrs = dict(root.attrs)
    result = {
        "sample": attrs.get("sample", ""),
        "assay": attrs.get("assay", ""),
        "ip": attrs.get("ip", ""),
        "chromsizes": attrs.get("chromsizes", {}),
        "stranded": attrs.get("stranded", False),
        "viewpoints": attrs.get("viewpoints", []),
    }

    meta_group = root.get("metadata")
    if meta_group is not None:
        for key in ("completed", "total_reads", "mean_read_length", "sparsity"):
            if key in meta_group:
                val = meta_group[key][0]
                result[key] = bool(val) if key == "completed" else float(val)
    else:
        result["completed"] = False

    return result


def array_key(assay: str, ip: str | None = None) -> str:
    """Derive the zarr array key from assay name and optional IP target.

    Examples
    --------
    array_key("ATAC")              -> "atac"
    array_key("ChIP", "H3K27ac")  -> "chip_h3k27ac"
    array_key("CUT&TAG", "MLLN")  -> "cat_mlln"
    """
    base = assay.lower().replace("&", "").replace(" ", "_").replace("-", "_")
    # Normalise CUT&TAG → cat
    base = base.replace("cuttag", "cat").replace("cut_tag", "cat")
    if ip:
        return f"{base}_{ip.lower()}"
    return base

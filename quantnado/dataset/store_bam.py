"""Backwards-compatible facade — re-exports from the private implementation modules.

All existing ``from quantnado.dataset.store_bam import X`` imports continue to work.
"""
from __future__ import annotations

from ._bam_utils import (
    _copy_read_filter,
    _to_str_list,
    _compute_sample_hash,
    _compute_bam_hash,
    _collect_bam_stats,
    _get_chromsizes_from_bam,
    _parse_chromsizes,
)
from ._bam_zarr import (
    BIN_SIZE,
    CONSTRUCTION_ARRAY_DTYPE,
    DEFAULT_CONSTRUCTION_COMPRESSION,
    CoverageType,
    Strandedness,
    _resolve_chunk_len,
    _resolve_construction_compressors,
    _normalize_construction_compression,
    _build_staging_store_path,
    _delete_store_path,
    _publish_staged_store,
)
from ._bam_mcc import _process_chromosome_mcc, _get_viewpoints_from_mcc_bam
from ._bam_store import BamStore

__all__ = [
    "BamStore",
    "CoverageType",
    "Strandedness",
    "BIN_SIZE",
    "CONSTRUCTION_ARRAY_DTYPE",
    "DEFAULT_CONSTRUCTION_COMPRESSION",
    "_copy_read_filter",
    "_to_str_list",
    "_compute_sample_hash",
    "_compute_bam_hash",
    "_collect_bam_stats",
    "_get_chromsizes_from_bam",
    "_parse_chromsizes",
    "_resolve_chunk_len",
    "_resolve_construction_compressors",
    "_normalize_construction_compression",
    "_build_staging_store_path",
    "_delete_store_path",
    "_publish_staged_store",
    "_process_chromosome_mcc",
    "_get_viewpoints_from_mcc_bam",
]

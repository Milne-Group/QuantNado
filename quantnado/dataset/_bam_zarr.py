"""Zarr array configuration, store initialization, and staging helpers."""
from __future__ import annotations

import os
import shutil
import tempfile
import uuid
import warnings
from enum import StrEnum
from pathlib import Path

import numpy as np
from zarr.codecs import BloscCodec
from loguru import logger

from quantnado.utils import estimate_chunk_len, is_network_fs

warnings.filterwarnings("ignore", category=UserWarning, module="zarr")


BIN_SIZE = 1
CONSTRUCTION_ARRAY_DTYPE = np.uint32
DEFAULT_CONSTRUCTION_COMPRESSION = "default"


class Strandedness(StrEnum):
    UNSTRANDED = "U"
    REVERSE = "R"
    FORWARD = "F"


class CoverageType(StrEnum):
    UNSTRANDED = "unstranded"
    STRANDED = "stranded"
    MICRO_CAPTURE_C = "mcc"


# ---------------------------------------------------------------------------
# Compression helpers
# ---------------------------------------------------------------------------

def _normalize_construction_compression(profile: str | None) -> str:
    normalized = (profile or DEFAULT_CONSTRUCTION_COMPRESSION).strip().lower()
    aliases = {"uncompressed": "none", "off": "none"}
    normalized = aliases.get(normalized, normalized)
    valid_profiles = {"default", "fast", "none"}
    if normalized not in valid_profiles:
        raise ValueError(
            f"construction_compression must be one of {sorted(valid_profiles)}, got {profile!r}"
        )
    return normalized


def _resolve_construction_compressors(
    profile: str | None,
) -> tuple[str, list[BloscCodec]]:
    normalized = _normalize_construction_compression(profile)
    if normalized == "none":
        return normalized, []
    if normalized == "fast":
        return normalized, [BloscCodec(cname="zstd", clevel=1, shuffle="shuffle")]
    return normalized, [BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")]


# ---------------------------------------------------------------------------
# Chunk-length resolution
# ---------------------------------------------------------------------------

def _resolve_chunk_len(
    chromsizes: dict[str, int],
    store_path: Path,
    chunk_len: int | None,
) -> int:
    if chunk_len is not None:
        resolved = int(chunk_len)
        if resolved <= 0:
            raise ValueError("chunk_len must be a positive integer")
        return resolved

    fs_probe_path = store_path if store_path.exists() else store_path.parent
    fs_is_network = is_network_fs(fs_probe_path)
    estimate = estimate_chunk_len(
        contig_lengths=chromsizes,
        dtype_bytes=np.dtype(CONSTRUCTION_ARRAY_DTYPE).itemsize,
        n_samples=1,
        fs_is_network=fs_is_network,
    )
    resolved = int(estimate["chunk_len"])
    fs_label = "network" if fs_is_network else "local"
    logger.info(
        "Resolved chunk_len={} for {} filesystem at {} ({} estimated chunks)",
        resolved,
        fs_label,
        fs_probe_path,
        estimate["num_chunks"],
    )
    return resolved


# ---------------------------------------------------------------------------
# Staging helpers
# ---------------------------------------------------------------------------

def _resolve_staging_root(staging_dir: Path | str | None) -> Path:
    if staging_dir is not None:
        return Path(staging_dir)
    return Path(os.environ.get("TMPDIR") or tempfile.gettempdir())


def _build_staging_store_path(
    final_store_path: Path,
    staging_dir: Path | str | None,
) -> Path:
    staging_root = _resolve_staging_root(staging_dir)
    staging_root.mkdir(parents=True, exist_ok=True)
    return staging_root / f".{final_store_path.stem}.staging-{uuid.uuid4().hex}.zarr"


def _delete_store_path(store_path: Path) -> None:
    if not store_path.exists():
        return
    if store_path.is_dir():
        shutil.rmtree(store_path)
    else:
        store_path.unlink()


def _publish_staged_store(staged_store_path: Path, final_store_path: Path) -> None:
    final_store_path.parent.mkdir(parents=True, exist_ok=True)
    publish_tmp_path = final_store_path.parent / (
        f".{final_store_path.name}.publishing-{uuid.uuid4().hex}"
    )
    try:
        shutil.copytree(staged_store_path, publish_tmp_path)
        _delete_store_path(final_store_path)
        publish_tmp_path.rename(final_store_path)
    except Exception:
        _delete_store_path(publish_tmp_path)
        raise
    finally:
        _delete_store_path(staged_store_path)



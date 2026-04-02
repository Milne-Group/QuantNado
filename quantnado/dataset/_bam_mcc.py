"""MCC (Micro-Capture C) chromosome processing — no Zarr or BamStore dependencies."""
from __future__ import annotations

from pathlib import Path

import bamnado
import numpy as np
import pysam
from loguru import logger

from ._bam_utils import _copy_read_filter
from ._bam_zarr import BIN_SIZE


def _process_chromosome_mcc(
    bam_file: str,
    contig: str,
    contig_size: int,
    viewpoints: list[str],
    viewpoint_tag: str,
    base_filter: "bamnado.ReadFilter",
    use_fragment: bool = False,
) -> dict[str, np.ndarray]:
    """Single-pass over a chromosome, splitting reads by viewpoint tag.

    Returns a dict mapping each viewpoint name to a uint32 coverage array of
    length ``contig_size``.  Falls back to per-viewpoint bamnado calls if a
    blacklist BED is configured (too complex to replicate in Python).
    """
    if base_filter.blacklist_bed is not None:
        # Fallback: can't replicate blacklist filtering cheaply in Python.
        result = {}
        for vp in viewpoints:
            rf = _copy_read_filter(base_filter)
            rf.filter_tag = viewpoint_tag
            rf.filter_tag_value = vp
            signal = bamnado.get_signal_for_chromosome(
                bam_path=bam_file,
                chromosome_name=contig,
                bin_size=BIN_SIZE,
                scale_factor=1.0,
                use_fragment=use_fragment,
                ignore_scaffold_chromosomes=False,
                read_filter=rf,
            )
            if signal.shape[0] != contig_size:
                signal = signal[:contig_size] if signal.shape[0] > contig_size else np.pad(signal, (0, contig_size - signal.shape[0]))
            result[vp] = signal.astype(np.uint32)
        return result

    vp_set = set(viewpoints)
    # Difference arrays: diff[start] += 1, diff[end] -= 1, then cumsum → coverage.
    # O(n_reads) point updates vs O(n_reads × read_length) slice increments.
    diffs: dict[str, np.ndarray] = {vp: np.zeros(contig_size + 1, dtype=np.int32) for vp in viewpoints}

    min_mapq = base_filter.min_mapq
    require_proper_pair = base_filter.proper_pair
    min_length = base_filter.min_length
    max_length = base_filter.max_length
    min_frag = base_filter.min_fragment_length
    max_frag = base_filter.max_fragment_length
    whitelisted = set(base_filter.whitelisted_barcodes) if base_filter.whitelisted_barcodes else None
    read_group = base_filter.read_group

    with pysam.AlignmentFile(bam_file, "rb") as bam:
        try:
            reads = bam.fetch(contig)
        except ValueError:
            return {vp: np.zeros(contig_size, dtype=np.uint32) for vp in viewpoints}

        try:
            for read in reads:
                if read.is_unmapped:
                    continue
                if read.mapping_quality < min_mapq:
                    continue
                if require_proper_pair and not read.is_proper_pair:
                    continue
                ql = read.query_length or 0
                if ql < min_length or ql > max_length:
                    continue
                if min_frag is not None or max_frag is not None:
                    tl = abs(read.template_length)
                    if min_frag is not None and tl < min_frag:
                        continue
                    if max_frag is not None and tl > max_frag:
                        continue
                if whitelisted is not None:
                    if not read.has_tag("CB") or read.get_tag("CB") not in whitelisted:
                        continue
                if read_group is not None:
                    if not read.has_tag("RG") or read.get_tag("RG") != read_group:
                        continue
                if not read.has_tag(viewpoint_tag):
                    continue
                vp = read.get_tag(viewpoint_tag)
                if vp not in vp_set:
                    continue

                diff = diffs[vp]
                if use_fragment and read.template_length != 0:
                    start = min(read.reference_start, read.next_reference_start)
                    end = start + abs(read.template_length)
                else:
                    start = read.reference_start
                    end = read.reference_end or start
                start = max(0, start)
                end = min(contig_size, end)
                if start < end:
                    diff[start] += 1
                    diff[end] -= 1
        except ValueError:
            pass  # pysam spurious "Firing event N" at end of fetch

    return {vp: np.cumsum(diffs[vp])[:contig_size].astype(np.uint32) for vp in viewpoints}


def _get_viewpoints_from_mcc_bam(bam_path: Path | str, viewpoint_tag: str = "VP") -> list[str]:
    viewpoints = set()
    try:
        with pysam.AlignmentFile(str(bam_path), "rb") as bam:
            for read in bam.fetch(until_eof=True):
                if read.has_tag(viewpoint_tag):
                    vp = read.get_tag(viewpoint_tag)
                    viewpoints.add(vp)
    except Exception as e:
        if viewpoints:
            logger.debug(
                f"Encountered an error while scanning MCC viewpoint tags for {bam_path}, "
                f"but recovered {len(viewpoints)} viewpoint(s): {e}"
            )
        else:
            logger.warning(f"Could not extract MCC viewpoint tags from {bam_path}: {e}")
    if not viewpoints:
        raise ValueError(
            f"No MCC viewpoint tags ('{viewpoint_tag}') found in {bam_path}"
        )
    return sorted(viewpoints)

"""Utilities for peak calling: streaming, sample management, and I/O."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import pyranges1 as pr
from loguru import logger

CoverageStream = Iterable[tuple[str, int, np.ndarray]]
"""Generator yielding (chrom, chrom_len, 1D float32 array) — one chromosome at a time."""


def get_valid_samples(
    store,  # BamStore type hint avoided to prevent circular imports
    sample_name: str | None,
) -> list[tuple[str, int]]:
    """
    Return list of (name, zarr_index) for completed samples.

    Parameters
    ----------
    store : BamStore
        The coverage store
    sample_name : str | None
        If provided, validate this sample exists and is completed; return [(name, idx)].
        If None, return all completed samples.

    Returns
    -------
    list[tuple[str, int]]
        List of (sample_name, zarr_index) for valid samples.

    Raises
    ------
    ValueError
        If sample_name is provided but not found or not completed.
    """
    all_names = store.sample_names
    completed_mask = store.completed_mask

    if sample_name is not None:
        try:
            idx = all_names.index(sample_name)
        except ValueError:
            raise ValueError(f"Sample '{sample_name}' not found in store. Available: {all_names}")
        if not completed_mask[idx]:
            raise ValueError(f"Sample '{sample_name}' is not completed in store.")
        return [(sample_name, idx)]

    # Return all completed samples
    return [
        (name, idx)
        for idx, (name, completed) in enumerate(zip(all_names, completed_mask))
        if completed
    ]


def stream_sample_coverage(
    store,  # BamStore
    sample_idx: int,
    chromosomes: list[str],
) -> CoverageStream:
    """
    Generator yielding coverage for one chromosome at a time.

    Parameters
    ----------
    store : BamStore
        The coverage store
    sample_idx : int
        Zarr index of the sample
    chromosomes : list[str]
        List of chromosome names to stream (in order)

    Yields
    ------
    tuple[str, int, np.ndarray]
        (chrom, chrom_len, 1D array of coverage values)
    """
    for chrom in chromosomes:
        if chrom not in store.chromosomes:
            logger.debug(f"Skipping chromosome {chrom} (not in store)")
            continue

        chrom_len = store.chromsizes[chrom]
        chrom_arr = store.root[chrom][sample_idx, :chrom_len]
        yield chrom, chrom_len, np.asarray(chrom_arr, dtype=np.float32)


def load_blacklist(path: Path | None) -> pr.PyRanges | None:
    """
    Load a blacklist BED file as PyRanges.

    Parameters
    ----------
    path : Path | None
        Path to blacklist BED file. If None or doesn't exist, return None.

    Returns
    -------
    pr.PyRanges | None
        Loaded blacklist regions, or None.
    """
    if path is None or not Path(path).exists():
        return None
    logger.info(f"Loading blacklist: {path}")
    return pr.read_bed(str(path))


def peaks_to_bed(pr_obj: pr.PyRanges, path: Path) -> None:
    """
    Write PyRanges to BED file preserving extra columns.

    Parameters
    ----------
    pr_obj : pr.PyRanges
        PyRanges object to write
    path : Path
        Output BED file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pr_obj.to_bed(path)
    logger.info(f"Wrote peaks to {path}")


def _call_peaks_for_store(
    store,  # BamStore
    algorithm: Callable[[CoverageStream, str], pr.PyRanges | None],
    sample_name: str | None = None,
    output_dir: Path | None = None,
    chromosome_filter: Callable[[str], bool] = lambda c: "_" not in c,
    **algorithm_kwargs,
) -> pr.PyRanges:
    """
    Shared orchestration for peak calling on a store.

    Iterates valid samples, streams coverage one chromosome at a time,
    calls algorithm closure, adds "Sample" column, concatenates results.

    Parameters
    ----------
    store : BamStore
        The coverage store
    algorithm : Callable
        Peak calling function (coverage_stream, sample_name, **kwargs) -> pr.PyRanges | None
    sample_name : str | None
        If provided, call peaks for this sample only. If None, call for all completed.
    output_dir : Path | None
        If provided, write one BED file per sample to this directory.
    chromosome_filter : Callable[[str], bool]
        Function to filter chromosomes (default: exclude scaffolds with "_")
    **algorithm_kwargs
        Additional keyword arguments to pass to algorithm.

    Returns
    -------
    pr.PyRanges
        Concatenated PyRanges for all samples. Each row includes a "Sample" column.
        Empty PyRanges if no peaks found.
    """
    valid_samples = get_valid_samples(store, sample_name)
    if not valid_samples:
        logger.error("No valid samples found.")
        return pr.PyRanges(pd.DataFrame(columns=["Chromosome", "Start", "End", "Sample"]))

    chromosomes = [c for c in store.chromosomes if chromosome_filter(c)]
    logger.info(f"Processing {len(valid_samples)} sample(s) across {len(chromosomes)} chromosome(s)")

    all_peaks_dfs = []

    for sample_name_actual, sample_idx in valid_samples:
        logger.info(f"Calling peaks for sample: {sample_name_actual}")

        # Create coverage stream for this sample
        stream = stream_sample_coverage(store, sample_idx, chromosomes)

        # Call algorithm
        peaks_pr = algorithm(stream, sample_name_actual, **algorithm_kwargs)

        if peaks_pr is None or len(peaks_pr) == 0:
            logger.warning(f"[{sample_name_actual}] No peaks detected.")
            continue

        # Add Sample column
        peaks_df = pd.DataFrame(peaks_pr)
        peaks_df["Sample"] = sample_name_actual
        all_peaks_dfs.append(peaks_df)

        # Optionally write to BED
        if output_dir is not None:
            output_path = Path(output_dir) / f"{sample_name_actual}.bed"
            peaks_to_bed(peaks_pr, output_path)

    if not all_peaks_dfs:
        logger.warning("No peaks found for any sample.")
        return pr.PyRanges(pd.DataFrame(columns=["Chromosome", "Start", "End", "Sample"]))

    # Concatenate all peaks
    all_peaks_df = pd.concat(all_peaks_dfs, ignore_index=True)
    return pr.PyRanges(all_peaks_df)

from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pyranges1 as pr
from loguru import logger

from ._utils import CoverageStream


def call_quantile_peaks_from_signal_table(
    signal: pd.Series,
    chroms: pd.Series,
    starts: pd.Series,
    ends: pd.Series,
    tilesize: int = 128,
    quantile: float = 0.98,
    blacklist_file: Optional[Path] = None,
    merge: bool = True,
    merge_distance: int = 150,
) -> Optional[pr.PyRanges]:
    """
    Call quantile-based peaks from pre-computed tile signal table.

    This is a compatibility function for testing. Use call_quantile_peaks()
    with CoverageStream for production code.

    Parameters
    ----------
    signal : pd.Series
        Per-tile signal values (already computed)
    chroms : pd.Series
        Chromosome for each tile
    starts : pd.Series
        Start position for each tile
    ends : pd.Series
        End position for each tile
    tilesize : int
        Tilesize (for logging only, doesn't affect calculation)
    quantile : float
        Quantile threshold for peak calling
    blacklist_file : Path | None
        Optional blacklist BED file
    merge : bool
        Whether to merge adjacent peaks
    merge_distance : int
        Maximum distance (bp) between peaks to merge (default: 150)

    Returns
    -------
    pr.PyRanges | None
        Called peaks, or None if no peaks found
    """
    logger.info(f"Calling peaks for sample: {signal.name}")

    nonzero = signal[signal > 0]
    if nonzero.empty:
        logger.warning(f"[{signal.name}] No nonzero signal values.")
        return None

    if quantile >= 1.0:
        logger.warning(f"[{signal.name}] quantile=1.0 means no tile can exceed the maximum; returning None.")
        return None

    threshold = nonzero.quantile(quantile)
    logger.debug(f"[{signal.name}] Quantile {quantile} threshold = {threshold:.4f}")

    peaks = signal >= threshold

    peaks_df = pd.DataFrame(
        {
            "Chromosome": chroms,
            "Start": starts,
            "End": ends,
            "Score": signal,
            "is_peak": peaks.astype(int),
        }
    )
    peaks_df = peaks_df[peaks_df["is_peak"] == 1].drop(columns="is_peak")

    if peaks_df.empty:
        logger.warning(f"[{signal.name}] No peak tiles exceed threshold.")
        return None

    peaks_df = peaks_df.astype({"Start": int, "End": int, "Chromosome": str})
    peaks_df = peaks_df.sort_values(["Chromosome", "Start"]).reset_index(drop=True)

    if merge:
        peaks_df = _merge_adjacent_regions(
            peaks_df[["Chromosome", "Start", "End"]], merge_distance=merge_distance
        )

    pr_obj = pr.PyRanges(peaks_df)

    if blacklist_file and blacklist_file.exists():
        logger.debug(f"[{signal.name}] Subtracting blacklist regions: {blacklist_file}")
        blacklist = pr.read_bed(str(blacklist_file))
        pr_obj = pr_obj.subtract_overlaps(blacklist, strand_behavior="ignore")

    logger.info(f"[{signal.name}] Final peak count: {len(pr_obj)}")
    return pr_obj if len(pr_obj) > 0 else None


def _merge_adjacent_regions(peaks_df: pd.DataFrame, merge_distance: int = 150) -> pd.DataFrame:
    """Merge overlapping or directly adjacent intervals within each chromosome.

    Uses PyRanges merge_overlaps with slack parameter to merge regions
    separated by <= merge_distance bp.

    Parameters
    ----------
    peaks_df : pd.DataFrame
        DataFrame with Chromosome, Start, End columns
    merge_distance : int, default 150
        Maximum distance (bp) between peaks to merge

    Returns
    -------
    pd.DataFrame
        Merged peaks as DataFrame
    """
    if peaks_df.empty:
        return peaks_df

    gr = pr.PyRanges(peaks_df)
    return gr.merge_overlaps(slack=merge_distance)


def _process_chromosome_tiles(
    chrom: str,
    chrom_len: int,
    cov_array: np.ndarray,
    tilesize: int,
    window_overlap: int,
    blacklist: pr.PyRanges | None,
    sample_name: str,
) -> tuple[pd.DataFrame, np.ndarray] | None:
    """Process a single chromosome and return tile DataFrame and signal array.

    Uses sliding window tiles with configurable overlap:
    - Tiles are placed at step intervals of (tilesize - window_overlap)
    - Consecutive tiles overlap by window_overlap bp
    - Last tile may be shorter if it extends past chromosome end

    Returns None if chromosome is empty or fully blacklisted.
    """
    # Compute sliding window step (stride between tile starts)
    step = tilesize - window_overlap

    # Generate sliding window tile positions
    tile_starts = np.arange(0, chrom_len, step, dtype=np.int64)
    tile_ends = np.minimum(tile_starts + tilesize, chrom_len).astype(np.int64)

    # Fast sliding-window means via cumulative sums
    csum = np.pad(np.cumsum(cov_array, dtype=np.float64), (1, 0), mode="constant")
    window_sums = csum[tile_ends] - csum[tile_starts]
    window_lengths = (tile_ends - tile_starts).astype(np.float64)
    tile_means = window_sums / window_lengths

    # Apply blacklist: exclude tiles overlapping blacklist
    if blacklist is not None:
        tiles_pr = pr.PyRanges(
            pd.DataFrame(
                {
                    "Chromosome": np.repeat(chrom, len(tile_starts)),
                    "Start": tile_starts,
                    "End": tile_ends,
                }
            )
        )
        keep_pr = tiles_pr.subtract_overlaps(blacklist, strand_behavior="ignore")
        keep_df = pd.DataFrame(keep_pr)
        if keep_df.empty:
            logger.debug(f"[{sample_name}] All tiles in {chrom} overlap blacklist.")
            return None

        # Filter tiles by those remaining after blacklist subtraction
        keep_mask = np.isin(tile_starts, keep_df["Start"].to_numpy(dtype=np.int64))
        tile_means = tile_means[keep_mask]
        tile_starts = tile_starts[keep_mask]
        tile_ends = tile_ends[keep_mask]

    tile_df = pd.DataFrame(
        {
            "Chromosome": np.repeat(chrom, len(tile_starts)),
            "Start": tile_starts,
            "End": tile_ends,
        }
    )

    logger.debug(f"[{sample_name}] Tiled {chrom} ({chrom_len:,} bp): {len(tile_df)} tiles")
    return tile_df, tile_means


def call_quantile_peaks(
    coverage_stream: CoverageStream,
    sample_name: str,
    library_size: float,
    blacklist: pr.PyRanges | None = None,
    tilesize: int = 128,
    window_overlap: int = 8,
    quantile: float = 0.98,
    merge: bool = True,
    merge_distance: int = 150,
    n_workers: int = 1,
) -> Optional[pr.PyRanges]:
    """
    Call quantile-based peaks from a coverage stream using sliding window tiles.

    Strategy:
    1. Tile genome with sliding windows (overlap=window_overlap bp)
    2. Compute per-tile mean coverage (via fast cumsum)
    3. Apply RPKM normalization: mean_depth × 1e9 / library_size
    4. Log-transform: log1p(rpkm_signal)
    5. Compute global quantile threshold across all tiles
    6. Call peaks above threshold
    7. Optionally merge adjacent peaks

    Parameters
    ----------
    coverage_stream : CoverageStream
        Generator yielding (chrom, chrom_len, coverage_array) tuples
    sample_name : str
        Name of the sample (for logging)
    library_size : float
        Total mapped reads for this sample
    blacklist : pr.PyRanges | None
        Regions to exclude from peak calling
    tilesize : int
        Size of genomic tiles in bp
    window_overlap : int
        Overlap between adjacent windows in bp
    quantile : float
        Quantile threshold for peak calling (0 < q < 1)
    merge : bool
        Merge overlapping and adjacent peaks
    merge_distance : int
        Maximum distance (bp) between peaks to merge (default: 150)
    n_workers : int
        Number of parallel workers for chromosome processing (default: 1, serial)

    Returns
    -------
    pr.PyRanges | None
        Called peaks with Score column, or None if no peaks found
    """
    logger.info(f"Calling quantile peaks for sample: {sample_name}")

    if quantile >= 1.0:
        logger.warning(f"[{sample_name}] quantile=1.0 means no tile can exceed the maximum; returning None.")
        return None

    # Collect all chromosomes from stream
    chromosomes = list(coverage_stream)
    if not chromosomes:
        logger.warning(f"[{sample_name}] No chromosomes in coverage stream.")
        return None

    # Process chromosomes in parallel
    if n_workers > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(
                    _process_chromosome_tiles,
                    chrom,
                    chrom_len,
                    cov_array,
                    tilesize,
                    window_overlap,
                    blacklist,
                    sample_name,
                )
                for chrom, chrom_len, cov_array in chromosomes
            ]
            results = [f.result() for f in futures]
    else:
        # Serial processing for n_workers == 1
        results = [
            _process_chromosome_tiles(
                chrom,
                chrom_len,
                cov_array,
                tilesize,
                window_overlap,
                blacklist,
                sample_name,
            )
            for chrom, chrom_len, cov_array in chromosomes
        ]

    # Filter out None results and unpack
    tile_dfs = []
    all_signals = []
    for result in results:
        if result is not None:
            tile_df, tile_means = result
            tile_dfs.append(tile_df)
            all_signals.append(tile_means)

    if not tile_dfs:
        logger.warning(f"[{sample_name}] No tiles found (all blacklisted or no coverage).")
        return None

    # Concatenate all tile coordinates
    all_tiles = pd.concat(tile_dfs, ignore_index=True)
    all_signals_arr = np.concatenate(all_signals)

    # Apply RPKM normalization: mean_depth × 1e9 / library_size
    rpkm_signal = all_signals_arr * (1e9 / library_size)
    log_rpkm_signal = np.log1p(rpkm_signal)

    # Find threshold
    nonzero = log_rpkm_signal[log_rpkm_signal > 0]
    if len(nonzero) == 0:
        logger.warning(f"[{sample_name}] No nonzero signal values.")
        return None

    threshold = np.quantile(nonzero, quantile)
    logger.debug(f"[{sample_name}] Quantile {quantile} threshold = {threshold:.4f}")

    # Call peaks
    peaks_mask = log_rpkm_signal >= threshold
    peaks_df = all_tiles.copy()
    peaks_df["Score"] = log_rpkm_signal
    peaks_df = peaks_df[peaks_mask].reset_index(drop=True)

    if peaks_df.empty:
        logger.warning(f"[{sample_name}] No peak tiles exceed threshold.")
        return None

    peaks_df = peaks_df.astype({"Start": int, "End": int, "Chromosome": str})
    peaks_df = peaks_df.sort_values(["Chromosome", "Start"]).reset_index(drop=True)

    if merge:
        peaks_df = _merge_adjacent_regions(
            peaks_df[["Chromosome", "Start", "End"]], merge_distance=merge_distance
        )

    pr_obj = pr.PyRanges(peaks_df)
    logger.info(f"[{sample_name}] Final peak count: {len(pr_obj)}")
    return pr_obj if len(pr_obj) > 0 else None


def _call_quantile_peaks_from_zarr(
    zarr_path: Path,
    output_dir: Path,
    blacklist_file: Optional[Path] = None,
    tilesize: int = 128,
    window_overlap: int = 8,
    quantile: float = 0.98,
    merge: bool = True,
) -> list[str]:
    """Legacy wrapper for CLI compatibility. Use QuantNado.call_peaks() instead."""
    from ..dataset.store_coverage import CoverageStore
    from ..analysis.normalise import get_library_sizes
    from ._utils import _call_peaks_for_store, load_blacklist

    zarr_path = Path(zarr_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    store = CoverageStore.open(zarr_path, read_only=True)
    library_sizes = get_library_sizes(store)
    blacklist_pr = load_blacklist(blacklist_file)

    def algo(stream, sname):
        return call_quantile_peaks(
            coverage_stream=stream,
            sample_name=sname,
            library_size=library_sizes[sname],
            blacklist=blacklist_pr,
            tilesize=tilesize,
            window_overlap=window_overlap,
            quantile=quantile,
            merge=merge,
        )

    peaks_pr = _call_peaks_for_store(
        store, algo, output_dir=output_dir
    )

    # Extract individual sample BEDs
    results = []
    if len(peaks_pr) > 0:
        # PyRanges in pyranges1 is already a DataFrame subclass, no conversion needed
        for sample_id, grp in peaks_pr.groupby("Sample"):
            output_bed = output_dir / f"{sample_id}.bed"
            pr_obj = pr.PyRanges(grp.drop(columns="Sample"))
            pr_obj.to_bed(output_bed)
            logger.success(f"Peak BED saved to: {output_bed}")
            results.append(str(output_bed))

    return results


# Legacy alias for backward compatibility
def call_quantile_peaks_from_zarr(
    zarr_path: Path,
    output_dir: Path,
    blacklist_file: Optional[Path] = None,
    tilesize: int = 128,
    window_overlap: int = 8,
    quantile: float = 0.98,
    merge: bool = True,
) -> list[str]:
    """Deprecated: Use QuantNado.call_peaks() instead."""
    return _call_quantile_peaks_from_zarr(
        zarr_path, output_dir, blacklist_file, tilesize, window_overlap, quantile, merge
    )



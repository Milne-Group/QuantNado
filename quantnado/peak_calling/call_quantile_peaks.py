from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyranges1 as pr
from crested import import_bigwigs
from loguru import logger


def call_quantile_peaks(
    signal: pd.Series,
    chroms: pd.Series,
    starts: pd.Series,
    ends: pd.Series,
    tilesize: int = 128,
    quantile: float = 0.98,
    blacklist_file: Optional[Path] = None,
    merge: bool = True,
) -> Optional[pr.PyRanges]:
    """Call quantile-based peaks from a single bigWig signal."""
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

    pr_obj = pr.PyRanges(peaks_df)
    if merge:
        pr_obj = pr_obj.merge_overlaps()

    if blacklist_file and blacklist_file.exists():
        logger.debug(f"[{signal.name}] Subtracting blacklist regions: {blacklist_file}")
        blacklist = pr.read_bed(str(blacklist_file))
        pr_obj = pr_obj.subtract_overlaps(blacklist, strand_behavior="ignore")

    logger.info(f"[{signal.name}] Final peak count: {len(pr_obj)}")
    return pr_obj if len(pr_obj) > 0 else None


def call_peaks_from_bigwig_dir(
    bigwig_dir: Path,
    output_dir: Path,
    chromsizes_file: Path,
    blacklist_file: Optional[Path] = None,
    tilesize: int = 128,
    quantile: float = 0.98,
    tmp_dir: Path = Path("tmp"),
) -> list[str]:
    """Call quantile-based peaks from all bigWig files in a directory."""
    bigwig_dir = Path(bigwig_dir)
    output_dir = Path(output_dir)
    chromsizes_file = Path(chromsizes_file)
    blacklist_file = Path(blacklist_file) if blacklist_file else None
    tmp_dir = Path(tmp_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    chromsizes = (
        pd.read_csv(chromsizes_file, sep="\t", names=["Chromosome", "End"])
        .query("~Chromosome.str.contains('_')", engine="python")
        .assign(Start=0)[["Chromosome", "Start", "End"]]
    )

    logger.info(f"Tiling genome with tilesize {tilesize} bp...")
    full_ranges = pr.PyRanges(chromsizes)
    tiled = full_ranges.tile_ranges(tilesize, use_strand=False)
    tiled = tiled.intersect_overlaps(full_ranges, strand_behavior="ignore")
    if blacklist_file and Path(blacklist_file).exists():
        logger.info(f"Applying blacklist from: {blacklist_file}")
        blacklist = pr.read_bed(str(blacklist_file))
        tiled = tiled.subtract_overlaps(blacklist, strand_behavior="ignore")
    tiled_df = pd.DataFrame(tiled)
    tmp_regions_bed = tmp_dir / "tiled_regions.bed"
    tiled.to_bed(tmp_regions_bed)
    logger.info(f"Tiled regions saved to: {tmp_regions_bed}")

    bigwig_paths = sorted(bigwig_dir.glob("*.bw")) + sorted(bigwig_dir.glob("*.bigWig"))
    if not bigwig_paths:
        logger.error(f"No .bw or .bigWig files found in {bigwig_dir}")
        return []
    logger.info(f"Found {len(bigwig_paths)} bigWig file(s)")
    logger.info("Importing all bigWig signals...")

    
    adata = import_bigwigs(
        regions_file=str(tmp_regions_bed),
        bigwigs_folder=str(bigwig_dir),
        chromsizes_file=str(chromsizes_file),
        target="mean",
    )

    adata.X = np.log1p(adata.X)
    df = adata.T.to_df().reset_index()
    df.to_parquet(output_dir / f"logged_rpkm_{tilesize}bp.parquet", index=False)
    sample_names = df.columns[1:]

    results = []
    for i, sample_name in enumerate(sample_names):
        logger.info(f"Processing sample: {sample_name}")
        signal = df[sample_name]

        pr_obj = call_quantile_peaks(
            signal=signal,
            chroms=tiled_df["Chromosome"],
            starts=tiled_df["Start"],
            ends=tiled_df["End"],
            tilesize=tilesize,
            quantile=quantile,
            blacklist_file=blacklist_file,
        )

        if pr_obj is not None:
            output_bed = output_dir / f"{sample_name}.bed"
            pr_obj.to_bed(output_bed)
            logger.success(f"Peak BED saved to: [{output_bed}]")
            results.append(str(output_bed))
        else:
            logger.warning(f"[{sample_name}] No peaks detected.")

    return results



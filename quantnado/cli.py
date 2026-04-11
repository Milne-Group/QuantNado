from __future__ import annotations

import os
import traceback
import warnings

os.environ["KMP_WARNINGS"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"

warnings.filterwarnings("ignore", category=FutureWarning)

from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from loguru import logger

from quantnado._version import __version__
from quantnado.utils import setup_logging

app = typer.Typer(
    help="QuantNado: High-performance genomic quantification and processing.",
    add_completion=False,
    no_args_is_help=True,
)


@app.callback(invoke_without_command=True)
def _root(
    ctx: typer.Context,
    version: bool = typer.Option(None, "--version", help="Show version and exit"),
):
    if version:
        typer.echo(f"quantnado {__version__}")
        raise typer.Exit()


def _setup_logging(log_file: Path, verbose: bool) -> None:
    if log_file.parent != Path(".") and not log_file.parent.exists():
        log_file.parent.mkdir(parents=True, exist_ok=True)
    if log_file.exists():
        try:
            log_file.unlink()
        except Exception:
            pass
    setup_logging(log_file, verbose)


def _parse_stranded_value(value) -> str | None:
    """Normalize RNA strandedness metadata to ``R`` / ``F`` / ``None``."""
    if value is None or pd.isna(value):
        return None

    text = str(value).strip().upper()
    if text in {"", "NONE", "FALSE", "0", "U", "UNSTRANDED"}:
        return None
    if text in {"R", "REVERSE", "2"}:
        return "R"
    if text in {"F", "FORWARD", "1"}:
        return "F"
    raise ValueError(
        f"Invalid stranded value '{value}'. Expected one of R/F/1/2/U."
    )


# ======================================================================
# quantnado create-dataset
# ======================================================================


@app.command("create-dataset")
def create_dataset(
    metadata: Path = typer.Option(
        ...,
        "--metadata",
        "-m",
        help=(
            "CSV/TSV file describing samples. Required columns: sample, assay, bam/methyl/vcf. "
            "Optional: ip (for ChIP/CUT&TAG), chromsizes (per sample). "
            "Assay values: ATAC, ChIP, RNA, CUT&TAG, METH, SNP, MCC."
        ),
    ),
    output_dir: Path = typer.Option(
        ..., "--output-dir", "-o", help="Directory to write per-sample .zarr stores."
    ),
    chromsizes: Optional[Path] = typer.Option(
        None,
        "--chromsizes",
        help="Path to .chrom.sizes file. If omitted, inferred from BAM headers or VCF ##contig lines.",
    ),
    filter_chromosomes: bool = typer.Option(
        True,
        "--filter-chromosomes/--no-filter-chromosomes",
        help="Keep only canonical chromosomes (chr* without underscore).",
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite/--no-overwrite", help="Overwrite existing stores."
    ),
    chunk_len: Optional[int] = typer.Option(
        None, "--chunk-len", help="Override position-axis chunk length."
    ),
    construction_compression: str = typer.Option(
        "default",
        "--construction-compression",
        help="Build-time compression: default, fast, or none.",
        case_sensitive=False,
    ),
    test: bool = typer.Option(
        False, "--test", help="Restrict to test chromosomes (default: chr9/chr13/chr21)."
    ),
    test_chrom: list[str] = typer.Option(
        None,
        "--test-chrom",
        help="Chromosome to keep in test mode. Repeat to pass multiple chromosomes.",
    ),
    log_file: Path = typer.Option(
        Path("quantnado_create.log"), "--log-file", help="Path to log file."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
):
    """Create per-sample zarr stores from a metadata CSV/TSV.

    The metadata file maps each sample to its input file(s) and assay type.
    One .zarr store is written per row to output-dir.

    \b
    Example metadata CSV:
      sample,assay,bam,ip
      ATAC-1,ATAC,ATAC-1.bam,
      ChIP-H3K27ac,ChIP,H3K27ac.bam,H3K27ac
      RNA-1,RNA,RNA-1.bam,
      METH-1,METH,METH-1.bam,METH-1.bedGraph
      SNP-1,SNP,,SNP-1.vcf.gz
    """
    _setup_logging(log_file, verbose)
    import pandas as pd
    from quantnado.dataset.metadata import array_key

    sep = "\t" if metadata.suffix in (".tsv", ".txt") else ","
    try:
        df = pd.read_csv(metadata, sep=sep)
    except Exception as e:
        logger.error(f"Could not read metadata file {metadata}: {e}")
        raise typer.Exit(code=1)

    required_cols = {"sample", "assay"}
    missing = required_cols - set(df.columns)
    if missing:
        logger.error(f"Metadata is missing required columns: {missing}")
        raise typer.Exit(code=1)

    output_dir.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []

    for _, row in df.iterrows():
        sample = str(row["sample"])
        assay = str(row.get("assay", "")).upper()
        ip = str(row.get("ip", "") or "")
        bam_path = str(row.get("bam", "") or "").strip()
        methyl_path = str(row.get("methyl", "") or "").strip()
        vcf_path = str(row.get("vcf", "") or "").strip()
        row_chromsizes = row.get("chromsizes", None)
        effective_chromsizes = (
            str(row_chromsizes).strip()
            if row_chromsizes and str(row_chromsizes).strip()
            else chromsizes
        )

        store_path = output_dir / f"{sample}.zarr"
        if store_path.exists() and not overwrite:
            logger.info(f"Skipping existing store: {store_path.name}")
            continue

        try:
            if assay == "METH":
                from quantnado.dataset.store_methyl import MethylStore
                if not bam_path:
                    raise ValueError("METH rows require a 'bam' column")
                if not methyl_path:
                    raise ValueError("METH rows require a 'methyl' column")
                MethylStore.from_files(
                    bam_path=bam_path,
                    methyl_path=methyl_path,
                    store_path=store_path,
                    sample=sample,
                    chromsizes=effective_chromsizes,
                    chunk_len=chunk_len,
                    construction_compression=construction_compression,
                    overwrite=overwrite,
                    filter_chromosomes=filter_chromosomes,
                    test=test or bool(test_chrom),
                    test_chromosomes=test_chrom or None,
                )
            elif assay == "SNP":
                from quantnado.dataset.store_variants import VariantStore
                if not vcf_path:
                    raise ValueError("SNP rows require a 'vcf' column")
                VariantStore.from_vcf(
                    vcf_path=vcf_path,
                    store_path=store_path,
                    sample=sample,
                    chromsizes=effective_chromsizes,
                    chunk_len=chunk_len,
                    construction_compression=construction_compression,
                    overwrite=overwrite,
                    filter_chromosomes=filter_chromosomes,
                    test=test or bool(test_chrom),
                    test_chromosomes=test_chrom or None,
                )
            else:
                # BAM-based: ATAC, ChIP, CUT&TAG, RNA, MCC
                from quantnado.dataset.store_bam import BamStore
                if not bam_path:
                    raise ValueError(f"{assay} rows require a 'bam' column")
                stranded = (
                    _parse_stranded_value(row.get("stranded", None))
                    if assay.upper() == "RNA"
                    else None
                )
                is_mcc = assay.upper() == "MCC"
                BamStore.from_bam_files(
                    bam_path=bam_path,
                    store_path=store_path,
                    assay=assay,
                    sample=sample,
                    ip=ip or None,
                    chromsizes=effective_chromsizes,
                    stranded=stranded,
                    is_mcc=is_mcc,
                    chunk_len=chunk_len,
                    construction_compression=construction_compression,
                    overwrite=overwrite,
                    filter_chromosomes=filter_chromosomes,
                    test=test or bool(test_chrom),
                    test_chromosomes=test_chrom or None,
                )
            logger.success(f"Wrote {store_path.name}")
        except Exception as e:
            logger.error(f"Failed to process sample '{sample}': {e}")
            logger.debug(traceback.format_exc())
            errors.append(sample)

    if errors:
        logger.warning(f"Failed samples: {errors}")
        raise typer.Exit(code=1)
    logger.success(f"All samples written to {output_dir}")


# ======================================================================
# quantnado call-peaks
# ======================================================================


@app.command("call-peaks")
def call_peaks(
    zarr: Path = typer.Option(
        ..., "--zarr", help="Path to QuantNado zarr store or directory of per-sample stores."
    ),
    method: str = typer.Option(
        "quantile", "--method", help="Peak calling method: quantile, seacr, or lanceotron."
    ),
    output_dir: Path = typer.Option(..., "--output-dir", help="Directory for output BED files."),
    assay: Optional[str] = typer.Option(
        None, "--assay", help="Assay key to call peaks on (e.g. 'atac', 'chip_h3k27ac'). Defaults to first assay."
    ),
    blacklist: Optional[Path] = typer.Option(None, "--blacklist", help="BED file of regions to exclude."),
    # quantile options
    tilesize: int = typer.Option(128, "--tilesize", help="[quantile] Tile size in bp."),
    window_overlap: int = typer.Option(8, "--window-overlap", help="[quantile] Overlap between adjacent windows in bp."),
    quantile: float = typer.Option(0.98, "--quantile", help="[quantile] Quantile threshold."),
    merge: bool = typer.Option(True, "--merge/--no-merge", help="[quantile] Merge adjacent peaks."),
    # seacr options
    control_zarr: Optional[Path] = typer.Option(
        None, "--control-zarr", help="[seacr] Control (IgG) zarr store."
    ),
    fdr_threshold: float = typer.Option(0.01, "--fdr", help="[seacr] FDR threshold (0–1)."),
    norm: str = typer.Option("non", "--norm", help='[seacr] "norm" or "non" to normalise control.'),
    stringency: str = typer.Option(
        "stringent", "--stringency", help='[seacr] "stringent" or "relaxed".'
    ),
    # lanceotron options
    score_threshold: float = typer.Option(
        0.5, "--score-threshold", help="[lanceotron] Minimum classification score (0–1)."
    ),
    smooth_window: int = typer.Option(400, "--smooth-window", help="[lanceotron] Rolling mean window (bp)."),
    batch_size: int = typer.Option(512, "--batch-size", help="[lanceotron] Inference batch size."),
    # shared
    n_workers: int = typer.Option(1, "--n-workers", help="Parallel workers (seacr/lanceotron)."),
    device: Optional[str] = typer.Option(
        None, "--device", help="Compute device: cpu, cuda, mps, or None for auto (seacr/lanceotron)."
    ),
    log_file: Path = typer.Option(Path("quantnado_peaks.log"), "--log-file", help="Path to log file."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
):
    """Call peaks from a QuantNado zarr store.

    \b
    Methods:
      quantile    Quantile-threshold peak calling (default; fast, no dependencies)
      seacr       SEACR-style AUC island calling (CUT&RUN / ATAC; pure Python)
      lanceotron  ML classifier (ChIP-seq; requires: pip install quantnado[lanceotron])
    """
    _setup_logging(log_file, verbose)
    try:
        if method == "quantile":
            from quantnado.peak_calling.call_quantile_peaks import call_quantile_peaks_from_zarr
            call_quantile_peaks_from_zarr(
                zarr_path=zarr,
                output_dir=output_dir,
                assay=assay,
                blacklist_file=blacklist,
                tilesize=tilesize,
                window_overlap=window_overlap,
                quantile=quantile,
                merge=merge,
            )
        elif method == "seacr":
            from quantnado.peak_calling.call_seacr_peaks import call_seacr_peaks_from_zarr
            call_seacr_peaks_from_zarr(
                zarr_path=zarr,
                output_dir=output_dir,
                assay=assay,
                control_zarr_path=control_zarr,
                fdr_threshold=fdr_threshold,
                norm=norm,
                stringency=stringency,
                blacklist_file=blacklist,
                n_workers=n_workers,
                device=device,
            )
        elif method == "lanceotron":
            from quantnado.peak_calling.call_lanceotron_peaks import call_lanceotron_peaks_from_zarr
            call_lanceotron_peaks_from_zarr(
                zarr_path=zarr,
                output_dir=output_dir,
                assay=assay,
                score_threshold=score_threshold,
                blacklist_file=blacklist,
                smooth_window=smooth_window,
                batch_size=batch_size,
                n_workers=n_workers,
                device=device,
            )
        else:
            logger.error(f"Unknown method '{method}'. Choose: quantile, seacr, or lanceotron.")
            raise typer.Exit(code=1)
        logger.success(f"Peak calling complete: {output_dir}")
    except typer.Exit:
        raise
    except Exception as e:
        logger.error(f"Peak calling failed: {type(e).__name__}: {e}")
        logger.debug(traceback.format_exc())
        raise typer.Exit(code=1)


# ======================================================================
# Entry point
# ======================================================================


def main():
    app()


if __name__ == "__main__":
    main()

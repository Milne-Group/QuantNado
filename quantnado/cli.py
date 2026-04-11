from __future__ import annotations

import os
import traceback
import warnings

os.environ["KMP_WARNINGS"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"

warnings.filterwarnings("ignore", category=FutureWarning)

from pathlib import Path
import shutil
import tempfile
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

dataset_app = typer.Typer(
    help="Create and combine QuantNado datasets.",
    add_completion=False,
    no_args_is_help=True,
)
app.add_typer(dataset_app, name="dataset")


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


def _create_store_impl(
    *,
    sample: str,
    assay: str,
    output_dir: Path,
    bamfile: Optional[Path],
    vcf_file: Optional[Path],
    methylation_file: Optional[Path],
    ip: Optional[str],
    chromsizes: Optional[Path],
    stranded: Optional[str],
    filter_chromosomes: bool,
    overwrite: bool,
    chunk_len: Optional[int],
    construction_compression: str,
    test: bool,
    test_chrom: list[str] | None,
    log_file: Path,
    verbose: bool,
) -> None:
    _setup_logging(log_file, verbose)

    assay = assay.upper()
    output_dir.mkdir(parents=True, exist_ok=True)
    store_path = output_dir / f"{sample}.zarr"

    if store_path.exists() and not overwrite:
        logger.info(f"Skipping existing store: {store_path.name}")
        return

    try:
        if assay == "METH":
            from quantnado.dataset.store_methyl import MethylStore

            if bamfile is None:
                raise ValueError("METH requires --bamfile")
            if methylation_file is None:
                raise ValueError("METH requires --methylation-file")

            MethylStore.from_files(
                bam_path=str(bamfile),
                methyl_path=str(methylation_file),
                store_path=store_path,
                sample=sample,
                chromsizes=chromsizes,
                chunk_len=chunk_len,
                construction_compression=construction_compression,
                overwrite=overwrite,
                filter_chromosomes=filter_chromosomes,
                test=test or bool(test_chrom),
                test_chromosomes=test_chrom or None,
            )
        elif assay == "SNP":
            from quantnado.dataset.store_variants import VariantStore

            if vcf_file is None:
                raise ValueError("SNP requires --vcf-file")

            VariantStore.from_vcf(
                vcf_path=str(vcf_file),
                store_path=store_path,
                sample=sample,
                chromsizes=chromsizes,
                chunk_len=chunk_len,
                construction_compression=construction_compression,
                overwrite=overwrite,
                filter_chromosomes=filter_chromosomes,
                test=test or bool(test_chrom),
                test_chromosomes=test_chrom or None,
            )
        else:
            from quantnado.dataset.store_bam import BamStore

            if bamfile is None:
                raise ValueError(f"{assay} requires --bamfile")

            normalised_stranded = _parse_stranded_value(stranded) if assay == "RNA" else None
            BamStore.from_bam_files(
                bam_path=str(bamfile),
                store_path=store_path,
                assay=assay,
                sample=sample,
                ip=ip or None,
                chromsizes=chromsizes,
                stranded=normalised_stranded,
                chunk_len=chunk_len,
                construction_compression=construction_compression,
                overwrite=overwrite,
                filter_chromosomes=filter_chromosomes,
                test=test or bool(test_chrom),
                test_chromosomes=test_chrom or None,
            )
        logger.success(f"Wrote {store_path.name}")
    except typer.Exit:
        raise
    except Exception as e:
        logger.error(f"Failed to process sample '{sample}': {e}")
        logger.debug(traceback.format_exc())
        raise typer.Exit(code=1)


def _create_dataset_from_metadata_impl(
    metadata: Path,
    output_dir: Path,
    chromsizes: Optional[Path],
    filter_chromosomes: bool,
    overwrite: bool,
    chunk_len: Optional[int],
    construction_compression: str,
    test: bool,
    test_chrom: list[str] | None,
    log_file: Path,
    verbose: bool,
) -> None:
    """Create per-sample zarr stores from a metadata CSV/TSV."""
    _setup_logging(log_file, verbose)

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

    errors: list[str] = []

    for _, row in df.iterrows():
        sample = str(row["sample"])
        assay = str(row.get("assay", "")).upper()
        row_chromsizes = row.get("chromsizes", None)
        effective_chromsizes = (
            Path(str(row_chromsizes).strip())
            if row_chromsizes and str(row_chromsizes).strip()
            else chromsizes
        )

        try:
            _create_store_impl(
                sample=sample,
                assay=assay,
                output_dir=output_dir,
                bamfile=Path(str(row.get("bam", "")).strip()) if str(row.get("bam", "")).strip() else None,
                vcf_file=Path(str(row.get("vcf", "")).strip()) if str(row.get("vcf", "")).strip() else None,
                methylation_file=Path(str(row.get("methyl", "")).strip()) if str(row.get("methyl", "")).strip() else None,
                ip=str(row.get("ip", "") or "") or None,
                chromsizes=effective_chromsizes,
                stranded=str(row.get("stranded", "")).strip() or None,
                filter_chromosomes=filter_chromosomes,
                overwrite=overwrite,
                chunk_len=chunk_len,
                construction_compression=construction_compression,
                test=test,
                test_chrom=test_chrom,
                log_file=log_file,
                verbose=verbose,
            )
        except typer.Exit:
            errors.append(sample)

    if errors:
        logger.warning(f"Failed samples: {errors}")
        raise typer.Exit(code=1)
    logger.success(f"All samples written to {output_dir}")


# ======================================================================
# quantnado dataset create
# ======================================================================


@dataset_app.command("create")
def create_dataset(
    sample: str = typer.Option(..., "--sample", help="Sample name for the output store."),
    assay: str = typer.Option(
        ...,
        "--assay",
        help="Assay type: ATAC, ChIP, RNA, CUT&TAG, METH, SNP, or MCC.",
    ),
    output_dir: Path = typer.Option(
        ..., "--output-dir", "-o", help="Directory to write the sample .zarr store into."
    ),
    bamfile: Optional[Path] = typer.Option(
        None,
        "--bamfile",
        "--bam-file",
        help="BAM file for BAM-based assays and METH.",
    ),
    vcf_file: Optional[Path] = typer.Option(
        None,
        "--vcf_file",
        "--vcf-file",
        help="VCF file for SNP assays.",
    ),
    methylation_file: Optional[Path] = typer.Option(
        None,
        "--methylation_file",
        "--methylation-file",
        help="Methylation bedGraph/TSV for METH assays.",
    ),
    ip: Optional[str] = typer.Option(
        None,
        "--ip",
        help="IP / target label for ChIP or CUT&TAG samples.",
    ),
    stranded: Optional[str] = typer.Option(
        None,
        "--stranded",
        help="RNA strandedness: R/F/1/2/U.",
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
    """Create one per-sample zarr store from direct input files."""
    _create_store_impl(
        sample=sample,
        assay=assay,
        output_dir=output_dir,
        bamfile=bamfile,
        vcf_file=vcf_file,
        methylation_file=methylation_file,
        ip=ip,
        chromsizes=chromsizes,
        stranded=stranded,
        filter_chromosomes=filter_chromosomes,
        overwrite=overwrite,
        chunk_len=chunk_len,
        construction_compression=construction_compression,
        test=test,
        test_chrom=test_chrom,
        log_file=log_file,
        verbose=verbose,
    )


@app.command("create-dataset", hidden=True)
def create_dataset_legacy(
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
    """Backward-compatible alias for `quantnado dataset create`."""
    _create_dataset_from_metadata_impl(
        metadata=metadata,
        output_dir=output_dir,
        chromsizes=chromsizes,
        filter_chromosomes=filter_chromosomes,
        overwrite=overwrite,
        chunk_len=chunk_len,
        construction_compression=construction_compression,
        test=test,
        test_chrom=test_chrom,
        log_file=log_file,
        verbose=verbose,
    )


@dataset_app.command(
    "combine",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": False},
)
def combine_dataset(
    ctx: typer.Context,
    stores: Path = typer.Option(
        ...,
        "--stores",
        help="One or more per-sample QuantNado .zarr stores after a single --stores flag.",
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Path to the combined multi-sample .zarr store.",
    ),
    overwrite: bool = typer.Option(
        True,
        "--overwrite/--no-overwrite",
        help="Overwrite the output store if it already exists.",
    ),
    log_file: Path = typer.Option(
        Path("quantnado_combine.log"), "--log-file", help="Path to log file."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
):
    """Combine per-sample zarr stores into a single multi-sample store."""
    _setup_logging(log_file, verbose)
    from quantnado.analysis.core import QuantNadoDataset

    def _combine_from_source(source: Path) -> None:
        QuantNadoDataset.combine(src=source, output=output, overwrite=overwrite)

    try:
        store_paths = [stores] + [Path(arg) for arg in ctx.args]

        if len(store_paths) == 1 and store_paths[0].is_dir() and store_paths[0].suffix != ".zarr":
            _combine_from_source(store_paths[0])
        else:
            with tempfile.TemporaryDirectory(prefix="quantnado-combine-") as tmp:
                tmpdir = Path(tmp)
                seen: set[str] = set()
                for idx, store in enumerate(store_paths):
                    if not store.exists():
                        raise FileNotFoundError(f"Store does not exist: {store}")
                    name = store.name
                    if name in seen:
                        stem = store.stem or f"store_{idx+1}"
                        name = f"{stem}_{idx+1}.zarr"
                    seen.add(name)
                    target = tmpdir / name
                    try:
                        os.symlink(store.resolve(), target, target_is_directory=True)
                    except Exception:
                        shutil.copytree(store, target)
                _combine_from_source(tmpdir)
        logger.success(f"Combined dataset written to {output}")
    except typer.Exit:
        raise
    except Exception as e:
        logger.error(f"Dataset combine failed: {type(e).__name__}: {e}")
        logger.debug(traceback.format_exc())
        raise typer.Exit(code=1)


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

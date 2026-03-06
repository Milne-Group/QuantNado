from __future__ import annotations

import os
import warnings

os.environ["KMP_WARNINGS"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning, module="sorted_nearest")

import typer
from loguru import logger
import traceback
from pathlib import Path

from quantnado.dataset.bam import BamStore
from quantnado.utils import setup_logging
from quantnado._version import __version__

app = typer.Typer(
    help="QuantNado: High-performance genomic quantification and processing.",
    add_completion=False,
    no_args_is_help=True,
)


@app.callback(invoke_without_command=True)
def _version_callback(
    ctx: typer.Context,
    version: bool = typer.Option(None, "--version", help="Show version and exit"),
):
    """QuantNado: High-performance genomic quantification and processing."""
    if version:
        typer.echo(f"quantnado {__version__}")
        raise typer.Exit()


def _setup_cli_logging(log_file: Path, verbose: bool):
    """Ensure log directory exists and initialize loguru via utils."""
    if log_file.parent != Path(".") and not log_file.parent.exists():
        log_file.parent.mkdir(parents=True, exist_ok=True)

    if log_file.exists():
        try:
            log_file.unlink()
        except Exception:
            pass

    setup_logging(log_file, verbose)


@app.command()
def call_peaks(
    bigwig_dir: Path = typer.Option(..., "--bigwig-dir", help="Directory containing bigWig files"),
    output_dir: Path = typer.Option(..., "--output-dir", help="Directory to save output peak files (BED format)"),
    chromsizes: str = typer.Option(..., "--chromsizes", help="Path to a two-column chromsizes file"),
    blacklist: Path | None = typer.Option(None, "--blacklist", help="Path to a BED file with regions to exclude"),
    tilesize: int = typer.Option(128, "--tilesize", help="Size of genomic tiles to create (default: 128 bp)"),
    quantile: float = typer.Option(0.98, "--quantile", help="Quantile threshold for peak calling"),
    merge: bool = typer.Option(False, "--merge/--no-merge", help="Merge overlapping peaks after quantile calling"),
    tmp_dir: Path = typer.Option("tmp", "--tmp-dir", help="Temporary directory for intermediate files"),
    log_file: Path = typer.Option("quantnado_peaks.log", "--log-file", help="Path to the log file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
):
    """
    Call quantile-based peaks from bigWig files.
    """
    _setup_cli_logging(log_file, verbose)
    try:
        # Lazy import to avoid zarr/anndata compatibility issues
        from quantnado.peak_calling.call_quantile_peaks import call_peaks_from_bigwig_dir

        call_peaks_from_bigwig_dir(
            bigwig_dir=bigwig_dir,
            output_dir=output_dir,
            chromsizes_file=chromsizes,
            blacklist_file=str(blacklist) if blacklist else None,
            tilesize=tilesize,
            quantile=quantile,
            merge=merge,
            tmp_dir=tmp_dir,
        )
        logger.success(f"Finished calling peaks: {output_dir}")
    except Exception as e:
        logger.error(f"Peak calling failed: {type(e).__name__}: {e}")
        logger.debug(traceback.format_exc())
        raise typer.Exit(code=1)


@app.command()
def create_dataset(
    output: Path = typer.Option(..., "--output", "-o", help="Output directory for the multiomics store."),
    bam_files: list[Path] = typer.Option([], "--bam", help="BAM file for coverage (repeat for multiple)."),
    bedgraph_files: list[Path] = typer.Option([], "--bedgraph", help="bedGraph file for methylation (repeat for multiple)."),
    vcf_files: list[Path] = typer.Option([], "--vcf", help="VCF.gz file for variants (repeat for multiple)."),
    chromsizes: Path | None = typer.Option(
        None, "--chromsizes", help="Path to chromsizes. If omitted, extracted from first BAM."
    ),
    metadata: Path | None = typer.Option(None, "--metadata", help="Path to metadata CSV file."),
    bam_sample_names: list[str] = typer.Option([], "--bam-sample-name", help="Override sample name for a BAM file (repeat for multiple)."),
    bedgraph_sample_names: list[str] = typer.Option([], "--bedgraph-sample-name", help="Override sample name for a bedGraph file (repeat for multiple)."),
    vcf_sample_names: list[str] = typer.Option([], "--vcf-sample-name", help="Override sample name for a VCF file (repeat for multiple)."),
    sample_column: str = typer.Option("sample_id", "--sample-column", help="Column in metadata matching sample names."),
    chunk_len: int = typer.Option(65536, "--chunk-len", help="Zarr chunk size for coverage store."),
    filter_chromosomes: bool = typer.Option(True, "--filter-chromosomes/--no-filter-chromosomes", help="Keep only canonical chromosomes."),
    overwrite: bool = typer.Option(False, "--overwrite/--no-overwrite", help="Overwrite existing sub-stores."),
    resume: bool = typer.Option(False, "--resume", help="Resume processing an existing store, skipping completed samples."),
    max_workers: int = typer.Option(1, "--max-workers", help="Number of parallel threads for BAM processing."),
    log_file: Path = typer.Option("quantnado_processing.log", "--log-file", help="Path to the log file."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
):
    """
    Create a multiomics store from BAM, bedGraph, and/or VCF files.

    At least one of --bam, --bedgraph, or --vcf must be provided.
    """
    _setup_cli_logging(log_file, verbose)

    if not any([bam_files, bedgraph_files, vcf_files]):
        logger.error("Provide at least one of --bam, --bedgraph, or --vcf.")
        raise typer.Exit(code=1)

    modality_counts = [
        f"{len(bam_files)} BAM" if bam_files else None,
        f"{len(bedgraph_files)} bedGraph" if bedgraph_files else None,
        f"{len(vcf_files)} VCF" if vcf_files else None,
    ]
    logger.info(f"Building multiomics store at {output}: {', '.join(m for m in modality_counts if m)}")

    try:
        from quantnado.dataset.multiomics import MultiomicsStore

        MultiomicsStore.from_files(
            store_dir=output,
            bam_files=[str(b) for b in bam_files] or None,
            bedgraph_files=[str(b) for b in bedgraph_files] or None,
            vcf_files=[str(b) for b in vcf_files] or None,
            chromsizes=chromsizes,
            metadata=metadata,
            bam_sample_names=bam_sample_names or None,
            bedgraph_sample_names=bedgraph_sample_names or None,
            vcf_sample_names=vcf_sample_names or None,
            sample_column=sample_column,
            chunk_len=chunk_len,
            filter_chromosomes=filter_chromosomes,
            overwrite=overwrite,
            resume=resume,
            max_workers=max_workers,
        )
        logger.success(f"Multiomics store created: {output}")
    except Exception as e:
        logger.error(f"Dataset creation failed: {e}")
        logger.debug(traceback.format_exc())
        raise typer.Exit(code=1)


def make_zarr_main():
    """Alias entry point for quantnado-make-zarr."""
    app()


def combine_metadata_main():
    """Entry point for quantnado-combine-metadata: merges multiple metadata CSV files."""
    combine_app = typer.Typer(
        help="Combine multiple metadata CSV files into one.",
        add_completion=False,
        no_args_is_help=True,
    )

    @combine_app.command()
    def combine(
        metadata_files: list[Path] = typer.Argument(..., help="Paths to metadata CSV files to combine."),
        output: Path = typer.Option(..., "--output", "-o", help="Path to save the combined metadata CSV."),
    ):
        """Merge multiple metadata CSV files into a single output CSV."""
        try:
            combined = BamStore._combine_metadata_files([str(p) for p in metadata_files])
            combined.to_csv(output, index=False)
            typer.echo(f"Combined metadata written to {output}")
        except Exception as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)

    combine_app()


def main():
    app()


if __name__ == "__main__":
    main()

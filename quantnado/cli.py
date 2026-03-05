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
    bam_files: list[Path] = typer.Argument(..., help="Paths to BAM files to process."),
    output: Path = typer.Option(..., "--output", "-o", help="Path to save the unified Zarr dataset."),
    chromsizes: Path | None = typer.Option(
        None, "--chromsizes", help="Path to chromsizes. If omitted, extracted from first BAM."
    ),
    metadata: Path | None = typer.Option(None, "--metadata", help="Path to metadata CSV file."),
    max_workers: int = typer.Option(1, "--max-workers", help="Number of parallel threads for processing BAM files."),
    log_file: Path = typer.Option("quantnado_processing.log", "--log-file", help="Path to the log file."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing dataset if it exists"),
    resume: bool = typer.Option(False, "--resume", help="Resume processing an existing dataset, skipping completed samples."),
):
    """
    Process BAM files and create a unified Zarr dataset.
    """
    _setup_cli_logging(log_file, verbose)

    logger.info(f"Processing {len(bam_files)} BAM files into {output}")
    try:
        BamStore.from_bam_files(
            bam_files=[str(b) for b in bam_files],
            chromsizes=chromsizes,
            store_path=output,
            metadata=metadata,
            max_workers=max_workers,
            log_file=log_file,
            overwrite=overwrite,
            resume=resume,
        )
        logger.success(f"Zarr dataset created: {output}")
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

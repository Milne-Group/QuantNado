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

from quantnado.dataset.store_coverage import BamStore, CoverageType
from quantnado.utils import classify_methylation_files, setup_logging
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
    zarr: Path = typer.Option(..., "--zarr", help="Path to a QuantNado zarr coverage store"),
    method: str = typer.Option("quantile", "--method", help="Peak calling method: quantile, seacr, or lanceotron"),
    output_dir: Path = typer.Option(..., "--output-dir", help="Directory to save output peak files (BED format)"),
    sample: str | None = typer.Option(None, "--sample", help="Call peaks for a single sample (default: all samples)"),
    blacklist: Path | None = typer.Option(None, "--blacklist", help="Path to a BED file with regions to exclude"),
    # quantile options
    tilesize: int = typer.Option(128, "--tilesize", help="[quantile] Size of genomic tiles in bp"),
    window_overlap: int = typer.Option(8, "--window-overlap", help="[quantile] Overlap between adjacent windows in bp"),
    quantile: float = typer.Option(0.98, "--quantile", help="[quantile] Quantile threshold for peak calling"),
    merge: bool = typer.Option(True, "--merge/--no-merge", help="[quantile] Merge overlapping and adjacent peaks after calling"),
    merge_distance: int = typer.Option(150, "--merge-distance", help="[quantile] Maximum distance (bp) between peaks to merge"),
    # seacr options
    control_zarr: Path | None = typer.Option(None, "--control-zarr", help="[seacr] Path to a control (IgG) QuantNado zarr store"),
    fdr_threshold: float = typer.Option(0.01, "--fdr", help="[seacr] Numeric FDR threshold (0–1) used when no control zarr is provided"),
    norm: str = typer.Option("non", "--norm", help='[seacr] "norm" to normalise control to experimental signal, "non" to skip'),
    stringency: str = typer.Option("stringent", "--stringency", help='[seacr] "stringent" (peak of AUC curve) or "relaxed" (knee of curve)'),
    # lanceotron options
    score_threshold: float = typer.Option(0.5, "--score-threshold", help="[lanceotron] Minimum overall_classification score (0–1)"),
    smooth_window: int = typer.Option(400, "--smooth-window", help="[lanceotron] Rolling mean window for candidate detection (bp)"),
    lanceotron_batch_size: int = typer.Option(512, "--batch-size", help="[lanceotron] Inference batch size"),
    # shared
    device: str | None = typer.Option(None, "--device", help="Compute device: 'cuda', 'mps', 'cpu', or None for auto-detect (seacr/lanceotron only)"),
    n_workers: int = typer.Option(1, "--n-workers", help="Number of parallel workers for chromosomes (quantile) or samples (seacr/lanceotron)"),
    log_file: Path = typer.Option("quantnado_peaks.log", "--log-file", help="Path to the log file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
):
    """
    Call peaks from a QuantNado zarr coverage store.

    --method quantile   quantile-threshold peak calling (default)
    --method seacr      SEACR-style AUC island calling (pure Python)
    --method lanceotron LanceOtron ML classifier (requires: pip install quantnado[lanceotron])
    """
    _setup_cli_logging(log_file, verbose)
    try:
        from quantnado.api import QuantNado
        from quantnado.peak_calling import QuantileOptions, SeacrOptions, LanceotronOptions

        ds = QuantNado.open_dataset(zarr)
        output_dir.mkdir(parents=True, exist_ok=True)

        if method == "quantile":
            options = QuantileOptions(
                tilesize=tilesize,
                window_overlap=window_overlap,
                quantile=quantile,
                merge=merge,
                merge_distance=merge_distance,
                n_workers=n_workers,
            )
            ds.call_peaks(
                method="quantile",
                sample_name=sample,
                output_dir=output_dir,
                blacklist_file=blacklist,
                options=options,
            )

        elif method == "seacr":
            options = SeacrOptions(
                control_sample_name=None,
                fdr_threshold=fdr_threshold,
                norm=norm,
                stringency=stringency,
            )
            ds.call_peaks(
                method="seacr",
                sample_name=sample,
                output_dir=output_dir,
                blacklist_file=blacklist,
                options=options,
                device=device,
                n_workers=n_workers,
            )

        elif method == "lanceotron":
            options = LanceotronOptions(
                score_threshold=score_threshold,
                smooth_window=smooth_window,
                batch_size=lanceotron_batch_size,
            )
            ds.call_peaks(
                method="lanceotron",
                sample_name=sample,
                output_dir=output_dir,
                blacklist_file=blacklist,
                options=options,
                device=device,
                n_workers=n_workers,
            )

        else:
            logger.error(f"Unknown method '{method}'. Choose 'quantile', 'seacr', or 'lanceotron'.")
            raise typer.Exit(code=1)

        logger.success(f"Finished calling peaks: {output_dir}")
    except typer.Exit:
        raise
    except Exception as e:
        logger.error(f"Peak calling failed: {type(e).__name__}: {e}")
        logger.debug(traceback.format_exc())
        raise typer.Exit(code=1)


@app.command()
def create_dataset(
    output: Path = typer.Option(..., "--output", "-o", help="Output directory for the multiomics store."),
    # Input files
    bam: str | None = typer.Option(None, "--bam", help="Comma-separated BAM files for coverage."),
    methylation: str | None = typer.Option(
        None,
        "--methylation",
        help=(
            "Comma-separated methylation files. File type is detected from the filename: "
            "*num_mc_cxreport* → mC, *num_hmc_cxreport* → hmC, *CXreport* → evoC whole-molecule, "
            "everything else → MethylDackel bedGraph."
        ),
    ),
    vcf: str | None = typer.Option(None, "--vcf", help="Comma-separated VCF.gz files for variants."),
    # Reference / metadata
    chromsizes: Path | None = typer.Option(
        None, "--chromsizes", help="Path to chromsizes. If omitted, extracted from first BAM."
    ),
    metadata: Path | None = typer.Option(None, "--metadata", help="Path to metadata CSV file."),
    sample_column: str = typer.Option("sample_id", "--sample-column", help="Column in metadata matching sample names."),
    # Sample naming
    bam_sample_names: str | None = typer.Option(None, "--bam-sample-names", help="Comma-separated sample name overrides for BAM files."),
    methylation_sample_names: str | None = typer.Option(
        None,
        "--methylation-sample-names",
        help=(
            "Comma-separated sample name overrides for methylation files, in the same order as --methylation "
            "after classification: bedGraph names first, then CXreport names, then mC/hmC names (one per sample pair)."
        ),
    ),
    vcf_sample_names: str | None = typer.Option(None, "--vcf-sample-names", help="Comma-separated sample name overrides for VCF files."),
    # Coverage options
    filter_chromosomes: bool = typer.Option(True, "--filter-chromosomes/--no-filter-chromosomes", help="Keep only canonical chromosomes."),
    coverage_type: str | None = typer.Option(
        None,
        "--coverage-type",
        help=(
            "BAM type for coverage processing. "
            "Single value: 'unstranded' (default), 'stranded', or 'mcc' (Micro-Capture C) — applies to all BAM files. "
            "Comma-separated list: 'stranded,unstranded' — one entry per BAM file in order. "
            "Comma-separated key:value pairs: 'sample1:stranded,sample2:mcc' — per sample name."
        ),
    ),
    count_fragments: bool = typer.Option(
        False,
        "--count-fragments/--no-count-fragments",
        help="Count fragments (insert-level) instead of individual reads.",
    ),
    viewpoint_tag: str = typer.Option(
        "VP",
        "--viewpoint-tag",
        help="SAM tag used to identify MCC viewpoints (default: VP). Only relevant when --coverage-type includes 'mcc'.",
    ),
    # Process control
    overwrite: bool = typer.Option(False, "--overwrite/--no-overwrite", help="Overwrite existing sub-stores."),
    resume: bool = typer.Option(False, "--resume", help="Resume processing an existing store, skipping completed samples."),
    max_workers: int = typer.Option(1, "--max-workers", help="Number of parallel threads for chromosome-level processing within each sample."),
    # Store format
    chunk_len: int | None = typer.Option(
        None,
        "--chunk-len",
        min=1,
        help="Override the position-axis chunk length. If omitted, QuantNado chooses a filesystem-aware default.",
    ),
    construction_compression: str = typer.Option(
        "default",
        "--construction-compression",
        help="Build-time compression profile: default, fast, or none.",
        case_sensitive=False,
    ),
    # Staging
    local_staging: bool = typer.Option(
        False,
        "--local-staging/--no-local-staging",
        help="Build under local scratch storage and publish to the output path after completion.",
    ),
    staging_dir: Path | None = typer.Option(
        None,
        "--staging-dir",
        help="Scratch directory to use for local staging. Defaults to TMPDIR when local staging is enabled.",
    ),
    # Misc
    test: bool = typer.Option(False, "--test", help="Restrict coverage to chr21/chr22/chrY (for testing)."),
    log_file: Path = typer.Option("quantnado_processing.log", "--log-file", help="Path to the log file."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
):
    """
    Create a multiomics store from BAM, methylation, and/or VCF files.

    At least one of --bam, --methylation, or --vcf must be provided.
    Multiple files can be passed as a comma-separated list, e.g. --bam a.bam,b.bam

    Coverage type (--coverage-type) controls how BAM reads are counted:

    \b
      unstranded          combined-strand coverage (default)
      stranded            separate forward and reverse arrays
      mcc                 Micro-Capture C — one virtual sample per viewpoint tag

    Pass a single value to apply to all BAM files, a comma-separated list
    in the same order as --bam, or sample:type pairs for per-sample control:

    \b
      --coverage-type stranded
      --coverage-type stranded,unstranded,mcc
      --coverage-type rna-rep1:stranded,chip-rep1:unstranded,capture:mcc
    """
    _setup_cli_logging(log_file, verbose)

    def _parse_coverage_type(raw: str | None) -> CoverageType | list[CoverageType] | dict[str, CoverageType]:
        """Convert the --coverage-type CLI string into the form BamStore expects.

        Accepted formats:
            single:  "stranded"
            list:    "stranded,unstranded"
            dict:    "sample1:stranded,sample2:mcc"
        """
        if raw is None:
            return CoverageType.UNSTRANDED
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        # Single bare value
        if len(parts) == 1 and ":" not in parts[0]:
            try:
                return CoverageType(parts[0].lower())
            except ValueError:
                logger.error(
                    f"--coverage-type '{parts[0]}' is not a valid coverage type. "
                    f"Valid values: {[e.value for e in CoverageType]}"
                )
                raise typer.Exit(code=1)
        # Detect dict form (any part contains ":")
        if any(":" in p for p in parts):
            result: dict[str, CoverageType] = {}
            for part in parts:
                if ":" not in part:
                    logger.error(
                        f"--coverage-type: '{part}' is missing a sample name — "
                        "use 'sample:type' pairs, e.g. 'sample1:stranded,sample2:mcc'"
                    )
                    raise typer.Exit(code=1)
                sample, _, type_str = part.partition(":")
                try:
                    result[sample.strip()] = CoverageType(type_str.strip().lower())
                except ValueError:
                    logger.error(
                        f"--coverage-type: '{type_str.strip()}' is not a valid coverage type. "
                        f"Valid values: {[e.value for e in CoverageType]}"
                    )
                    raise typer.Exit(code=1)
            return result
        # List form
        try:
            return [CoverageType(p.lower()) for p in parts]
        except ValueError as exc:
            logger.error(f"--coverage-type list contains invalid value: {exc}")
            raise typer.Exit(code=1)

    coverage_type_parsed = _parse_coverage_type(coverage_type)

    def _split(s: str | None) -> list[str]:
        return [v.strip() for v in s.split(",") if v.strip()] if s else []

    bam_files = _split(bam)
    all_meth_files = _split(methylation)
    vcf_files = _split(vcf)
    bam_names = _split(bam_sample_names)
    all_meth_names = _split(methylation_sample_names)
    vcf_names = _split(vcf_sample_names)

    methyldackel_files, cxreport_files, mc_files, hmc_files = classify_methylation_files(all_meth_files)

    # Slice sample names to match each classified group in order:
    # [bedgraph names][cxreport names][mc/hmc names (one per sample pair)]
    n_bg = len(methyldackel_files)
    n_cx = len(cxreport_files)
    n_mchmc = max(len(mc_files), len(hmc_files))
    bedgraph_names = all_meth_names[:n_bg] if all_meth_names else []
    cxreport_names = all_meth_names[n_bg:n_bg + n_cx] if all_meth_names else []
    mc_hmc_names = all_meth_names[n_bg + n_cx:n_bg + n_cx + n_mchmc] if all_meth_names else []

    if not any([bam_files, methyldackel_files, cxreport_files, mc_files, hmc_files, vcf_files]):
        logger.error("Provide at least one of --bam, --methylation, or --vcf.")
        raise typer.Exit(code=1)

    modality_counts = [
        f"{len(bam_files)} BAM" if bam_files else None,
        f"{len(methyldackel_files)} bedGraph" if methyldackel_files else None,
        f"{len(cxreport_files)} CXreport" if cxreport_files else None,
        f"{len(mc_files)} mC / {len(hmc_files)} hmC" if mc_files or hmc_files else None,
        f"{len(vcf_files)} VCF" if vcf_files else None,
    ]
    logger.info(f"Building multiomics store at {output}: {', '.join(m for m in modality_counts if m)}")

    try:
        from quantnado.dataset.store_multiomics import MultiomicsStore

        MultiomicsStore.from_files(
            store_dir=output,
            bam_files=bam_files or None,
            methyldackel_files=methyldackel_files or None,
            cxreport_files=cxreport_files or None,
            mc_files=mc_files or None,
            hmc_files=hmc_files or None,
            vcf_files=vcf_files or None,
            chromsizes=chromsizes,
            metadata=metadata,
            bam_sample_names=bam_names or None,
            methyldackel_sample_names=bedgraph_names or None,
            cxreport_sample_names=cxreport_names or None,
            mc_hmc_sample_names=mc_hmc_names or None,
            vcf_sample_names=vcf_names or None,
            sample_column=sample_column,
            chunk_len=chunk_len,
            filter_chromosomes=filter_chromosomes,
            construction_compression=construction_compression,
            local_staging=local_staging,
            staging_dir=staging_dir,
            log_file=log_file,
            overwrite=overwrite,
            resume=resume,
            max_workers=max_workers,
            test=test,
            coverage_type=coverage_type_parsed,
            count_fragments=count_fragments,
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

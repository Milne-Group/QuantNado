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
import pandas as pd
from pathlib import Path

from quantnado.dataset.store_bam import BamStore, CoverageType
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
    blacklist: Path | None = typer.Option(None, "--blacklist", help="Path to a BED file with regions to exclude"),
    # quantile options
    tilesize: int = typer.Option(128, "--tilesize", help="[quantile] Size of genomic tiles in bp"),
    window_overlap: int = typer.Option(8, "--window-overlap", help="[quantile] Overlap between adjacent windows in bp"),
    quantile: float = typer.Option(0.98, "--quantile", help="[quantile] Quantile threshold for peak calling"),
    merge: bool = typer.Option(True, "--merge/--no-merge", help="[quantile] Merge overlapping and adjacent peaks after calling"),
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
    n_workers: int = typer.Option(1, "--n-workers", help="Number of parallel workers for peak calling (seacr/lanceotron only)"),
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
        if method == "quantile":
            from quantnado.peak_calling.call_quantile_peaks import call_peaks_from_zarr

            call_peaks_from_zarr(
                zarr_path=zarr,
                output_dir=output_dir,
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
                score_threshold=score_threshold,
                blacklist_file=blacklist,
                smooth_window=smooth_window,
                batch_size=lanceotron_batch_size,
                n_workers=n_workers,
                device=device,
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
    # SeqNado convenience shortcut
    seqnado_dir: Path | None = typer.Option(
        None,
        "--seqnado-dir",
        help=(
            "Path to a SeqNado project directory. When provided, BAM/methylation/VCF files "
            "are discovered automatically from seqnado_output/ and design CSV files."
        ),
    ),
    subset_samples: int | None = typer.Option(
        None,
        "--subset-samples",
        help="When using --seqnado-dir, keep only the first N samples per assay.",
    ),
    sample: str | None = typer.Option(
        None,
        "--sample",
        help="Comma-separated sample_id value(s) to process. Filters the metadata to matching rows.",
    ),
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
    metadata: str | None = typer.Option(
        None,
        "--metadata",
        help=(
            "Path to metadata CSV file, or a comma-separated list of CSV files (e.g. one SeqNado design "
            "file per assay). Files are merged on shared columns; r1/r2 path columns are ignored. "
            "Any 'assay' column present in the CSV(s) is stored per sample automatically."
        ),
    ),
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
    # Assay labels (optional, one per file or broadcast from a single value)
    bam_assays: str | None = typer.Option(
        None,
        "--bam-assays",
        help="Assay label(s) for BAM files. Single value is broadcast to all BAM files; comma-separated list must match the number of BAM files.",
    ),
    methylation_assays: str | None = typer.Option(
        None,
        "--methylation-assays",
        help="Assay label(s) for methylation files. Single value is broadcast to all methylation files; comma-separated list must match the number of methylation files.",
    ),
    vcf_assays: str | None = typer.Option(
        None,
        "--vcf-assays",
        help="Assay label(s) for VCF files. Single value is broadcast to all VCF files; comma-separated list must match the number of VCF files.",
    ),
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

    # --- SeqNado shortcut ---
    if seqnado_dir is not None and not any([bam, methylation, vcf]):
        from quantnado.api import QuantNado
        _sample_list = [s.strip() for s in sample.split(",") if s.strip()] if sample else None
        _sample_arg: str | list[str] | None = (
            _sample_list[0] if _sample_list and len(_sample_list) == 1 else _sample_list
        )
        try:
            QuantNado.create_dataset(
                store_dir=output,
                seqnado_dir=seqnado_dir,
                subset_samples=subset_samples,
                sample=_sample_arg,
                sample_column=sample_column,
                filter_chromosomes=filter_chromosomes,
                overwrite=overwrite,
                resume=resume,
                max_workers=max_workers,
                chunk_len=chunk_len,
                construction_compression=construction_compression,
                local_staging=local_staging,
                staging_dir=staging_dir,
                test=test,
                log_file=log_file,
            )
            logger.success(f"Multiomics store created: {output}")
        except Exception as e:
            logger.error(f"Dataset creation failed: {e}")
            logger.debug(traceback.format_exc())
            raise typer.Exit(code=1)
        return

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

    # Load and merge metadata CSV(s). Multiple paths (comma-separated) are supported so that
    # per-assay SeqNado design files can be passed together, e.g.:
    #   --metadata metadata_atac.csv,metadata_chip.csv,metadata_rna.csv
    # r1/r2 fastq path columns are stripped; all other columns (assay, scaling_group, etc.) are kept.
    metadata_paths = [Path(p) for p in _split(metadata)]
    if not metadata_paths:
        base_metadata_df: pd.DataFrame | Path | None = None
    elif len(metadata_paths) == 1:
        base_metadata_df = metadata_paths[0]
    else:
        base_metadata_df = BamStore._combine_metadata_files(metadata_paths)

    if not any([bam_files, methyldackel_files, cxreport_files, mc_files, hmc_files, vcf_files]):
        logger.error("Provide at least one of --bam, --methylation, or --vcf.")
        raise typer.Exit(code=1)

    # Build optional assay metadata DataFrame.
    # Per-modality assay lists are resolved by broadcasting a single value
    # or validating that a multi-value list matches the file count.
    def _resolve_assays(raw: str | None, sample_names: list[str], label: str) -> list[tuple[str, str]]:
        """Return [(sample_name, assay), ...] or [] if no assays specified."""
        if raw is None:
            return []
        parts = [v.strip() for v in raw.split(",") if v.strip()]
        if len(parts) == 1:
            parts = parts * len(sample_names)
        elif len(parts) != len(sample_names):
            logger.error(
                f"--{label}-assays: got {len(parts)} value(s) but {len(sample_names)} sample(s). "
                "Provide a single value to broadcast or one value per sample."
            )
            raise typer.Exit(code=1)
        return list(zip(sample_names, parts))

    # Resolve effective sample names for each modality (mirrors what MultiomicsStore will use).
    effective_bam_names = bam_names if bam_names else [Path(f).stem for f in bam_files]
    effective_meth_names = (
        bedgraph_names + cxreport_names + mc_hmc_names
        if (bedgraph_names or cxreport_names or mc_hmc_names)
        else [Path(f).stem for f in (methyldackel_files + cxreport_files + mc_files)]
    )
    effective_vcf_names = vcf_names if vcf_names else [Path(f).stem for f in vcf_files]

    assay_pairs: list[tuple[str, str]] = (
        _resolve_assays(bam_assays, effective_bam_names, "bam")
        + _resolve_assays(methylation_assays, effective_meth_names, "methylation")
        + _resolve_assays(vcf_assays, effective_vcf_names, "vcf")
    )

    # Merge assay labels from --bam-assays/--methylation-assays/--vcf-assays into metadata.
    # When SeqNado design CSVs supply an 'assay' column these flags are not needed.
    if assay_pairs:
        assay_df = pd.DataFrame(assay_pairs, columns=[sample_column, "assay"])
        if base_metadata_df is not None:
            loaded = pd.read_csv(base_metadata_df) if isinstance(base_metadata_df, Path) else base_metadata_df
            if "assay" not in loaded.columns:
                loaded = loaded.merge(assay_df, on=sample_column, how="left")
            metadata_to_pass: pd.DataFrame | Path | None = loaded
        else:
            metadata_to_pass = assay_df
    else:
        metadata_to_pass = base_metadata_df
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
            metadata=metadata_to_pass,
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


@app.command()
def combine_stores(
    inputs: list[Path] = typer.Option(..., "--input", "-i", help="Per-sample Zarr store paths (repeat flag or pass multiple)"),
    output: Path = typer.Option(..., "--output", "-o", help="Path for the combined output Zarr store"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite output if it already exists"),
    log_file: Path = typer.Option(Path("quantnado_combine.log"), "--log-file", help="Path to log file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
) -> None:
    """Combine per-sample Zarr stores into one multi-sample store.

    The combined store is rechunked with all samples in one chunk per position
    window, so any region query reads all samples in a single I/O operation.

    Example::

        quantnado combine-stores -i s1.zarr -i s2.zarr -i s3.zarr -o combined.zarr
    """
    _setup_cli_logging(log_file, verbose)
    from quantnado.dataset.combine_stores import combine_bam_stores
    try:
        combine_bam_stores(inputs, output, overwrite=overwrite)
    except Exception as e:
        logger.error(f"combine-stores failed: {e}")
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

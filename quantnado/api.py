"""QuantNado API facade.

Thin wrapper around :class:`QuantNadoDataset <quantnado.analysis.core.QuantNadoDataset>`
that keeps the same analysis API while exposing a slightly friendlier entry point.

Example::

    qn = QuantNado.open("dataset/")           # directory of per-sample zarr stores
    qn = QuantNado.open("combined.zarr")      # combined multi-sample zarr

    region = qn.sel(chrom="chr1", start=1_000_000, end=1_001_000)
    reduced = qn.reduce("promoters.bed", reduction="mean", modality="coverage")
    binned = qn.extract(
        feature_type="promoter",
        GTF_FILE="genes.gtf",
        assay="ATAC",
        modality="coverage",
        upstream=1000,
        downstream=1000,
    )

    cpm = qn.normalise(reduced, method="cpm")
    qn.metaplot(binned, modality="coverage")
    qn.heatmap(reduced, variable="mean")
"""

from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence
from typing import Any

import pandas as pd
import xarray as xr
from loguru import logger

from quantnado.analysis.core import QuantNadoDataset
from quantnado.analysis.normalise import normalise as _normalise
from quantnado.dataset.metadata import AnchorPoint, FeatureType, ReductionMethod


# ---------------------------------------------------------------------------
# SeqNado integration helper
# ---------------------------------------------------------------------------


def metadata_from_seqnado(
    seqnado_dir: str | Path, output_dir: str | Path | None = None
) -> pd.DataFrame:
    """Build a QuantNado metadata DataFrame from a SeqNado project directory.

    Scans the standard SeqNado output layout for BAM files (indexed),
    MethylDackel bedGraphs, and VCF files, then joins them against all
    ``metadata_*.csv`` design files found in ``seqnado_dir``.

    Parameters
    ----------
    seqnado_dir:
        Root of the SeqNado project (contains ``metadata_*.csv`` files and a
        ``seqnado_output/`` subdirectory).
    output_dir:
        If given, writes ``quantnado_metadata.csv`` to this directory.

    Returns
    -------
    pd.DataFrame
        Columns: assay, sample_id, ip, bam_path, stranded, methylation_path,
        variant_path.
    """
    import yaml

    seqnado_dir = Path(seqnado_dir)
    out = seqnado_dir / "seqnado_output"

    def _index_is_fresh(bam: Path) -> bool:
        idx = Path(str(bam) + ".bai")
        return idx.exists() and idx.stat().st_mtime >= bam.stat().st_mtime

    bams = sorted(str(p) for p in out.glob("*/aligned/*.bam") if _index_is_fresh(p))
    stale = [
        str(p)
        for p in out.glob("*/aligned/*.bam")
        if Path(str(p) + ".bai").exists() and not _index_is_fresh(p)
    ]
    if stale:
        logger.warning(
            "Skipping {} BAM(s) with stale index (re-run samtools index): {}",
            len(stale),
            stale,
        )
    meth_files = sorted(str(p) for p in out.glob("meth/methylation/methyldackel/*.bedGraph"))

    _vcf_by_stem: dict[str, Path] = {}
    for _p in out.glob("snp/variant/*.vcf.gz"):
        _stem = _p.name.replace(".anno.vcf.gz", "").replace(".vcf.gz", "")
        if _stem not in _vcf_by_stem or ".anno." in _p.name:
            _vcf_by_stem[_stem] = _p
    snp_files = sorted(str(p) for p in _vcf_by_stem.values())

    design_files = sorted(seqnado_dir.glob("metadata_*.csv"))
    if not design_files:
        raise FileNotFoundError(f"No metadata_*.csv files found in {seqnado_dir}")

    metadata = pd.concat([pd.read_csv(f) for f in design_files], ignore_index=True)
    # Ensure 'ip' column exists (may not be present in all metadata files)
    if "ip" not in metadata.columns:
        metadata["ip"] = None
    metadata = metadata[["assay", "sample_id", "ip"]]
    metadata["sample_id"] = metadata.apply(
        lambda row: f"{row['sample_id']}_{row['ip']}" if pd.notna(row["ip"]) else row["sample_id"],
        axis=1,
    )
    
    # Try to match metadata to BAM files
    if len(bams) > 0:
        metadata = metadata[metadata["sample_id"].apply(lambda s: any(s in b for b in bams))].copy()
        metadata["bam_path"] = metadata["sample_id"].apply(
            lambda s: next((b for b in bams if s in b), None)
        )
    else:
        # No BAM files found — log warning and set bam_path to None
        logger.warning(
            "No indexed BAM files found in {}. Found metadata for assays: {}. "
            "Set bam_path to None. Download BAM files if needed and re-run.",
            out,
            metadata["assay"].unique().tolist(),
        )
        metadata["bam_path"] = None

    rna_config_path = seqnado_dir / "config_rna.yaml"
    if rna_config_path.exists():
        with open(rna_config_path) as f:
            rna_config = yaml.safe_load(f)
        config_strand = rna_config["assay_config"]["rna_quantification"]["strandedness"]
        strand_value = "R" if config_strand == 2 else "F" if config_strand == 1 else "U"
    else:
        strand_value = "U"
    metadata["stranded"] = metadata["assay"].apply(lambda a: strand_value if a == "RNA" else None)
    metadata["methylation_path"] = metadata["sample_id"].apply(
        lambda s: next((m for m in meth_files if s in m), None)
    )
    metadata["variant_path"] = metadata["sample_id"].apply(
        lambda s: next((v for v in snp_files if s in v), None)
    )
    snp_with_vcf = (metadata["assay"].str.upper() == "SNP") & metadata["variant_path"].notna()
    metadata.loc[snp_with_vcf, "bam_path"] = None
    metadata.reset_index(drop=True, inplace=True)

    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        metadata.to_csv(Path(output_dir) / "quantnado_metadata.csv", index=False)
    return metadata


# ---------------------------------------------------------------------------
# Per-sample creation helper
# ---------------------------------------------------------------------------


def create_dataset(
    sample_id: str,
    assay: str,
    output_path: "Path | str",
    *,
    bam_path: "str | Path | None" = None,
    methyl_path: "str | Path | None" = None,
    vcf_path: "str | Path | None" = None,
    variants_path: "str | Path | None" = None,  # alias for vcf_path
    ip: "str | None" = None,
    stranded: "str | None" = None,
    chromsizes: "str | Path | dict | None" = None,
    chunk_len: "int | None" = None,
    construction_compression: str = "default",
    overwrite: bool = True,
    filter_chromosomes: bool = True,
    test: bool = False,
    test_chromosomes: "Sequence[str] | None" = None,
    log_file: "Path | None" = None,
):
    """Create a single per-sample zarr store and return its path.

    Dispatches to the correct builder based on ``assay``.

    Parameters
    ----------
    sample_id:
        Sample name — used as the zarr's ``sample`` attribute.
    assay:
        Assay type: ``"ATAC"``, ``"ChIP"``, ``"RNA"``, ``"CUT&TAG"``,
        ``"MCC"``, ``"METH"``, or ``"SNP"``.
    output_path:
        Where to write the ``.zarr`` store.
    bam_path:
        BAM file path (required for all except SNP).
    methyl_path:
        MethylDackel bedGraph (required for METH).
    vcf_path:
        Annotated VCF (required for SNP).
    ip:
        IP target name for ChIP / CUT&TAG (e.g. ``"H3K27ac"``).
    stranded:
        Strand orientation for RNA: "R" (reverse), "F" (forward), or None (unstranded).
    chromsizes:
        Path to ``.chrom.sizes``, a dict, or ``None`` to infer from BAM/VCF.

    Returns
    -------
    Path
        Path to the written ``.zarr`` store.

    Examples
    --------
    >>> qn.create_dataset("ATAC-1", "ATAC", "dataset/ATAC-1.zarr", bam_path="ATAC-1.bam")
    >>> qn.create_dataset("TAPS", "METH", "dataset/TAPS.zarr",
    ...     bam_path="TAPS.bam", methyl_path="TAPS_CpG.bedGraph")
    >>> qn.create_dataset("gDNA", "SNP", "dataset/gDNA.zarr", vcf_path="gDNA.vcf.gz")
    >>> qn.create_dataset("RNA-1", "RNA", "dataset/RNA-1.zarr", bam_path="RNA-1.bam", stranded="R")
    """
    vcf_path = vcf_path or variants_path
    assay_upper = assay.upper()
    output_path = Path(output_path)

    shared = dict(
        chromsizes=chromsizes,
        chunk_len=chunk_len,
        construction_compression=construction_compression,
        overwrite=overwrite,
        filter_chromosomes=filter_chromosomes,
        test=test,
        test_chromosomes=list(test_chromosomes) if test_chromosomes is not None else None,
        log_file=log_file,
    )

    if assay_upper == "METH":
        from quantnado.dataset.store_methyl import MethylStore
        if not bam_path or not methyl_path:
            raise ValueError("METH requires both bam_path and methyl_path")
        MethylStore.from_files(
            bam_path=bam_path,
            methyl_path=methyl_path,
            store_path=output_path,
            sample=sample_id,
            **shared,
        )
    elif assay_upper == "SNP":
        from quantnado.dataset.store_variants import VariantStore
        if not vcf_path:
            raise ValueError("SNP requires vcf_path")
        VariantStore.from_vcf(
            vcf_path=vcf_path,
            store_path=output_path,
            sample=sample_id,
            **shared,
        )
    else:
        from quantnado.dataset.store_bam import BamStore
        if not bam_path:
            raise ValueError(f"{assay} requires bam_path")
        BamStore.from_bam_files(
            bam_path=bam_path,
            store_path=output_path,
            assay=assay,
            sample=sample_id,
            ip=ip or None,
            stranded=stranded,
            **shared,
        )

    return output_path


# ---------------------------------------------------------------------------
# QuantNado facade
# ---------------------------------------------------------------------------


class QuantNado:
    """Facade wrapping :class:`QuantNadoDataset` with analysis convenience methods.

    Construction
    ------------
    >>> qn = QuantNado.open("dataset/")        # directory of per-sample zarrs
    >>> qn = QuantNado.open("combined.zarr")    # combined store

    Basic access
    ------------
    >>> qn.sample_names
    >>> qn.assays
    >>> qn.chromosomes
    >>> region = qn.sel(chrom="chr1", start=1_000_000, end=1_001_000)
    >>> tree = qn.to_datatree()
    """

    def __init__(self, dataset: QuantNadoDataset) -> None:
        self._dataset = dataset

    def __getattr__(self, name):
        return getattr(self._dataset, name)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def open(cls, path: str | Path) -> "QuantNado":
        """Open an existing QuantNado store (auto-detects layout).

        Parameters
        ----------
        path:
            Directory of per-sample ``.zarr`` stores, a combined ``.zarr``, or a
            ``.tar.gz``/``.tgz`` archive containing either layout.
        """
        return cls(QuantNadoDataset(path))

    # Alias kept for backward compatibility
    open_dataset = open

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def sample_names(self) -> list[str]:
        return self._dataset.sample_names

    @property
    def assays(self) -> list[str]:
        return self._dataset.assays

    @property
    def chromosomes(self) -> list[str]:
        return self._dataset.chromosomes

    @property
    def chromsizes(self) -> dict[str, int]:
        return self._dataset.chromsizes

    @property
    def completed_mask(self):
        return self._dataset.completed_mask

    @property
    def groups(self):
        return self._dataset.groups

    @property
    def metadata(self) -> pd.DataFrame:
        return self._dataset.metadata

    @property
    def available_peak_methods(self) -> list[str]:
        return self._dataset.available_peak_methods

    def group_by(
        self,
        by: str = "assay",
        *,
        groups: "dict[str, list[str] | str] | None" = None,
        match: str = "exact",
        drop_empty: bool = True,
        **named_groups: "dict[str, list[str] | str]",
    ):
        return self._dataset.group_by(
            by=by,
            groups=groups,
            match=match,
            drop_empty=drop_empty,
            **named_groups,
        )

    @property
    def info(self) -> dict[str, object]:
        return self._dataset.info

    def subset(
        self,
        assay: "str | Sequence[str] | None" = None,
        samples: "str | Sequence[str] | None" = None,
        ip: "str | Sequence[str] | None" = None,
        group: "str | Sequence[str] | dict[str, str | Sequence[str]] | None" = None,
    ) -> "QuantNado":
        """Return a filtered QuantNado view. See :meth:`QuantNadoDataset.subset`."""
        return QuantNado(self._dataset.subset(assay=assay, samples=samples, ip=ip, group=group))

    def info_of(self, obj):
        """Return a compact summary for xarray / pandas objects."""
        return self._dataset.info_of(obj)

    # ------------------------------------------------------------------
    # Region access
    # ------------------------------------------------------------------

    def sel(
        self,
        chrom: str,
        start: int | None = None,
        end: int | None = None,
        assay: str | Sequence[str] | None = None,
        samples: str | Sequence[str] | None = None,
    ) -> xr.Dataset:
        """Extract a genomic region as an xr.Dataset.

        Parameters
        ----------
        chrom:
            Chromosome name (e.g. ``"chr1"``).
        start:
            1-based start position (inclusive). Defaults to 1.
        end:
            1-based end position (inclusive). Defaults to chromosome length.
        assay:
            Restrict to samples of this assay type.
        samples:
            Explicit sample names (overrides *assay*).

        Returns
        -------
        xr.Dataset
            dims: sample × position; one data_var per assay.
        """
        return self._dataset.sel(chrom=chrom, start=start, end=end, assay=assay, samples=samples)

    def to_datatree(self, chromosomes: list[str] | None = None) -> xr.DataTree:
        """Return the full dataset as an xr.DataTree (one node per chromosome)."""
        return self._dataset.to_datatree(chromosomes=chromosomes)

    @classmethod
    def combine(
        cls,
        src: Path | str,
        output: Path | str,
        overwrite: bool = True,
        n_workers: int = 1,
    ) -> "QuantNado":
        """Combine a directory of per-sample zarrs into a single combined zarr."""
        return cls(
            QuantNadoDataset.combine(
                src,
                output,
                overwrite=overwrite,
                n_workers=n_workers,
            )
        )

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def normalise(
        self,
        data: "xr.Dataset | xr.DataArray | pd.DataFrame | None" = None,
        *,
        method: str = "cpm",
        assay: str | Sequence[str] | None = None,
        samples: str | Sequence[str] | None = None,
        library_sizes: "pd.Series | dict | None" = None,
        feature_lengths: "pd.Series | Any | None" = None,
    ) -> "xr.Dataset | xr.DataArray | pd.DataFrame | Any":
        """Normalise coverage signal or feature counts. See :meth:`QuantNadoDataset.normalise`."""
        return self._dataset.normalise(
            data, method=method, assay=assay, samples=samples,
            library_sizes=library_sizes, feature_lengths=feature_lengths,
        )

    def pca(
        self,
        data_or_query=None,
        n_components: int = 5,
        assay: str | Sequence[str] | None = None,
        samples: str | Sequence[str] | None = None,
        modality: str | Sequence[str] | None = None,
        chromosome: str | None = None,
        nan_handling_strategy: str = "drop",
        standardize: bool = False,
        random_state: int | None = None,
        subset_size: int | None = None,
        subset_strategy: str = "random",
    ) -> tuple[Any, xr.DataArray]:
        """Run PCA on reduced genomic signal. See :meth:`QuantNadoDataset.pca`."""
        return self._dataset.pca(
            data_or_query, n_components=n_components, assay=assay, samples=samples,
            modality=modality,
            chromosome=chromosome, nan_handling_strategy=nan_handling_strategy,
            standardize=standardize, random_state=random_state,
            subset_size=subset_size, subset_strategy=subset_strategy,
        )

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def metaplot(
        self,
        data: xr.DataArray,
        data_rev: xr.DataArray | None = None,
        *,
        assay: str | Sequence[str] | None = None,
        samples: str | Sequence[str] | None = None,
        modality: str | Sequence[str] | None = None,
        groups: dict[str, list[str]] | None = None,
        flip_minus_strand: bool = True,
        error_stat: str | None = "sem",
        palette=None,
        reference_point: float | None = 0,
        reference_label: str = "TSS",
        xlabel: str = "Relative position",
        ylabel: str | None = None,
        title: str = "Metagene profile",
        figsize: tuple[float, float] = (8, 4),
        ax: Any = None,
        filepath: str | Path | None = None,
    ) -> Any:
        """Plot a metagene profile. See :meth:`QuantNadoDataset.metaplot`."""
        return self._dataset.metaplot(
            data, data_rev,
            assay=assay, samples=samples, modality=modality,
            groups=groups, flip_minus_strand=flip_minus_strand,
            error_stat=error_stat, palette=palette,
            reference_point=reference_point, reference_label=reference_label,
            xlabel=xlabel, ylabel=ylabel, title=title, figsize=figsize,
            ax=ax, filepath=filepath,
        )

    def tornadoplot(
        self,
        data: xr.DataArray,
        data_rev: xr.DataArray | None = None,
        *,
        assay: str | Sequence[str] | None = None,
        samples: str | Sequence[str] | None = None,
        modality: str | Sequence[str] | None = None,
        sample_names: list[str] | None = None,
        groups: dict[str, list[str]] | None = None,
        flip_minus_strand: bool = True,
        sort_by: str | None = "mean",
        vmin: float | None = None,
        vmax: float | None = None,
        scale_each: bool = False,
        cmap: str | None = None,
        reference_point: float | None = 0,
        reference_label: str = "TSS",
        xlabel: str = "Relative position",
        ylabel: str | None = None,
        title: str = "Signal heatmap",
        figsize: tuple[float, float] | None = None,
        filepath: str | Path | None = None,
    ) -> list:
        """Tornado / heatmap plot. See :meth:`QuantNadoDataset.tornadoplot`."""
        return self._dataset.tornadoplot(
            data, data_rev,
            assay=assay, samples=samples, modality=modality,
            sample_names=sample_names, groups=groups,
            flip_minus_strand=flip_minus_strand, sort_by=sort_by,
            vmin=vmin, vmax=vmax, scale_each=scale_each, cmap=cmap,
            reference_point=reference_point, reference_label=reference_label,
            xlabel=xlabel, ylabel=ylabel, title=title, figsize=figsize,
            filepath=filepath,
        )

    def heatmap(
        self,
        data: xr.Dataset,
        *,
        assay: str | Sequence[str] | None = None,
        samples: str | Sequence[str] | None = None,
        variable: str = "mean",
        title: str = "Signal heatmap",
        cmap: str = "viridis",
        figsize: tuple[float, float] = (10, 8),
        filepath: str | Path | None = None,
        **kwargs,
    ) -> Any:
        """Heatmap of reduced signal. See :meth:`QuantNadoDataset.heatmap`."""
        return self._dataset.heatmap(
            data,
            assay=assay, samples=samples, variable=variable,
            title=title, cmap=cmap, figsize=figsize, filepath=filepath,
            **kwargs,
        )

    def correlate(
        self,
        data: xr.Dataset,
        *,
        assay: str | Sequence[str] | None = None,
        samples: str | Sequence[str] | None = None,
        variable: str = "mean",
        method: str = "pearson",
        title: str = "Sample correlation",
        figsize: tuple[float, float] = (8, 7),
        filepath: str | Path | None = None,
        **kwargs,
    ) -> tuple[pd.DataFrame, Any]:
        """Compute and plot sample correlation. See :meth:`QuantNadoDataset.correlate`."""
        return self._dataset.correlate(
            data,
            assay=assay, samples=samples, variable=variable,
            method=method, title=title, figsize=figsize, filepath=filepath,
            **kwargs,
        )

    def locus_plot(self, *args, **kwargs) -> Any:
        """Plot a genomic locus. See :meth:`QuantNadoDataset.locus_plot`."""
        return self._dataset.locus_plot(*args, **kwargs)

    # ------------------------------------------------------------------
    # Data extraction / reduction
    # ------------------------------------------------------------------

    def reduce(
        self,
        intervals_path: str | None = None,
        ranges_df=None,
        gtf_path: str | None = None,
        feature_type: str | None = None,
        reduction: str = "mean",
        assay: str | Sequence[str] | None = None,
        samples: str | Sequence[str] | None = None,
        modality: str | Sequence[str] | None = None,
        progress: bool = False,
        workers: int | None = None,
        **kwargs,
    ):
        """Reduce signal over genomic intervals. See :meth:`QuantNadoDataset.reduce`."""
        return self._dataset.reduce(
            intervals_path=intervals_path,
            ranges_df=ranges_df,
            gtf_path=gtf_path,
            feature_type=feature_type,
            reduction=reduction,
            assay=assay,
            samples=samples,
            modality=modality,
            progress=progress,
            workers=workers,
            **kwargs,
        )

    def count_features(
        self,
        gtf_file: str | None = None,
        bed_file: str | None = None,
        ranges_df=None,
        feature_type: str = "gene",
        engine: str = "signal",
        assay: str | Sequence[str] | None = None,
        samples: str | Sequence[str] | None = None,
        modality: str | Sequence[str] | None = None,
        **kwargs,
    ):
        """Count reads over features. See :meth:`QuantNadoDataset.count_features`."""
        return self._dataset.count_features(
            gtf_file=gtf_file,
            bed_file=bed_file,
            ranges_df=ranges_df,
            feature_type=feature_type,
            engine=engine,
            assay=assay,
            samples=samples,
            modality=modality,
            **kwargs,
        )

    def quantify_signal(
        self,
        gtf_file: str | None = None,
        bed_file: str | None = None,
        ranges_df=None,
        feature_type: str = "gene",
        assay: str | Sequence[str] | None = None,
        samples: str | Sequence[str] | None = None,
        modality: str | Sequence[str] | None = None,
        return_metadata: bool = True,
        **kwargs,
    ):
        """Quantify stored signal over features. See :meth:`QuantNadoDataset.quantify_signal`."""
        return self._dataset.quantify_signal(
            gtf_file=gtf_file,
            bed_file=bed_file,
            ranges_df=ranges_df,
            feature_type=feature_type,
            assay=assay,
            samples=samples,
            modality=modality,
            return_metadata=return_metadata,
            **kwargs,
        )

    def extract(self, *args, **kwargs):
        """Extract signal into bins. See :meth:`QuantNadoDataset.extract`."""
        return self._dataset.extract(*args, **kwargs)

    def library_sizes(self, assay: str | Sequence[str] | None = None, samples: str | Sequence[str] | None = None):
        """Return total mapped reads per sample. See :meth:`QuantNadoDataset.library_sizes`."""
        return self._dataset.library_sizes(assay=assay, samples=samples)

    # ------------------------------------------------------------------
    # PCA extras
    # ------------------------------------------------------------------

    def pca_scree(self, pca_obj, **kwargs) -> Any:
        """Plot PCA scree. See :meth:`QuantNadoDataset.pca_scree`."""
        return self._dataset.pca_scree(pca_obj, **kwargs)

    def pca_scatter(self, pca_obj, pca_result, colour_by=None, shape_by=None, **kwargs) -> Any:
        """Scatter plot of PCA-transformed samples. See :meth:`QuantNadoDataset.pca_scatter`."""
        return self._dataset.pca_scatter(pca_obj, pca_result, colour_by=colour_by, shape_by=shape_by, **kwargs)

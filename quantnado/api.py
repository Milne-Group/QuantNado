"""
Unified QuantNado API for genomic signal analysis.

This module provides a high-level `QuantNado` facade that unifies dataset creation,
loading, signal reduction, feature counting, and PCA analysis into a single,
convenient interface.

Example:
    # Create a new multi-modal dataset
    qn = QuantNado.from_files(
        store_dir="dataset/",
        bam_files=["sample1.bam", "sample2.bam"],
        bedgraph_files=["sample1.bedGraph"],
        vcf_files=["sample1.vcf.gz"],
    )

    # Open existing dataset (auto-detects MultiomicsStore or BamStore)
    qn = QuantNado.open("dataset/")

    # Access sub-stores
    qn.coverage     # BamStore
    qn.methylation  # MethylStore
    qn.variants     # VariantStore

    # Analyse coverage signal
    reduced = qn.reduce("promoters.bed", reduction="mean")
    pca_obj, transformed = qn.pca(reduced["mean"], n_components=10)
    counts, features = qn.count_features("genes.gtf", feature_type="gene")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import xarray as xr

from quantnado.analysis.counts import count_features as _feature_counts
from quantnado.analysis.pca import run_pca as _run_pca
from quantnado.analysis.plot import locus_plot, metaplot, tornadoplot
from quantnado.analysis.reduce import extract_byranges_signal, reduce_byranges_signal
from quantnado.dataset.enums import AnchorPoint, FeatureType, ReductionMethod
from quantnado.dataset.store_bam import BamStore
from quantnado.dataset.store_methyl import MethylStore
from quantnado.dataset.store_multiomics import DEFAULT_CHUNK_LEN, MultiomicsStore
from quantnado.dataset.store_variants import VariantStore


class QuantNado:
    """
    Unified facade for QuantNado genomic analysis.

    Wraps a ``MultiomicsStore`` (or a bare ``BamStore``) and provides
    properties to access each modality plus convenience methods for
    coverage-based analysis (reduce, extract, count_features, pca,
    metaplot, tornadoplot).

    Construction
    ------------
    >>> qn = QuantNado.open("dataset/")            # MultiomicsStore directory
    >>> qn = QuantNado.open("coverage.zarr")        # BamStore only
    >>> qn = QuantNado.from_bam_files(              # BAM-only dataset
    ...     bam_files=["s1.bam", "s2.bam"],
    ...     store_path="coverage.zarr",
    ... )
    >>> qn = QuantNado.create_dataset(              # multi-omics dataset
    ...     store_dir="dataset/",
    ...     bam_files=["s1.bam"],
    ...     bedgraph_files=["s1.bedGraph"],
    ... )

    Modality access
    ---------------
    >>> qn.coverage     # BamStore | None
    >>> qn.methylation  # MethylStore | None
    >>> qn.variants     # VariantStore | None
    >>> qn.modalities   # ['coverage', 'methylation', ...]
    """

    def __init__(self, store: MultiomicsStore | BamStore) -> None:
        if isinstance(store, MultiomicsStore):
            self._multiomics: MultiomicsStore | None = store
            self._bam: BamStore | None = store.coverage
        elif isinstance(store, BamStore):
            self._multiomics = None
            self._bam = store
        else:
            raise TypeError(
                f"store must be a MultiomicsStore or BamStore, got {type(store).__name__}"
            )

    # ========== Construction ==========

    @classmethod
    def open(cls, path: str | Path, read_only: bool = True) -> "QuantNado":
        """
        Open an existing QuantNado dataset.

        Auto-detects whether ``path`` is a ``MultiomicsStore`` directory or a
        single ``BamStore`` (``.zarr``).

        Parameters
        ----------
        path : str or Path
            Path to the store directory or ``.zarr`` file.
        read_only : bool, default True
            If True, disables write operations (BamStore only).

        Returns
        -------
        QuantNado
        """
        path = Path(path)
        if str(path).endswith(".zarr"):
            return cls(BamStore.open(path, read_only=read_only))
        zarr_path = path.with_suffix(".zarr")
        if zarr_path.exists():
            return cls(BamStore.open(zarr_path, read_only=read_only))
        return cls(MultiomicsStore.open(path))

    @classmethod
    def from_bam_files(
        cls,
        bam_files: list[str],
        store_path: "str | Path",
        chromsizes: "str | Path | dict[str, int] | None" = None,
        **kwargs,
    ) -> "QuantNado":
        """
        Create a QuantNado dataset from BAM files.

        Parameters
        ----------
        bam_files : list of str
            BAM file paths.
        store_path : str or Path
            Output Zarr store path.
        chromsizes : str, Path, dict, or None
            Chromosome sizes. Extracted from first BAM if not provided.
        **kwargs
            Passed through to ``BamStore.from_bam_files``.

        Returns
        -------
        QuantNado
        """
        store = BamStore.from_bam_files(
            bam_files=bam_files,
            store_path=store_path,
            chromsizes=chromsizes,
            **kwargs,
        )
        return cls(store)

    @classmethod
    def create_dataset(
        cls,
        store_dir: str | Path,
        bam_files: list[str | Path] | None = None,
        bedgraph_files: list[str | Path] | None = None,
        vcf_files: list[str | Path] | None = None,
        chromsizes: str | Path | dict[str, int] | None = None,
        metadata: pd.DataFrame | Path | str | None = None,
        *,
        bam_sample_names: list[str] | callable | None = None,
        bedgraph_sample_names: list[str] | callable | None = None,
        vcf_sample_names: list[str] | callable | None = None,
        filter_chromosomes: bool = True,
        overwrite: bool = True,
        resume: bool = False,
        sample_column: str = "sample_id",
        chunk_len: int = DEFAULT_CHUNK_LEN,
        construction_compression: str = "default",
        local_staging: bool = False,
        staging_dir: str | Path | None = None,
        log_file: Path | None = None,
        max_workers: int = 1,
        chr_workers: int = 1,
        test: bool = False,
    ) -> "QuantNado":
        """
        Create a new QuantNado dataset from genomic files.

        At least one of ``bam_files``, ``bedgraph_files``, or ``vcf_files``
        must be provided.

        Parameters
        ----------
        store_dir : str or Path
            Output directory. Created if it does not exist.
        bam_files : list of Path, optional
            BAM files for per-base coverage storage.
        bedgraph_files : list of Path, optional
            MethylDackel CpG bedGraph files for methylation storage.
        vcf_files : list of Path, optional
            VCF.gz files (one per sample) for variant storage.
        chromsizes : str, Path, or dict, optional
            Chromosome sizes. Extracted from the first BAM if not provided.
        metadata : DataFrame, Path, or str, optional
            Sample metadata CSV attached to all sub-stores.
        bam_sample_names : list of str or callable, optional
            Override sample names for BAM files. A callable receives each
            ``Path`` and returns a ``str`` (e.g. ``lambda p: p.stem``).
        bedgraph_sample_names : list of str or callable, optional
            Override sample names for bedGraph files. A callable receives each
            ``Path`` and returns a ``str``
            (e.g. ``lambda p: p.stem.split("_hg38")[0]``).
        vcf_sample_names : list of str or callable, optional
            Override sample names for VCF files. A callable receives each
            ``Path`` and returns a ``str``.
        filter_chromosomes : bool, default True
            Keep only canonical chromosomes (``chr*`` without underscores).
        overwrite : bool, default True
            Overwrite existing sub-stores.
        resume : bool, default False
            Resume processing an existing sub-store.
        sample_column : str, default "sample_id"
            Column in ``metadata`` matching sample names.
        chunk_len : int, default 65536
            Zarr chunk size for the position dimension (coverage store).
        construction_compression : {"default", "fast", "none"}, default "default"
            Build-time compression profile for the coverage store.
        local_staging : bool, default False
            Build the coverage store under local scratch before publishing.
        staging_dir : str or Path, optional
            Scratch directory for local staging.
        log_file : Path, optional
            Path to write BAM processing logs.
        max_workers : int, default 1
            Sample-level parallel workers for BAM processing.
        chr_workers : int, default 1
            Chromosome-level parallel workers within each sample thread.
            Total concurrent BAM reads = max_workers * chr_workers.
            On SSD/NVMe or HPC parallel filesystems, values of 2-4 can
            significantly reduce wall time.
        test : bool, default False
            Restrict coverage to chr21/chr22/chrY (for testing).

        Returns
        -------
        QuantNado
        """

        def _resolve_names(names, files):
            if callable(names) and files is not None:
                return [names(Path(f)) for f in files]
            return names

        bam_sample_names = _resolve_names(bam_sample_names, bam_files)
        bedgraph_sample_names = _resolve_names(bedgraph_sample_names, bedgraph_files)
        vcf_sample_names = _resolve_names(vcf_sample_names, vcf_files)

        ms = MultiomicsStore.from_files(
            store_dir=store_dir,
            bam_files=bam_files,
            bedgraph_files=bedgraph_files,
            vcf_files=vcf_files,
            chromsizes=chromsizes,
            metadata=metadata,
            bam_sample_names=bam_sample_names,
            bedgraph_sample_names=bedgraph_sample_names,
            vcf_sample_names=vcf_sample_names,
            filter_chromosomes=filter_chromosomes,
            overwrite=overwrite,
            resume=resume,
            sample_column=sample_column,
            chunk_len=chunk_len,
            construction_compression=construction_compression,
            local_staging=local_staging,
            staging_dir=staging_dir,
            log_file=log_file,
            max_workers=max_workers,
            chr_workers=chr_workers,
            test=test,
        )
        return cls(ms)

    # ========== Modality Access ==========

    @property
    def coverage(self) -> BamStore | None:
        """The coverage (BAM) sub-store, or None if not present."""
        return self._bam

    @property
    def methylation(self) -> MethylStore | None:
        """The methylation sub-store, or None if not present."""
        return self._multiomics.methylation if self._multiomics else None

    @property
    def variants(self) -> VariantStore | None:
        """The variant sub-store, or None if not present."""
        return self._multiomics.variants if self._multiomics else None

    @property
    def modalities(self) -> list[str]:
        """List of available modalities."""
        if self._multiomics:
            return self._multiomics.modalities
        return ["coverage"] if self._bam else []

    # ========== Coverage Properties ==========

    def _require_coverage(self) -> BamStore:
        if self._bam is None:
            raise RuntimeError("No coverage store available in this QuantNado instance.")
        return self._bam

    @property
    def store(self) -> BamStore:
        """The underlying coverage BamStore (raises if not present)."""
        return self._require_coverage()

    @property
    def store_path(self) -> Path:
        """Path to the coverage Zarr store."""
        return self._require_coverage().store_path

    @property
    def samples(self) -> list[str]:
        """Sample names in the coverage store."""
        return self._require_coverage().sample_names

    @property
    def chromosomes(self) -> list[str]:
        """Chromosome names in the coverage store."""
        return self._require_coverage().chromosomes

    @property
    def chromsizes(self) -> dict[str, int]:
        """Mapping of chromosome names to sizes."""
        return self._require_coverage().chromsizes

    @property
    def metadata(self) -> pd.DataFrame:
        """Metadata DataFrame. Combined across all modalities if multiomics."""
        return self.get_metadata()

    def get_metadata(self) -> pd.DataFrame:
        """
        Return metadata as a DataFrame.

        If this is a multiomics store, returns combined metadata across all
        modalities with a ``modalities`` column. Otherwise returns coverage
        store metadata.
        """
        if self._multiomics is not None:
            return self._multiomics.get_metadata()
        return self._require_coverage().get_metadata()

    @property
    def n_completed(self) -> int:
        """Number of fully processed samples in the coverage store."""
        return self._require_coverage().n_completed

    # ========== Data Access ==========

    def to_xarray(
        self,
        chromosomes: list[str] | None = None,
        chunks: str | dict | None = None,
    ) -> dict[str, xr.DataArray]:
        """
        Extract the coverage dataset as per-chromosome Xarray DataArrays.

        Parameters
        ----------
        chromosomes : list of str, optional
            Chromosomes to extract. If None, extracts all.
        chunks : str or dict, optional
            Dask chunking strategy.

        Returns
        -------
        dict[str, DataArray]
        """
        return self._require_coverage().to_xarray(chromosomes=chromosomes, chunks=chunks)

    def extract_region(
        self,
        region: str | None = None,
        chrom: str | None = None,
        start: int | None = None,
        end: int | None = None,
        samples: list[str] | list[int] | None = None,
        as_xarray: bool = True,
    ) -> xr.DataArray | Any:
        """
        Extract coverage signal for a specific genomic region.

        Parameters
        ----------
        region : str, optional
            Region string e.g. ``"chr1:1000-5000"``.
        chrom, start, end : optional
            Alternative to ``region``.
        samples : list of str or int, optional
            Sample names or indices. If None, uses all completed samples.
        as_xarray : bool, default True
            Return DataArray; if False return numpy array.

        Returns
        -------
        DataArray or ndarray
        """
        return self._require_coverage().extract_region(
            region=region,
            chrom=chrom,
            start=start,
            end=end,
            samples=samples,
            as_xarray=as_xarray,
        )

    # ========== Analysis Methods ==========

    def reduce(
        self,
        intervals_path: str | Path | None = None,
        ranges_df: Any | None = None,
        feature_type: FeatureType | str | None = None,
        gtf_path: str | Path | Iterable[str | Path] | None = None,
        reduction: ReductionMethod | str = ReductionMethod.MEAN,
        filter_incomplete: bool = True,
    ) -> xr.Dataset:
        """
        Reduce per-sample coverage signal over genomic ranges.

        Parameters
        ----------
        intervals_path : str or Path, optional
            Path to BED or GTF file.
        ranges_df : DataFrame or PyRanges, optional
            Pre-parsed genomic ranges.
        feature_type : FeatureType or str, optional
            Predefined feature type to extract from GTF.
        gtf_path : str, Path, or Iterable, optional
            Path(s) to GTF file(s).
        reduction : ReductionMethod or str, default "mean"
            Aggregation statistic: 'mean', 'sum', 'max', 'min', 'median'.
        filter_incomplete : bool, default True
            Exclude samples not yet marked complete.

        Returns
        -------
        Dataset
            Dimensions: (ranges, sample).
        """
        return reduce_byranges_signal(
            self._require_coverage(),
            ranges_df=ranges_df,
            intervals_path=intervals_path,
            feature_type=feature_type,
            gtf_path=gtf_path,
            reduction=reduction,
            include_incomplete=not filter_incomplete,
        )

    def extract(
        self,
        intervals_path: str | Path | None = None,
        ranges_df: Any | None = None,
        feature_type: FeatureType | str | None = None,
        gtf_path: str | Path | Iterable[str | Path] | None = None,
        fixed_width: int | None = None,
        upstream: int | None = None,
        downstream: int | None = None,
        anchor: AnchorPoint | str = AnchorPoint.MIDPOINT,
        bin_size: int | None = None,
        bin_agg: ReductionMethod | str = ReductionMethod.MEAN,
        filter_incomplete: bool = True,
        modality: str | None = None,
        variable: str | None = None,
        samples: list[str] | None = None,
    ) -> xr.DataArray:
        """
        Extract signal over genomic ranges.

        Routes to the appropriate modality sub-store based on ``modality``.

        Parameters
        ----------
        intervals_path : str or Path, optional
            Path to BED or GTF file.
        ranges_df : DataFrame or PyRanges, optional
            Pre-parsed genomic ranges.
        feature_type : FeatureType or str, optional
            Predefined feature type to extract from GTF.
        gtf_path : str, Path, or Iterable, optional
            Path(s) to GTF file(s).
        fixed_width : int, optional
            Symmetric window total width centered on anchor.
        upstream : int, optional
            Bases upstream of anchor (cannot combine with ``fixed_width``).
        downstream : int, optional
            Bases downstream of anchor (cannot combine with ``fixed_width``).
        anchor : AnchorPoint or str, default "midpoint"
            Anchor point: 'midpoint', 'start', or 'end'.
        bin_size : int, optional
            Aggregate positions into bins of this size.
        bin_agg : ReductionMethod or str, default "mean"
            Aggregation method for binning (coverage only).
        filter_incomplete : bool, default True
            Exclude samples not yet marked complete (coverage only).
        modality : {"coverage", "methylation"}, optional
            Which sub-store to extract from. Defaults to "coverage".
        variable : str, optional
            For methylation: which variable to extract
            (``"methylation_pct"``, ``"n_methylated"``, ``"n_unmethylated"``).
            Defaults to ``"methylation_pct"``.
        samples : list of str, optional
            Subset of sample names (methylation only).

        Returns
        -------
        DataArray
            Dimensions: (interval, relative_position|bin, sample).
        """
        if modality == "methylation":
            if self.methylation is None:
                raise RuntimeError("No methylation store available.")
            return self.methylation.extract(
                intervals_path=intervals_path,
                ranges_df=ranges_df,
                feature_type=feature_type,
                gtf_path=gtf_path,
                variable=variable or "methylation_pct",
                upstream=upstream,
                downstream=downstream,
                fixed_width=fixed_width,
                anchor=anchor if isinstance(anchor, str) else anchor.value,
                bin_size=bin_size if bin_size is not None else 50,
                samples=samples,
            )

        return extract_byranges_signal(
            self._require_coverage(),
            ranges_df=ranges_df,
            intervals_path=intervals_path,
            feature_type=feature_type,
            gtf_path=gtf_path,
            fixed_width=fixed_width,
            upstream=upstream,
            downstream=downstream,
            anchor=anchor,
            bin_size=bin_size,
            bin_agg=bin_agg,
            include_incomplete=not filter_incomplete,
        )

    def count_features(
        self,
        gtf_file: str | Path | None = None,
        bed_file: str | Path | None = None,
        ranges: Any | None = None,
        feature_type: FeatureType | str = FeatureType.GENE,
        feature_id_col: str | list[str] | None = None,
        aggregation: str | None = None,
        strand: str | None = None,
        assays: list[str] | None = None,
        integerize: bool = False,
        fillna_value: float | int | None = 0,
        min_count: int = 1,
        filter_zero: bool = False,
        include_incomplete: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate feature count matrix compatible with DESeq2.

        Parameters
        ----------
        gtf_file : str or Path, optional
            Path to GTF file.
        bed_file : str or Path, optional
            Path to BED file.
        ranges : DataFrame or PyRanges, optional
            Pre-parsed genomic ranges.
        feature_type : FeatureType or str, default "gene"
            GTF feature type to extract.
        feature_id_col : str or list of str, optional
            Column(s) to use as feature identifiers.
        aggregation : str, optional
            Column to aggregate sub-features by.
        assays : list of str, optional
            Which assays to include in output.
        integerize : bool, default False
            Round counts to nearest integer for DESeq2.
        fillna_value : float or int or None, default 0
            Value to fill NaNs before integerization.
        min_count : int, default 1
            Minimum count threshold for mean masking.
        filter_zero : bool, default False
            Remove features with zero counts across all samples.
        include_incomplete : bool, default False
            Include samples not yet marked complete.

        Returns
        -------
        counts : DataFrame
            Count matrix (features × samples).
        features : DataFrame
            Feature metadata.
        """
        return _feature_counts(
            self._require_coverage(),
            gtf_file=gtf_file,
            bed_file=bed_file,
            ranges_df=ranges,
            feature_type=feature_type,
            feature_id_col=feature_id_col,
            aggregate_by=aggregation,
            strand=strand,
            assay=assays[0] if assays else None,
            integerize=integerize,
            fillna_value=fillna_value,
            min_count=min_count,
            filter_zero=filter_zero,
            include_incomplete=include_incomplete,
        )

    def pca(
        self,
        data: xr.DataArray,
        n_components: int = 10,
        nan_handling_strategy: str = "drop",
    ) -> tuple[Any, xr.DataArray]:
        """
        Run PCA on reduced genomic signal data.

        Parameters
        ----------
        data : DataArray
            Input data with dimensions (feature, sample). Typically output
            from ``.reduce()`` (e.g. ``reduced["mean"]``).
        n_components : int, default 10
            Number of principal components.
        nan_handling_strategy : str, default "drop"
            How to handle NaN values: "drop", "set_to_zero", or
            "mean_value_imputation".

        Returns
        -------
        pca_obj : sklearn.decomposition.PCA
        transformed : DataArray
            Dimensions: (sample, component).
        """
        return _run_pca(
            data,
            n_components=n_components,
            nan_handling_strategy=nan_handling_strategy,
        )

    def metaplot(
        self,
        data: xr.DataArray,
        *,
        modality: str | None = None,
        samples: list[str] | None = None,
        groups: dict[str, list[str]] | None = None,
        flip_minus_strand: bool = True,
        error_stat: str | None = "sem",
        palette: str | list | dict | None = None,
        reference_point: float | None = 0,
        reference_label: str = "TSS",
        xlabel: str = "Relative position",
        ylabel: str | None = None,
        title: str = "Metagene profile",
        figsize: tuple[float, float] = (8, 4),
        ax: Any = None,
        filepath: str | Path | None = None,
    ) -> Any:
        """
        Plot a metagene profile from the output of ``.extract()``.

        Parameters
        ----------
        data : DataArray
            Output of ``.extract()`` with dims (interval, bin|relative_position, sample).
        modality : str, optional
            Sets ylabel and colour defaults: "coverage", "methylation", "variant".
        samples : list of str, optional
            Subset of samples to plot (ignored when ``groups`` is set).
        groups : dict {label: [samples]}, optional
            Group samples for averaging with inter-sample error bands.
        flip_minus_strand : bool, default True
            Reverse minus-strand intervals so all profiles run 5'→3'.
        error_stat : {"sem", "std", None}, default "sem"
            Shaded confidence band.
        reference_point : float or None, default 0
            X position of the vertical reference line. None omits it.
        reference_label : str, default "TSS"
            Legend label for the reference line.
        xlabel, ylabel, title : str
            Axis labels and figure title.
        ax : matplotlib Axes, optional
            Axes to draw on; creates a new figure if None.
        filepath : str or Path, optional
            Save figure to this path if provided.

        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        return metaplot(
            data,
            modality=modality,
            samples=samples,
            groups=groups,
            flip_minus_strand=flip_minus_strand,
            error_stat=error_stat,
            palette=palette,
            reference_point=reference_point,
            reference_label=reference_label,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            figsize=figsize,
            ax=ax,
            filepath=filepath,
        )

    def tornadoplot(
        self,
        data: xr.DataArray,
        *,
        modality: str | None = None,
        samples: list[str] | None = None,
        sample_names: list[str] | None = None,
        groups: dict[str, list[str]] | None = None,
        flip_minus_strand: bool = True,
        sort_by: str | None = "mean",
        vmin: float | None = None,
        vmax: float | None = None,
        cmap: str | None = None,
        reference_point: float | None = 0,
        reference_label: str = "TSS",
        xlabel: str = "Relative position",
        ylabel: str | None = None,
        title: str = "Signal heatmap",
        figsize: tuple[float, float] | None = None,
        filepath: str | Path | None = None,
    ) -> list:
        """
        Tornado / heatmap plot from the output of ``.extract()``.

        Parameters
        ----------
        data : DataArray
            Output of ``.extract()``.
        modality : str, optional
            Sets colour defaults: "coverage", "methylation", "variant".
        samples : list of str, optional
            Subset of samples (one panel each). Ignored when ``groups`` is set.
        sample_names : list of str, optional
            Display names for samples.
        groups : dict {label: [samples]}, optional
            Average samples within each group (one panel per group).
        flip_minus_strand : bool, default True
            Reverse minus-strand intervals before plotting.
        sort_by : {"mean", "max", None}, default "mean"
            Sort intervals by signal (descending).
        vmin, vmax : float, optional
            Colour scale limits.
        cmap : str, optional
            Matplotlib colormap.
        reference_point : float or None, default 0
            X position of the vertical reference line. None omits it.
        reference_label : str, default "TSS"
            Label for the reference line.
        xlabel : str, default "Relative position"
        ylabel : str, optional
        title : str, default "Signal heatmap"
        figsize : tuple, optional
        filepath : str or Path, optional

        Returns
        -------
        axes : list of matplotlib.axes.Axes
        """
        return tornadoplot(
            data,
            modality=modality,
            samples=samples,
            sample_names=sample_names,
            groups=groups,
            flip_minus_strand=flip_minus_strand,
            sort_by=sort_by,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            reference_point=reference_point,
            reference_label=reference_label,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            figsize=figsize,
            filepath=filepath,
        )

    def locus_plot(
        self,
        locus: str,
        *,
        sample_names: list[str],
        modality: list[str],
        allele_depth_ref: xr.DataArray | None = None,
        allele_depth_alt: xr.DataArray | None = None,
        genotype: xr.DataArray | None = None,
        coverage: xr.DataArray | None = None,
        methylation: xr.DataArray | None = None,
        palette: str | list | dict | None = None,
        title: str = "Locus plot",
        figsize: tuple[float, float] = (12, 6),
        filepath: str | Path | None = None,
    ) -> list:
        """
        Multi-omics genome-browser-style locus plot.

        Draws one horizontal track per entry in ``sample_names``, stacked
        vertically on a shared genomic x-axis. Coverage and methylation data
        are fetched automatically from the store if not explicitly provided.

        Parameters
        ----------
        locus : str
            Genomic region, e.g. ``"chr21:5200000-5260000"``.
        sample_names : list of str
            Sample name for each track (must exist in the relevant sub-store).
        modality : list of str
            Modality per track — ``"coverage"``, ``"methylation"``, or
            ``"variant"``. Must be the same length as ``sample_names``.
        allele_depth_ref : DataArray, optional
            Pre-computed reference allele depth from
            ``var.extract_region(locus, variable="allele_depth_ref")``.
            Required for any ``"variant"`` track.
        allele_depth_alt : DataArray, optional
            Pre-computed alternate allele depth from
            ``var.extract_region(locus, variable="allele_depth_alt")``.
            Required for any ``"variant"`` track.
        genotype : DataArray, optional
            Pre-computed genotype array from
            ``var.extract_region(locus, variable="genotype")``.
            Optional; used to style variant lollipops by true genotype
            (blue = het, red = hom-alt). If omitted, genotype is approximated
            from allele frequency.
        coverage : DataArray, optional
            Pre-computed coverage region array. If None and coverage tracks
            are requested, fetched automatically from the coverage store.
        methylation : DataArray, optional
            Pre-computed methylation region array. If None and methylation
            tracks are requested, fetched automatically from the methylation store.
        palette : str, list, or dict, optional
            Colour palette. A dict maps sample names to colours.
        title : str, default "Locus plot"
            Figure title.
        figsize : tuple, default (12, 6)
            Figure size ``(width, height)`` in inches.
        filepath : str or Path, optional
            Save figure to this path if provided.

        Returns
        -------
        axes : list of matplotlib.axes.Axes

        Examples
        --------
        >>> adr = ds.variants.extract_region(locus, variable="allele_depth_ref").compute()
        >>> ada = ds.variants.extract_region(locus, variable="allele_depth_alt").compute()
        >>> ds.locus_plot(
        ...     locus="chr21:5200000-5260000",
        ...     sample_names=["atac", "chip", "meth-rep1", "snp"],
        ...     modality=["coverage", "coverage", "methylation", "variant"],
        ...     allele_depth_ref=adr,
        ...     allele_depth_alt=ada,
        ...     title="Multi-omics locus",
        ... )
        """
        # Auto-fetch coverage if needed and not already provided
        if coverage is None and any(m == "coverage" for m in modality):
            coverage = self._require_coverage().extract_region(locus)

        # Auto-fetch methylation if needed and not already provided
        if methylation is None and any(m == "methylation" for m in modality):
            if self.methylation is None:
                raise RuntimeError("No methylation store available for auto-fetch.")
            methylation = self.methylation.extract_region(
                locus, variable="methylation_pct"
            )

        return locus_plot(
            locus,
            sample_names=sample_names,
            modality=modality,
            coverage=coverage,
            methylation=methylation,
            allele_depth_ref=allele_depth_ref,
            allele_depth_alt=allele_depth_alt,
            genotype=genotype,
            palette=palette,
            title=title,
            figsize=figsize,
            filepath=filepath,
        )

    # ========== Metadata ==========

    def set_metadata(
        self,
        metadata: pd.DataFrame,
        sample_column: str = "sample_id",
        merge: bool = True,
    ) -> None:
        """
        Attach metadata to the coverage store.

        Parameters
        ----------
        metadata : DataFrame
        sample_column : str, default "sample_id"
        merge : bool, default True
            If True, merge with existing metadata. If False, replace entirely.
        """
        self._require_coverage().set_metadata(metadata, sample_column=sample_column, merge=merge)

    def update_metadata(self, updates: dict) -> None:
        """
        Update metadata columns using a dictionary.

        Parameters
        ----------
        updates : dict
            Column name → list (aligned with sample order) or dict {sample: value}.
        """
        self._require_coverage().update_metadata(updates)

    def list_metadata_columns(self) -> list[str]:
        """List current metadata column names in the coverage store."""
        return self._require_coverage().list_metadata_columns()

    def remove_metadata_columns(self, columns: list[str]) -> None:
        """Remove metadata columns from the coverage store."""
        self._require_coverage().remove_metadata_columns(columns)

    def metadata_to_csv(self, path: str | Path) -> None:
        """Export coverage store metadata to CSV."""
        self._require_coverage().metadata_to_csv(path)

    def metadata_to_json(self, path: str | Path) -> None:
        """Export coverage store metadata to JSON."""
        self._require_coverage().metadata_to_json(path)

    @staticmethod
    def metadata_from_csv(path: str | Path, **kwargs) -> pd.DataFrame:
        """Load a metadata DataFrame from CSV."""
        return BamStore.metadata_from_csv(path, **kwargs)

    @staticmethod
    def metadata_from_json(path: str | Path) -> pd.DataFrame:
        """Load a metadata DataFrame from JSON."""
        return BamStore.metadata_from_json(path)

    # ========== Repr ==========

    def __repr__(self) -> str:
        if self._multiomics:
            return repr(self._multiomics)
        if self._bam:
            return f"QuantNado(coverage={self._bam.store_path})"
        return "QuantNado(empty)"


__all__ = [
    "QuantNado",
    "MethylStore",
    "VariantStore",
    "MultiomicsStore",
    "metaplot",
    "tornadoplot",
    "locus_plot",
]

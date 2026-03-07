"""
Unified QuantNado API for genomic signal analysis.

This module provides a high-level `QuantNado` facade that unifies dataset creation,
loading, signal reduction, feature counting, and PCA analysis into a single,
convenient interface.

Example:
    # Create a new dataset from BAM files
    qn = QuantNado.from_bam_files(
        bam_files=["sample1.bam", "sample2.bam"],
        store_path="dataset.zarr",
    )

    # Open existing dataset
    qn = QuantNado.open("dataset.zarr")

    # Reduce signal over genomic ranges (e.g., promoters)
    reduced_ds = qn.reduce("promoters.bed", reduction="mean")

    # Compute feature counts for DESeq2
    counts_df, feature_df = qn.count_features("genes.gtf", feature_type="gene")

    # Run PCA on reduced data
    pca_obj, transformed = qn.pca(reduced_ds["mean"], n_components=10)

    # Extract specific region
    region_data = qn.extract_region("chr1:1000-5000")

    # Access metadata and sample information
    print(qn.samples)
    print(qn.metadata)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import xarray as xr
import pandas as pd
from typing import Iterable

from quantnado.dataset.bam import BamStore
from quantnado.dataset.reduce import reduce_byranges_signal, extract_byranges_signal
from quantnado.dataset.counts import count_features as _feature_counts
from quantnado.dataset.pca import run_pca as _run_pca
from quantnado.dataset.plot import metaplot, tornadoplot
from quantnado.dataset.enums import FeatureType, ReductionMethod, AnchorPoint


class QuantNado:
    """
    Unified facade for QuantNado genomic analysis.

    Wraps a `BamStore` instance and provides convenient methods for common
    analysis workflows (signal reduction, feature counting, PCA). All methods
    execute eagerly and return their natural result types.

    Attributes
    ----------
    store : BamStore
        The underlying Zarr-backed BAM signal store.

    Examples
    --------
    >>> # Create from BAM files
    >>> qn = QuantNado.from_bam_files(
    ...     bam_files=["s1.bam", "s2.bam"],
    ...     store_path="dataset.zarr"
    ... )

    >>> # Open existing store
    >>> qn = QuantNado.open("dataset.zarr")

    >>> # Reduce signal and run analysis
    >>> reduced = qn.reduce("promoters.bed", reduction="mean")
    >>> counts, features = qn.count_features("genes.gtf")
    >>> pca, transformed = qn.pca(reduced["mean"])
    """

    def __init__(self, store: BamStore) -> None:
        """
        Initialize QuantNado with a BamStore instance.

        Parameters
        ----------
        store : BamStore
            The underlying Zarr-backed store.
        """
        self.store = store

    @classmethod
    def open(cls, store_path: str | Path, read_only: bool = True) -> "QuantNado":
        """
        Open an existing QuantNado dataset.

        Parameters
        ----------
        store_path : str or Path
            Path to the Zarr store directory.
        read_only : bool, default True
            If True, disables write operations.

        Returns
        -------
        QuantNado
            Instance wrapping the opened store.

        Raises
        ------
        FileNotFoundError
            If the store does not exist.
        ValueError
            If the store is missing required metadata.
        """
        store = BamStore.open(store_path, read_only=read_only)
        return cls(store)

    @classmethod
    def from_bam_files(
        cls,
        bam_files: list[str],
        store_path: str | Path,
        chromsizes: str | Path | dict[str, int] | None = None,
        metadata: pd.DataFrame | Path | str | list[Path | str] | None = None,
        *,
        filter_chromosomes: bool = True,
        overwrite: bool = True,
        resume: bool = False,
        sample_column: str = "sample_id",
        chunk_len: int | None = None,
        construction_compression: str = "default",
        local_staging: bool = False,
        staging_dir: str | Path | None = None,
        log_file: Path | None = None,
        test: bool = False,
    ) -> "QuantNado":
        """
        Create a new QuantNado dataset from BAM files.

        Parameters
        ----------
        bam_files : list of str
            Paths to BAM files to process.
        store_path : str or Path
            Output path for the Zarr store.
        chromsizes : str, Path, or dict, optional
            Chromosome sizes. If None, extracted from the first BAM file.
            Can be a .chrom.sizes file or a dict mapping chromosome names to sizes.
        metadata : DataFrame, Path, or list of Paths, optional
            Sample metadata. Can be a DataFrame, path to CSV, or list of CSV paths.
        filter_chromosomes : bool, default True
            If True, keep only canonical chromosomes (chr1-22, chrX, chrY, chrM).
        overwrite : bool, default True
            If True, overwrite existing store at the same path.
        resume : bool, default False
            If True, resume processing an existing store.
        sample_column : str, default "sample_id"
            Column name in metadata DataFrame that matches BAM file stems.
        chunk_len : int, optional
            Zarr chunk size for the position dimension. If omitted, dataset
            construction derives a filesystem-aware default from the target
            store path. Older stores without a persisted value still fall back
            to ``DEFAULT_CHUNK_LEN``.
        construction_compression : {"default", "fast", "none"}, default "default"
            Build-time compression profile. Use ``fast`` for lower zstd
            compression overhead or ``none`` for uncompressed construction
            arrays when benchmarking filesystem write throughput.
        local_staging : bool, default False
            If True, build the dataset under local scratch storage and publish
            it to ``store_path`` after construction completes.
        staging_dir : str or Path, optional
            Scratch directory to use for local staging. If omitted while
            ``local_staging`` is enabled, QuantNado uses ``TMPDIR`` or the
            system temporary directory.
        log_file : Path, optional
            Path to write processing logs.
        test : bool, default False
            If True, process only test chromosomes (chr21, chr22, chrY).

        Returns
        -------
        QuantNado
            Instance wrapping the newly created store.
        """
        store = BamStore.from_bam_files(
            bam_files=bam_files,
            chromsizes=chromsizes,
            store_path=store_path,
            metadata=metadata,
            filter_chromosomes=filter_chromosomes,
            overwrite=overwrite,
            resume=resume,
            sample_column=sample_column,
            chunk_len=chunk_len,
            construction_compression=construction_compression,
            local_staging=local_staging,
            staging_dir=staging_dir,
            log_file=log_file,
            test=test,
        )
        return cls(store)

    # ========== Passthrough Properties ==========

    @property
    def store_path(self) -> Path:
        """Path to the Zarr store directory."""
        return self.store.store_path

    @property
    def samples(self) -> list[str]:
        """List of sample names in the store."""
        return self.store.sample_names

    @property
    def chromosomes(self) -> list[str]:
        """List of chromosome names in the store."""
        return self.store.chromosomes

    @property
    def chromsizes(self) -> dict[str, int]:
        """Mapping of chromosome names to sizes."""
        return self.store.chromsizes

    @property
    def metadata(self) -> pd.DataFrame:
        """Metadata DataFrame indexed by sample name."""
        return self.store.get_metadata()

    # ========== Data Access Methods ==========

    def to_xarray(
        self,
        chromosomes: list[str] | None = None,
        chunks: str | dict | None = None,
    ) -> dict[str, xr.DataArray]:
        """
        Extract the dataset as per-chromosome Xarray DataArrays with lazy evaluation.

        All samples must be marked complete.

        Parameters
        ----------
        chromosomes : list of str, optional
            Chromosomes to extract. If None, extracts all.
        chunks : str or dict, optional
            Dask chunking strategy. Default matches store chunk size.

        Returns
        -------
        dict[str, DataArray]
            Dictionary mapping chromosome names to DataArrays with dimensions
            (sample, position) and metadata coordinates.

        Raises
        ------
        RuntimeError
            If any sample is incomplete.
        """
        return self.store.to_xarray(chromosomes=chromosomes, chunks=chunks)

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
        Extract signal data for a specific genomic region.

        Parameters
        ----------
        region : str, optional
            Genomic region in format "chr:start-end" (e.g., "chr1:1000-5000").
            Alternative to specifying chrom/start/end separately.
        chrom : str, optional
            Chromosome name.
        start : int, optional
            Start position (0-based, inclusive). Defaults to 0 if None.
        end : int, optional
            End position (0-based, exclusive). Defaults to chromosome end if None.
        samples : list of str or int, optional
            Sample names or indices. If None, uses all completed samples.
        as_xarray : bool, default True
            If True, return DataArray with lazy dask array.
            If False, return computed numpy array.

        Returns
        -------
        DataArray or ndarray
            Extracted region with dimensions (sample, position).

        Examples
        --------
        >>> qn = QuantNado.open("store.zarr")
        >>> region = qn.extract_region("chr1:1000-5000")
        >>> region.shape
        (n_samples, 4000)
        """
        return self.store.extract_region(
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
        Reduce per-sample signal over genomic ranges (e.g., genes, promoters, exons).

        This method aggregates signal across genomic regions, useful for
        feature-level analysis (DESeq2 counts, PCA, etc.).

        Supports three input modes for specifying ranges:
        1. `intervals_path`: Path to BED/GTF file
        2. `ranges_df`: Direct DataFrame or PyRanges with genomic ranges
        3. `feature_type + gtf_path`: Predefined feature extraction from GTF

        Parameters
        ----------
        intervals_path : str or Path, optional
            Path to intervals file (BED or GTF) with genomic ranges.
        ranges_df : DataFrame or PyRanges, optional
            Pandas DataFrame or PyRanges with genomic ranges.
            Required columns: [Chromosome/contig, Start/start, End/end]
        feature_type : FeatureType or str, optional
            Predefined feature type to extract from GTF.
            Options: 'gene', 'transcript', 'exon', 'promoter'.
            Requires `gtf_path` to be set.
        gtf_path : str, Path, or Iterable[str|Path], optional
            Path(s) to GTF file(s) for feature extraction.
        reduction : ReductionMethod or str, default ReductionMethod.MEAN
            Aggregation statistic. Options: 'mean', 'sum', 'max', 'min', 'median'.
        filter_incomplete : bool, default True
            If True, exclude samples not marked complete in metadata.

        Returns
        -------
        Dataset
            Xarray Dataset with variables for each reduction type:
            - 'mean', 'sum', 'count', and the specified reduction type
            Dimensions: (ranges, sample)
            Coordinates: range_index, start, end, range_length, contig, sample names.

        Examples
        --------
        >>> qn = QuantNado.open("store.zarr")
        
        >>> # From BED file
        >>> reduced = qn.reduce(intervals_path="regions.bed", reduction="mean")
        
        >>> # From DataFrame
        >>> import pandas as pd
        >>> ranges = pd.DataFrame({
        ...     "Chromosome": ["chr1", "chr1"],
        ...     "Start": [1000, 5000],
        ...     "End": [2000, 6000]
        ... })
        >>> reduced = qn.reduce(ranges_df=ranges)
        
        >>> # From predefined GTF features
        >>> reduced = qn.reduce(feature_type="promoter", gtf_path="genes.gtf")
        >>> reduced["mean"].shape
        (n_regions, n_samples)
        
        >>> # Use reduced data for PCA
        >>> pca, transformed = qn.pca(reduced["mean"])
        """
        return reduce_byranges_signal(
            self.store,
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
    ) -> xr.DataArray:
        """
        Extract raw per-position signal over genomic ranges.

        Unlike reduce(), this returns the full signal vector for each interval,
        optionally resized to a fixed window and binned into coarser resolution.

        Window specification (pick one):
        - ``upstream`` / ``downstream``: asymmetric window around the anchor point,
          like deeptools ``-b`` / ``-a``. Position coordinate is bp offset from anchor
          (negative = upstream, positive = downstream).
        - ``fixed_width``: symmetric window of this total width centered on anchor.

        Supports three input modes for specifying ranges:
        1. `intervals_path`: Path to BED/GTF file
        2. `ranges_df`: Direct DataFrame or PyRanges with genomic ranges
        3. `feature_type + gtf_path`: Predefined feature extraction from GTF

        Parameters
        ----------
        intervals_path : str or Path, optional
            Path to intervals file (BED or GTF) with genomic ranges.
        ranges_df : DataFrame or PyRanges, optional
            Pandas DataFrame or PyRanges with genomic ranges.
            Required columns: [Chromosome/contig, Start/start, End/end]
        feature_type : FeatureType or str, optional
            Predefined feature type to extract from GTF.
            Options: 'gene', 'transcript', 'exon', 'promoter'.
            Requires `gtf_path` to be set.
        gtf_path : str, Path, or Iterable[str|Path], optional
            Path(s) to GTF file(s) for feature extraction.
        fixed_width : int, optional
            Symmetric window: total bp around anchor (upstream = downstream = width//2).
            Cannot be combined with upstream/downstream.
        upstream : int, optional
            Bases upstream of the anchor to include (e.g. 2000 for 2 kb upstream).
            Cannot be combined with fixed_width.
        downstream : int, optional
            Bases downstream of the anchor to include (e.g. 2000 for 2 kb downstream).
            Cannot be combined with fixed_width.
        anchor : AnchorPoint or str, default "midpoint"
            Anchor point: 'midpoint', 'start' (5' end, strand-aware), or
            'end' (3' end, strand-aware).
        bin_size : int, optional
            If set, aggregate positions into bins of this size (e.g., 50 bp).
            Total window must be divisible by bin_size.
        bin_agg : ReductionMethod or str, default "mean"
            Aggregation method for binning: 'mean', 'sum', 'max', 'min', 'median'.
        filter_incomplete : bool, default True
            If True, exclude samples not marked complete in metadata.

        Returns
        -------
        DataArray
            Xarray DataArray with dimensions (interval, relative_position|bin, sample).
            When upstream/downstream or fixed_width is set, the position coordinate
            contains actual bp offsets from the anchor (negative = upstream).
            Dask-backed for lazy evaluation; call `.compute()` to materialize.

        Examples
        --------
        >>> qn = QuantNado.open("store.zarr")

        >>> # 2 kb upstream, 2 kb downstream of TSS, binned at 50 bp (deeptools style)
        >>> binned = qn.extract(
        ...     feature_type="transcript",
        ...     gtf_path="genes.gtf",
        ...     upstream=2000,
        ...     downstream=2000,
        ...     anchor="start",
        ...     bin_size=50,
        ... )
        >>> binned.coords["bin"].values  # [-2000, -1950, ..., 1950]

        >>> # Asymmetric: 1 kb upstream, 3 kb downstream
        >>> binned = qn.extract(
        ...     feature_type="transcript",
        ...     gtf_path="genes.gtf",
        ...     upstream=1000,
        ...     downstream=3000,
        ...     anchor="start",
        ...     bin_size=100,
        ... )

        >>> # Symmetric (old style)
        >>> qn.extract(feature_type="promoter", gtf_path="genes.gtf", fixed_width=2000)
        """
        return extract_byranges_signal(
            self.store,
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
            Path to GTF file for feature extraction.
        bed_file : str or Path, optional
            Path to BED file with genomic ranges.
        ranges : DataFrame or PyRanges, optional
            Pre-parsed genomic ranges (DataFrame or PyRanges).
        feature_type : FeatureType or str, default FeatureType.GENE
            GTF feature type to extract (e.g., 'gene', 'transcript', 'exon').
        feature_id_col : str, optional
            Column to use as feature identifiers and index the counts matrix.
            For GTF inputs defaults to the first available of: gene_id, transcript_id,
            gene_name, transcript_name.
        aggregation : str, optional
            Column to aggregate sub-features by (e.g., 'gene_id' to sum exons to genes).
        assays : list of str, optional
            Which assays to include in output (e.g., ['sum']).
        integerize : bool, default False
            If True, round counts to nearest integer for DESeq2.
        fillna_value : float or int or None, default 0
            Value to fill NaNs before integerization. None skips filling.
        min_count : int, default 1
            Minimum count threshold for mean masking.
        filter_zero : bool, default False
            If True, remove features with zero counts across all samples.
        include_incomplete : bool, default False
            If True, include samples not yet marked complete.

        Returns
        -------
        counts : DataFrame
            Count matrix (features x samples).
        features : DataFrame
            Feature metadata (name, length, etc.).

        Examples
        --------
        >>> qn = QuantNado.open("store.zarr")
        >>> counts, features = qn.count_features("genes.gtf", feature_type="gene")
        >>> counts.shape
        (n_genes, n_samples)
        """
        return _feature_counts(
            self.store,
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

    def metaplot(
        self,
        data: xr.DataArray,
        *,
        modality: "str | None" = None,
        samples: list[str] | None = None,
        groups: "dict[str, list[str]] | None" = None,
        flip_minus_strand: bool = True,
        error_stat: "str | None" = "sem",
        palette: "str | list | dict | None" = None,
        reference_point: "float | None" = 0,
        reference_label: str = "TSS",
        xlabel: str = "Relative position",
        ylabel: "str | None" = None,
        title: str = "Metagene profile",
        figsize: "tuple[float, float]" = (8, 4),
        ax: Any = None,
        filepath: "str | Path | None" = None,
    ) -> Any:
        """
        Plot a metagene profile from the output of ``qn.extract()``.

        Thin wrapper around :func:`quantnado.dataset.plot.metaplot`.
        See that function for full parameter documentation.

        Parameters
        ----------
        data : DataArray
            Output of ``qn.extract()`` — dimensions
            ``(interval, relative_position|bin, sample)``.
        samples : list of str, optional
            Subset of samples to plot (ignored when ``groups`` is provided).
        groups : dict {label: [samples]}, optional
            Group samples for averaging. One line per group with inter-sample
            error bands. Example: ``{"ctrl": ["s1","s2"], "treat": ["s3","s4"]}``
        flip_minus_strand : bool, default True
            Reverse minus-strand intervals before averaging so all profiles
            run 5'→3'.
        error_stat : {"sem", "std", None}, default "sem"
            Shaded confidence band around each line.
            ``"sem"`` matches deeptools plotProfile default.
        reference_point : float or None, default 0
            X position of the vertical reference line (e.g. TSS). ``None`` omits it.
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
        modality: "str | None" = None,
        samples: "list[str] | None" = None,
        sample_names: "list[str] | None" = None,
        groups: "dict[str, list[str]] | None" = None,
        flip_minus_strand: bool = True,
        sort_by: "str | None" = "mean",
        vmin: "float | None" = None,
        vmax: "float | None" = None,
        cmap: "str | None" = None,
        reference_point: "float | None" = 0,
        reference_label: str = "TSS",
        xlabel: str = "Relative position",
        ylabel: "str | None" = None,
        title: str = "Signal heatmap",
        figsize: "tuple[float, float] | None" = None,
        filepath: "str | Path | None" = None,
    ) -> list:
        """
        Tornado / heatmap plot from the output of ``qn.extract()``.

        One panel per sample (or group). Rows = intervals, colour = signal.

        Parameters
        ----------
        data : DataArray
            Output of ``qn.extract()``.
        samples : list of str, optional
            Subset of samples (one panel each). Ignored when ``groups`` is set.
        sample_names : list of str, optional
            Display names for samples. Must match length of ``samples``.
        groups : dict {label: [samples]}, optional
            Average samples within each group (one panel per group).
        flip_minus_strand : bool, default True
            Reverse minus-strand intervals before plotting.
        sort_by : {"mean", "max", None}, default "mean"
            Sort intervals by signal (descending) using the first panel.
        vmin, vmax : float, optional
            Colour scale limits. Defaults: 0 and 99th percentile.
        cmap : str, default "RdYlBu_r"
            Matplotlib colormap.
        reference_point : float or None, default 0
            X position of the vertical reference line. ``None`` omits it.
        reference_label : str, default "TSS"
            Label for the reference line.
        xlabel : str, default "Relative position"
            X-axis label.
        ylabel : str, optional
            Y-axis label. Defaults to "Intervals (n=<count>)".
        title : str, default "Signal heatmap"
            Figure suptitle.
        figsize : tuple, optional
            Figure size. Auto-computed if None.
        filepath : str or Path, optional
            Save figure to this path if provided.

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

    def pca(
        self,
        data: xr.DataArray,
        n_components: int = 10,
        nan_handling_strategy: str = "drop",
    ) -> tuple[Any, xr.DataArray]:
        """
        Run PCA on reduced genomic signal data.

        This method performs dimensionality reduction on feature-level signal
        (typically from `.reduce()`) for visualization and quality control.

        Parameters
        ----------
        data : DataArray
            Input data with dimensions (feature, sample). Typically output
            from `.reduce()` method (e.g., `reduced["mean"]`).
        n_components : int, default 10
            Number of principal components to compute.
        nan_handling_strategy : str, default "drop"
            How to handle NaN values. Options:
            - "drop": Remove features with any NaN
            - "set_to_zero": Replace NaN with 0
            - "mean_value_imputation": Replace NaN with feature mean

        Returns
        -------
        pca_obj : sklearn.decomposition.PCA
            Fitted PCA object.
        transformed : DataArray
            Transformed data with dimensions (sample, component).
            Coordinates include sample names and component indices.

        Examples
        --------
        >>> qn = QuantNado.open("store.zarr")
        >>> reduced = qn.reduce("promoters.bed", reduction="mean")
        >>> pca, transformed = qn.pca(reduced["mean"], n_components=10)
        >>> transformed.shape
        (n_samples, 10)
        """
        return _run_pca(
            data,
            n_components=n_components,
            nan_handling_strategy=nan_handling_strategy,
        )

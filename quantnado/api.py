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
    counts_df, feature_df = qn.feature_counts("genes.gtf", feature_type="gene")

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
from quantnado.dataset.counts import feature_counts as _feature_counts
from quantnado.dataset.pca import run_pca as _run_pca
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
    >>> counts, features = qn.feature_counts("genes.gtf")
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
        chunk_len: int = 65536,
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
        chunk_len : int, default 65536
            Zarr chunk size for position dimension.
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
        anchor: AnchorPoint | str = AnchorPoint.MIDPOINT,
        bin_size: int | None = None,
        bin_agg: ReductionMethod | str = ReductionMethod.MEAN,
        filter_incomplete: bool = True,
    ) -> xr.DataArray:
        """
        Extract raw per-position signal over genomic ranges.

        Unlike reduce(), this returns the full signal vector for each interval,
        optionally resized to fixed_width and binned into coarser resolution.

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
            If set, all intervals are resized to this width (centered on anchor).
            Shorter intervals are padded with NaN; longer ones are trimmed.
            Must be divisible by bin_size if bin_size is also set.
        anchor : AnchorPoint or str, default "midpoint"
            Anchor point for fixed_width: 'midpoint', 'start', or 'end'.
            'start' and 'end' are strand-aware (5'/3') when Strand column is present.
        bin_size : int, optional
            If set, aggregate positions into bins of this size (e.g., 50 bp).
            Must divide fixed_width evenly if fixed_width is also set.
        bin_agg : ReductionMethod or str, default "mean"
            Aggregation method for binning: 'mean', 'sum', 'max', 'min', 'median'.
        filter_incomplete : bool, default True
            If True, exclude samples not marked complete in metadata.

        Returns
        -------
        DataArray
            Xarray DataArray with dimensions (interval, relative_position, sample).
            - interval: unique ID for each genomic region
            - relative_position: 0-based offset within interval (or bin index if bin_size set)
            - sample: sample name
            
            Coordinates include interval metadata:
            - start, end: genomic coordinates
            - contig: chromosome name
            - strand: DNA strand (if present in input)
            
            Dask-backed for lazy evaluation; call `.compute()` to materialize.

        Raises
        ------
        ValueError
            If fixed_width is not divisible by bin_size.
        TypeError
            If neither ranges_df nor intervals_path nor (feature_type, gtf_path) are provided.

        Examples
        --------
        >>> qn = QuantNado.open("store.zarr")
        
        >>> # Extract raw signal over promoters, fixed to 2 kb
        >>> promoter_signal = qn.extract(
        ...     feature_type="promoter",
        ...     gtf_path="genes.gtf",
        ...     fixed_width=2000
        ... )
        >>> promoter_signal.shape
        (n_promoters, 2000, n_samples)
        
        >>> # Bin into 50 bp windows
        >>> binned = qn.extract(
        ...     feature_type="promoter",
        ...     gtf_path="genes.gtf",
        ...     fixed_width=2000,
        ...     bin_size=50,
        ...     bin_agg="mean"
        ... )
        >>> binned.shape
        (n_promoters, 40, n_samples)
        
        >>> # Extract from BED file without fixed_width (variable length)
        >>> peaks_signal = qn.extract(intervals_path="peaks.bed")
        >>> peaks_signal.shape
        (n_peaks, max_peak_length, n_samples)  # NaN-padded to max length
        
        >>> # Materialize and plot
        >>> data = binned.compute()
        >>> import matplotlib.pyplot as plt
        >>> plt.imshow(data[0, :, :].T, aspect='auto')  # 1st region, all samples
        """
        return extract_byranges_signal(
            self.store,
            ranges_df=ranges_df,
            intervals_path=intervals_path,
            feature_type=feature_type,
            gtf_path=gtf_path,
            fixed_width=fixed_width,
            anchor=anchor,
            bin_size=bin_size,
            bin_agg=bin_agg,
            include_incomplete=not filter_incomplete,
        )

    def feature_counts(
        self,
        gtf_file: str | Path | None = None,
        bed_file: str | Path | None = None,
        ranges: Any | None = None,
        feature_type: FeatureType | str = FeatureType.GENE,
        aggregation: str | None = None,
        assays: list[str] | None = None,
        integerize: bool = False,
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
        aggregation : str, optional
            How to aggregate sub-features (e.g., 'sum' to merge exons into genes).
        assays : list of str, optional
            Which assays to include in output (e.g., ['sum']).
        integerize : bool, default False
            If True, round counts to nearest integer for DESeq2.

        Returns
        -------
        counts : DataFrame
            Count matrix (features x samples).
        features : DataFrame
            Feature metadata (name, length, etc.).

        Examples
        --------
        >>> qn = QuantNado.open("store.zarr")
        >>> counts, features = qn.feature_counts("genes.gtf", feature_type="gene")
        >>> counts.shape
        (n_genes, n_samples)
        """
        return _feature_counts(
            self.store,
            gtf_file=gtf_file,
            bed_file=bed_file,
            ranges_df=ranges,
            feature_type=feature_type,
            aggregate_by=aggregation,
            assay=assays[0] if assays else None,
            integerize=integerize,
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

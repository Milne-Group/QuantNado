from __future__ import annotations

import numpy as np
import pandas as pd

from ..dataset.enums import FeatureType
from .features import extract_feature_ranges, load_gtf
from .reduce import reduce_byranges_signal


def count_features(
    dataset,
    *,
    ranges_df=None,
    bed_file: str | None = None,
    gtf_df: pd.DataFrame | None = None,
    gtf_file: str | None = None,
    feature_type: str = "gene",
    start_col: str = "start",
    end_col: str = "end",
    contig_col: str | None = None,
    feature_id_col: str | list[str] | None = None,
    aggregate_by: str | None = None,
    strand: str | int | None = None,
    assay: str | None = None,
    samples: list[str] | None = None,
    filter_chromosomes: bool = True,
    integerize: bool = True,
    fillna_value: float | int | None = 0,
    min_count: int = 1,
    filter_zero: bool = False,
    include_incomplete: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute per-feature summed counts suitable for DESeq2.

    Parameters
    ----------
    dataset : BamStore
        Source dataset providing per-chromosome signal arrays. Sample names and
        completion status are read from the store.
    ranges_df : pandas.DataFrame, optional
        Table containing 0-based end-exclusive ranges. Highest priority if provided.
    bed_file : str, optional
        Path to a BED file providing feature ranges (first 3 columns used).
    gtf_df : pandas.DataFrame, optional
        GTF dataframe (from load_gtf). Used if ranges_df/bed_file are not provided.
    gtf_file : str, optional
        Path to a GTF file; loaded if provided and ranges_df/bed_file not set.
    feature_type : str
        GTF feature level to summarize (e.g., "gene" or "transcript") when using gtf_df/gtf_file.
    start_col, end_col : str
        Column names for range boundaries within ranges_df.
    contig_col : str, optional
        Column name for contig in ranges_df. If provided and present, included in metadata.
    feature_id_col : str, optional
        Column in ranges_df to use as feature identifiers and index the counts matrix. If omitted
        for GTF inputs, falls back to common IDs (gene_id, transcript_id, gene_name, transcript_name).
    aggregate_by : str, optional
        Optional column to aggregate counts/metadata by (e.g., gene_id when feature_type="exon").
        When set, counts are summed over that key and metadata are aggregated (start min, end max,
        length summed) grouped by the key.
    strand : str or int, optional
        Feature strand filtering or read counting mode:

        - ``"+"``, ``"-"``: filter features by strand annotation (requires a
          ``strand`` column in the ranges table; unlisted features are excluded).
        - ``0``: unstranded — sum all reads regardless of strand (default).
        - ``1`` (fr-secondstrand): per-feature strand-aware counting.
          ``+`` features use ``{chrom}_fwd``; ``-`` features use ``{chrom}_rev``.
        - ``2`` (fr-firststrand / dUTP): per-feature strand-aware counting.
          ``+`` features use ``{chrom}_rev``; ``-`` features use ``{chrom}_fwd``.

        Int modes require the BamStore to have been built with ``stranded`` set
        and a ``strand`` column in the ranges table.
    min_count : int
        Minimum count threshold passed to reduction (affects mean masking only; sums unaffected).
    integerize : bool
        If True, round sums to int64 for DESeq2 compatibility.
    assay : str, optional
        If set, limit columns to samples matching this assay using attrs['assay_by_sample'] when available.
    samples : list of str, optional
        If set, count only these specific sample names (e.g., ["RNA-SEM-1", "RNA-SEM-2"]).
        If both assay and samples are provided, samples takes precedence.
    filter_chromosomes : bool, default True
        Keep only canonical chromosomes (``chr*`` without underscores).
    integerize : bool
        If True, round sums to int64 for DESeq2 compatibility.
    filter_zero : bool
        If True, remove features with zero counts across all samples after filling/aggregation.
    Returns
    -------
    counts_df : pandas.DataFrame
        Features x samples matrix of summed counts (rounded to integers if requested).
    feature_metadata : pandas.DataFrame
        Metadata for each feature (start/end/length and contig if available), aligned to counts rows.
    """
    # Resolve ranges priority: explicit df > BED > GTF
    resolved_ranges = ranges_df
    resolved_contig_col = contig_col
    resolved_feature_id_col = feature_id_col
    resolved_aggregate_by = aggregate_by

    if resolved_ranges is None and bed_file is None:
        if gtf_df is None and gtf_file is None:
            raise TypeError("Provide ranges_df, bed_file, or gtf_df/gtf_file")
        gtf_source = (
            gtf_df if gtf_df is not None else load_gtf(gtf_file, feature_types=None)
        )
        
        # Extract feature ranges and convert from PyRanges to DataFrame
        feature_ranges_pr = extract_feature_ranges(gtf_source, feature_type=feature_type)
        resolved_ranges = pd.DataFrame(feature_ranges_pr)
        
        # Normalize column names from PyRanges convention to internal convention
        resolved_ranges = resolved_ranges.rename(columns={
            "Chromosome": "contig",
            "Start": "start",
            "End": "end",
        })
        
        resolved_contig_col = "contig"
        start_col = "start"
        end_col = "end"
        
        if resolved_feature_id_col is None:
            for candidate in (
                "gene_id",
                "transcript_id",
                "gene_name",
                "transcript_name",
            ):
                if candidate in resolved_ranges.columns:
                    resolved_feature_id_col = candidate
                    break
        # Default aggregation for exon-level: sum to gene_id when available.
        if (
            resolved_aggregate_by is None
            and feature_type == "exon"
            and "gene_id" in resolved_ranges.columns
        ):
            resolved_aggregate_by = "gene_id"

    # Normalize PyRanges Strand column name across all input sources.
    if resolved_ranges is not None and "Strand" in resolved_ranges.columns and "strand" not in resolved_ranges.columns:
        resolved_ranges = resolved_ranges.rename(columns={"Strand": "strand"})

    # Apply chromosome filtering if requested
    if filter_chromosomes and resolved_ranges is not None:
        contig_name = resolved_contig_col or "contig"
        if contig_name in resolved_ranges.columns:
            resolved_ranges = resolved_ranges[
                resolved_ranges[contig_name].str.startswith("chr") & 
                ~resolved_ranges[contig_name].str.contains("_")
            ].reset_index(drop=True)

    # Apply strand filtering or resolve strand_mode for read counting
    _strand_mode = 0
    if isinstance(strand, int):
        _strand_mode = strand
    elif strand in ("+", "-") and resolved_ranges is not None and "strand" in resolved_ranges.columns:
        resolved_ranges = resolved_ranges[
            resolved_ranges["strand"] == strand
        ].reset_index(drop=True)

    # Convert sample names to indices if provided
    sample_indices = None
    if samples is not None:
        all_samples = (
            getattr(dataset, "samples", None) or 
            getattr(dataset, "sample_names", None)
        )
        if all_samples is None:
            raise ValueError("Dataset does not expose sample names; cannot filter by samples")
        sample_indices = np.array(
            [i for i, s in enumerate(all_samples) if s in samples],
            dtype=int
        )
        if len(sample_indices) == 0:
            raise ValueError(f"No samples matched. Available: {all_samples}")


    reduced = reduce_byranges_signal(
        dataset,
        ranges_df=resolved_ranges,
        intervals_path=bed_file,
        start_col=start_col,
        end_col=end_col,
        contig_col=resolved_contig_col or "contig",
        min_count=min_count,
        reduction="mean",
        sample_indices=sample_indices,
        include_incomplete=include_incomplete,
        strand_mode=_strand_mode,
    )

    aligned_ranges = resolved_ranges
    if resolved_ranges is not None and "range_index" in reduced.coords:
        idx = reduced["range_index"].values
        aligned_ranges = resolved_ranges.loc[idx]

    counts_da = reduced["sum"].transpose("ranges", "sample")

    sample_labels = [str(s) for s in counts_da["sample"].values]
    counts_df = counts_da.to_pandas()
    counts_df.columns = sample_labels

    if assay is not None:
        assay_map = None
        for source in (dataset, reduced):
            if source is None:
                continue
            maybe = getattr(source, "attrs", {}).get("assay_by_sample") if hasattr(source, "attrs") else None
            if maybe is not None and len(maybe) == len(sample_labels):
                assay_map = pd.Series(maybe, index=sample_labels)
                break
        if assay_map is None:
            raise ValueError("assay filter requested but no assay_by_sample attribute available")
        keep = assay_map[assay_map == assay].index
        counts_df = counts_df.loc[:, counts_df.columns.intersection(keep)]

    feature_metadata = pd.DataFrame(
        {
            "start": reduced["start"].values,
            "end": reduced["end"].values,
            "range_length": reduced["range_length"].values,
        }
    )
    if "contig" in reduced.coords:
        feature_metadata.insert(0, "contig", reduced["contig"].values)

    strand_col = next(
        (c for c in ("strand", "Strand") if aligned_ranges is not None and c in aligned_ranges.columns),
        None,
    )
    if strand_col and aligned_ranges is not None and len(aligned_ranges) == len(feature_metadata):
        feature_metadata.insert(feature_metadata.columns.get_loc("start"), "strand", aligned_ranges[strand_col].values)

    # If caller provided feature IDs and they align, use them to index the counts matrix.
    id_cols = (
        [resolved_feature_id_col] if isinstance(resolved_feature_id_col, str)
        else (resolved_feature_id_col or [])
    )
    if (
        id_cols
        and aligned_ranges is not None
        and all(c in aligned_ranges.columns for c in id_cols)
        and len(aligned_ranges) == len(feature_metadata)
    ):
        for col in reversed(id_cols):
            if col not in feature_metadata.columns:
                feature_metadata.insert(0, col, aligned_ranges[col].values)
        if len(id_cols) == 1:
            counts_df.index = feature_metadata[id_cols[0]].values
        else:
            counts_df.index = pd.MultiIndex.from_frame(feature_metadata[id_cols])

    # Ensure the aggregation key is present in metadata if available in ranges.
    if (
        resolved_aggregate_by
        and aligned_ranges is not None
        and resolved_aggregate_by in aligned_ranges.columns
        and resolved_aggregate_by not in feature_metadata.columns
        and len(aligned_ranges) == len(feature_metadata)
    ):
        feature_metadata.insert(
            0, resolved_aggregate_by, aligned_ranges[resolved_aggregate_by].values
        )

    # Optional aggregation (e.g., sum exons to gene-level).
    if resolved_aggregate_by is not None:
        if resolved_aggregate_by not in feature_metadata.columns:
            raise ValueError(
                f"aggregate_by column '{resolved_aggregate_by}' not found in feature metadata"
            )

        counts_df = counts_df.groupby(feature_metadata[resolved_aggregate_by].values).sum()

        agg_meta = feature_metadata.copy()
        agg_meta[resolved_aggregate_by] = feature_metadata[resolved_aggregate_by].values
        agg_spec = {"start": ("start", "min"), "end": ("end", "max"), "range_length": ("range_length", "sum")}
        if "contig" in agg_meta.columns:
            agg_spec["contig"] = ("contig", "first")
        if "strand" in agg_meta.columns:
            agg_spec["strand"] = ("strand", "first")
        agg_meta = agg_meta.groupby(resolved_aggregate_by).agg(**agg_spec).reset_index()
        feature_metadata = agg_meta
        counts_df.index.name = resolved_aggregate_by
        # Reindex to metadata order without raising if keys differ; missing keys become NaN rows.
        counts_df = counts_df.reindex(feature_metadata[resolved_aggregate_by].values)

    # Align feature_metadata index to counts_df for downstream filtering.
    if len(feature_metadata) == len(counts_df):
        feature_metadata.index = counts_df.index

    # Final NaN handling and integerization (after any reindexing/aggregation that may introduce NaNs)
    if fillna_value is not None:
        counts_df = counts_df.fillna(fillna_value)
    if integerize:
        counts_df = counts_df.round().astype("int64")

    if filter_zero:
        nonzero_mask = (counts_df != 0).any(axis=1)
        counts_df = counts_df.loc[nonzero_mask]
        feature_metadata = feature_metadata.loc[nonzero_mask]
    return counts_df, feature_metadata

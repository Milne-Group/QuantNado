from __future__ import annotations

import pyranges1 as pr
import pandas as pd
import numpy as np
from typing import Iterable
from loguru import logger

from ..dataset.enums import FeatureType


"""GTF utilities using PyRanges to extract feature ranges (genes/transcripts)
and construct promoter windows. Designed to mirror a small slice of ChIPseeker's
annotatePeak use-case."""


GTF_COLUMNS = [
	"seqname",
	"source",
	"feature",
	"start",
	"end",
	"score",
	"strand",
	"frame",
	"attribute",
]


def _parse_attributes(attr_str: str) -> dict:
	"""Parse the GTF attributes column into a dict of key→value strings."""

	if pd.isna(attr_str) or not attr_str:
		return {}
	parts = [p.strip() for p in attr_str.split(";") if p.strip()]
	attrs = {}
	for part in parts:
		if " " not in part:
			continue
		key, val = part.split(" ", 1)
		val = val.strip().strip('"')
		attrs[key] = val
	return attrs


def load_gtf(
	gtf_path: str | Iterable[str],
	feature_types: Iterable[str] | None = None,
	usecols: list[str] | None = None,
) -> pr.PyRanges:
	"""Load a GTF file into a PyRanges object with parsed attributes.

	Parameters
	----------
	gtf_path : str or iterable of str
		Path (or paths) to GTF file(s).
	feature_types : iterable of str, optional
		If provided, filter to these feature types (e.g., ["gene", "transcript"],
		["exon"], etc.).
	usecols : list[str], optional
		Additional attribute keys to extract (e.g., ["gene_id", "gene_name", "transcript_id"].
		Defaults to the common gene/transcript keys.

	Returns
	-------
	pr.PyRanges
		PyRanges object with columns: Chromosome, Start, End, Strand, feature, and any extracted attributes.
	"""

	if usecols is None:
		usecols = ["gene_id", "gene_name", "transcript_id", "gene_type", "gene_biotype"]

	paths = [gtf_path] if isinstance(gtf_path, str) else list(gtf_path)
	ranges_list = []
	
	for path in paths:
		# Use PyRanges to read GTF for proper handling of 1-based coordinates
		pr_obj = pr.read_gtf(path)
		
		# Convert to DataFrame for attribute parsing
		df = pd.DataFrame(pr_obj)
		df = df.rename(columns={"Chromosome": "seqname", "Start": "start", "End": "end"})
		
		if "feature" not in df.columns:
			df["feature"] = df.get("Feature", "unknown")
		
		if feature_types is not None:
			df = df[df["feature"].isin(feature_types)]

		# Parse attributes if present (for custom GTF parsing)
		if "attribute" in df.columns:
			attr_dicts = df["attribute"].apply(_parse_attributes)
			for key in usecols:
				df[key] = attr_dicts.apply(lambda d: d.get(key))
			df = df.drop(columns=["attribute"])
		else:
			# PyRanges may have already parsed attributes into columns; only fill missing ones
			for key in usecols:
				if key not in df.columns:
					df[key] = pd.NA

		ranges_list.append(df)

	if not ranges_list:
		# Return empty PyRanges with proper structure
		empty_df = pd.DataFrame({
			"seqname": pd.Series([], dtype=str),
			"start": pd.Series([], dtype=np.int64),
			"end": pd.Series([], dtype=np.int64),
			"strand": pd.Series([], dtype=str),
			"feature": pd.Series([], dtype=str),
		})
		for col in usecols:
			empty_df[col] = pd.Series([], dtype=object)
		empty_df = empty_df.rename(columns={"seqname": "Chromosome", "start": "Start", "end": "End"})
		return pr.PyRanges(empty_df)

	combined_df = pd.concat(ranges_list, ignore_index=True)
	combined_df = combined_df.rename(columns={"seqname": "Chromosome", "start": "Start", "end": "End"})

	return pr.PyRanges(combined_df)


def extract_feature_ranges(
	gtf_source: pr.PyRanges | pd.DataFrame,
	feature_type: FeatureType | str = FeatureType.GENE,
) -> pr.PyRanges:
	"""Return ranges for a specific feature type (e.g., gene, transcript, exon).

	Adds convenience columns:
	- Length: end - start
	- Midpoint: midpoint of the interval

	Parameters
	----------
	gtf_source : pr.PyRanges or pd.DataFrame
		GTF data loaded via load_gtf or as a DataFrame with columns
		[Chromosome/seqname, Start/start, End/end, feature, ...].
	feature_type : FeatureType or str
		Feature type to extract (e.g., "gene", "transcript", "exon").

	Returns
	-------
	pr.PyRanges
		PyRanges object containing features of the specified type with Length and Midpoint columns.
	"""

	feature_type = FeatureType(feature_type) if isinstance(feature_type, str) else feature_type

	# Convert input to DataFrame for consistent processing
	df = pd.DataFrame(gtf_source)

	# Normalize column names to PyRanges convention
	if "seqname" in df.columns:
		df = df.rename(columns={"seqname": "Chromosome"})
	if "start" in df.columns:
		df = df.rename(columns={"start": "Start"})
	if "end" in df.columns:
		df = df.rename(columns={"end": "End"})

	# Filter by feature type
	if "feature" not in df.columns and "Feature" in df.columns:
		df["feature"] = df["Feature"]
	
	subset = df[df["feature"] == str(feature_type)].copy()
	
	if not subset.empty:
		subset["Length"] = subset["End"] - subset["Start"]
		subset["Midpoint"] = (subset["Start"] + subset["End"]) // 2

	return pr.PyRanges(subset)


def extract_promoters(
	gtf_source: pr.PyRanges | pd.DataFrame,
	upstream: int = 1000,
	downstream: int = 200,
	anchor_feature: FeatureType | str = FeatureType.GENE,
) -> pr.PyRanges:
	"""Build promoter windows around TSS of genes/transcripts.

	Parameters
	----------
	gtf_source : pr.PyRanges or pd.DataFrame
		GTF data loaded via load_gtf or as a DataFrame.
	upstream : int
		Bases upstream of TSS (default: 1000).
	downstream : int
		Bases downstream of TSS (default: 200).
	anchor_feature : FeatureType or str
		Feature type to anchor on ("gene" or "transcript").

	Returns
	-------
	pr.PyRanges
		PyRanges object with promoter windows.
	"""

	anchor_feature = FeatureType(anchor_feature) if isinstance(anchor_feature, str) else anchor_feature
	
	# Get anchors using extract_feature_ranges
	anchors_pr = extract_feature_ranges(gtf_source, feature_type=anchor_feature)
	anchors = pd.DataFrame(anchors_pr)

	# If no features found and anchor was GENE, try TRANSCRIPT as fallback
	if anchors.empty and anchor_feature == FeatureType.GENE:
		logger.warning(
			f"No '{anchor_feature}' features found in GTF; falling back to '{FeatureType.TRANSCRIPT}'"
		)
		anchors_pr = extract_feature_ranges(gtf_source, feature_type=FeatureType.TRANSCRIPT)
		anchors = pd.DataFrame(anchors_pr)

	if anchors.empty:
		# Return empty PyRanges with proper column structure
		empty_df = pd.DataFrame({
			"Chromosome": pd.Series([], dtype=str),
			"Start": pd.Series([], dtype=np.int64),
			"End": pd.Series([], dtype=np.int64),
			"Strand": pd.Series([], dtype=str),
			"feature": pd.Series([], dtype=str),
		})
		return pr.PyRanges(empty_df)

	starts = anchors["Start"].to_numpy()
	ends = anchors["End"].to_numpy()
	strands = anchors["Strand"].fillna("+").to_numpy()

	# Calculate TSS based on strand
	tss = starts.copy()
	tss[strands == "-"] = ends[strands == "-"]

	# Build promoter coordinates
	promo_start = (tss - upstream).clip(min=0)
	promo_end = tss + downstream

	# Create promoter DataFrame
	promoters = anchors.copy()
	promoters["Start"] = promo_start
	promoters["End"] = promo_end
	promoters["feature"] = "promoter"
	
	return pr.PyRanges(promoters)


def annotate_intervals(
	intervals: pr.PyRanges | pd.DataFrame,
	feature_ranges: pr.PyRanges | pd.DataFrame,
	feature_prefix: str = "feature_",
	require_overlap: bool = True,
) -> pr.PyRanges | pd.DataFrame:
	"""Annotate intervals with overlapping feature records using PyRanges join.

	Parameters
	----------
	intervals : pr.PyRanges or pd.DataFrame
		Intervals to annotate. DataFrames must contain columns [Chromosome/contig, Start/start, End/end].
	feature_ranges : pr.PyRanges or pd.DataFrame
		Feature table with [Chromosome/contig, Start/start, End/end] and optional annotation columns.
	feature_prefix : str
		Prefix to add to feature columns in the output.
	require_overlap : bool
		If True, return only overlapping rows. If False and no overlap is found,
		returns the original intervals with NaNs for feature columns.

	Returns
	-------
	pr.PyRanges or pd.DataFrame
		Same type as input (intervals). Contains original columns plus annotated feature columns.
	"""

	# Convert inputs to PyRanges if needed
	intervals_pr = _to_pyranges(intervals, "Chromosome", "Start", "End") if not isinstance(intervals, pr.PyRanges) else intervals
	features_pr = _to_pyranges(feature_ranges, "Chromosome", "Start", "End") if not isinstance(feature_ranges, pr.PyRanges) else feature_ranges

	# Perform join operation
	joined = intervals_pr.join_overlaps(features_pr, strand_behavior="ignore")

	# Get result as DataFrame
	result_df = pd.DataFrame(joined)

	# Rename feature columns with prefix
	feature_cols = [c for c in features_pr.columns if c not in {"Chromosome", "Start", "End", "Strand"}]
	rename_map = {c: f"{feature_prefix}{c}" for c in feature_cols}
	result_df = result_df.rename(columns=rename_map)

	# Handle require_overlap logic
	if not require_overlap:
		# Get original intervals as DataFrame
		intervals_df = pd.DataFrame(intervals_pr)
		
		# For non-overlapping intervals, add them back with NaN feature columns
		original_ids = set(zip(intervals_df["Chromosome"], intervals_df["Start"], intervals_df["End"]))
		result_ids = set(zip(result_df["Chromosome"], result_df["Start"], result_df["End"]))
		missing_ids = original_ids - result_ids
		
		if missing_ids:
			missing_rows = []
			for chrom, start, end in missing_ids:
				row = intervals_df[(intervals_df["Chromosome"] == chrom) & 
								  (intervals_df["Start"] == start) & 
								  (intervals_df["End"] == end)].iloc[0].to_dict()
				for col in rename_map.values():
					row[col] = pd.NA
				missing_rows.append(row)
			result_df = pd.concat([result_df, pd.DataFrame(missing_rows)], ignore_index=True)
	
	# Return in same format as input
	if isinstance(intervals, pr.PyRanges):
		return pr.PyRanges(result_df)
	else:
		return result_df


def _to_pyranges(
	data: pd.DataFrame,
	chromosome_col: str = "Chromosome",
	start_col: str = "Start",
	end_col: str = "End",
) -> pr.PyRanges:
	"""Convert DataFrame with flexible column names to PyRanges.

	Parameters
	----------
	data : pd.DataFrame
		DataFrame with genomic intervals.
	chromosome_col : str
		Column name for chromosome/contig (default: "Chromosome").
	start_col : str
		Column name for start position (default: "Start").
	end_col : str
		Column name for end position (default: "End").

	Returns
	-------
	pr.PyRanges
		PyRanges object with standard columns [Chromosome, Start, End].
	"""

	df = data.copy()
	
	# Normalize column names to PyRanges standard
	rename_map = {}
	if chromosome_col not in df.columns:
		for col in ["contig", "seqname", "chr"]:
			if col in df.columns:
				rename_map[col] = chromosome_col
				break
	if start_col not in df.columns:
		for col in ["start", "pos"]:
			if col in df.columns:
				rename_map[col] = start_col
				break
	if end_col not in df.columns:
		for col in ["end"]:
			if col in df.columns:
				rename_map[col] = end_col
				break
	
	if rename_map:
		df = df.rename(columns=rename_map)
	
	# Validate required columns exist
	required = {chromosome_col, start_col, end_col}
	if not required.issubset(df.columns):
		missing = required - set(df.columns)
		raise ValueError(f"DataFrame must contain columns: {missing}")
	
	return pr.PyRanges(df)


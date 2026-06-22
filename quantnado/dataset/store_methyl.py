"""MethylStore — per-sample Zarr store for TAPS/WGBS methylation data.

Store layout::

    {SampleName}.zarr/
    ├── {chrom}/
    │   ├── coverage      (1, chrom_len)  uint32   — BAM coverage via bamnado
    │   ├── methyl_pct    (1, chrom_len)  float32  — NaN where no CpG coverage
    │   ├── n_methylated  (1, chrom_len)  uint32
    │   └── n_total       (1, chrom_len)  uint32
    ├── metadata/
    │   ├── completed        bool    (1,)
    │   ├── total_reads      int64   (1,)
    │   ├── mean_read_length float32 (1,)
    │   └── sparsity         float32 (1,)
    └── root.attrs: assay="meth", sample, chromsizes, chunk_len
"""

from __future__ import annotations

from pathlib import Path

import bamnado
import numpy as np
import pandas as pd
import pyarrow.csv as pa_csv
import pyarrow.compute as pc
import zarr
from loguru import logger
from zarr.storage import LocalStore

from .metadata import _parse_chromsizes, create_metadata_group
from .store_bam import (
    _collect_bam_stats,
    _copy_read_filter,
    _delete_path,
    _get_chromsizes_from_bam,
    _resolve_chunk_len,
    _resolve_compressors,
    _normalize_path,
    BIN_SIZE,
    DEFAULT_CONSTRUCTION_COMPRESSION,
)


# ---------------------------------------------------------------------------
# bedGraph / CXreport readers  (kept from previous implementation)
# ---------------------------------------------------------------------------


def _read_bedgraph(
    path: Path | str,
    filter_chromosomes: bool = True,
    test: bool = False,
    test_chromosomes: list[str] | tuple[str, ...] | None = None,
) -> dict[str, pd.DataFrame]:
    """Read a CpG bedGraph file from MethylDackel or similar tools.

    Returns dict mapping chromosome → DataFrame with columns:
        start (int64, 0-based), methylation_pct (float32),
        n_methylated (uint32), n_unmethylated (uint32)

    Uses pyarrow's CSV reader which is 5-8× faster than pandas for
    large whole-genome bedGraph files (~28M rows for hg38 CpG).
    """
    path = Path(path)

    # Detect and skip track/browser header lines
    skip = 0
    with open(path) as fh:
        for line in fh:
            if line.startswith(("track ", "browser ")) or "type=" in line:
                skip += 1
            else:
                break

    # Detect column count from first data line
    with open(path) as fh:
        for i, line in enumerate(fh):
            if i >= skip:
                n_cols = len(line.rstrip("\n").split("\t"))
                break
        else:
            n_cols = 0

    if n_cols < 4:
        raise ValueError(
            f"bedGraph file {path.name} has only {n_cols} columns; expected at least 4"
        )

    # Column names: bed4 = chrom, start, end, methyl_pct [, col4, col5]
    all_col_names = ["chrom", "start", "end", "methylation_pct", "col4", "col5"]
    col_names = all_col_names[:n_cols]

    read_opts = pa_csv.ReadOptions(skip_rows=skip, column_names=col_names)
    parse_opts = pa_csv.ParseOptions(delimiter="\t")
    convert_opts = pa_csv.ConvertOptions(null_values=[".", "NA", "nan"])

    table = pa_csv.read_csv(path, read_options=read_opts, parse_options=parse_opts,
                            convert_options=convert_opts)
    df = table.to_pandas()

    result = pd.DataFrame()
    result["chrom"] = df["chrom"]
    result["start"] = df["start"].astype(np.int64)
    result["methylation_pct"] = df["methylation_pct"].astype(np.float32)

    if n_cols >= 6:
        result["n_unmethylated"] = df["col4"].fillna(0).astype(np.uint32)
        result["n_methylated"] = df["col5"].fillna(0).astype(np.uint32)
    elif n_cols == 5:
        coverage = df["col4"].fillna(0)
        pct = result["methylation_pct"].fillna(0) / 100.0
        result["n_methylated"] = (pct * coverage).round().astype(np.uint32)
        result["n_unmethylated"] = ((1 - pct) * coverage).round().astype(np.uint32)
    else:
        result["n_unmethylated"] = np.uint32(0)
        result["n_methylated"] = np.uint32(0)

    result["n_total"] = (result["n_methylated"] + result["n_unmethylated"]).astype(np.uint32)

    if filter_chromosomes:
        result = result[
            result["chrom"].str.startswith("chr") & ~result["chrom"].str.contains("_")
        ]
    if test or test_chromosomes:
        test_chroms = _parse_chromsizes({}, test=test, test_chromosomes=test_chromosomes)
        result = result[result["chrom"].isin(test_chroms)]

    return {chrom: grp.reset_index(drop=True) for chrom, grp in result.groupby("chrom")}


# ---------------------------------------------------------------------------
# Dense-array scatter helper
# ---------------------------------------------------------------------------


def _scatter_to_dense(
    positions: np.ndarray,
    values: np.ndarray,
    chrom_len: int,
    fill_value,
    dtype,
) -> np.ndarray:
    """Scatter sparse CpG values into a dense array of length chrom_len.

    positions : 0-based int64 positions
    values    : values at those positions
    Returns a 1D array of shape (chrom_len,).
    """
    arr = np.full(chrom_len, fill_value, dtype=dtype)
    mask = (positions >= 0) & (positions < chrom_len)
    arr[positions[mask]] = values[mask]
    return arr


# ---------------------------------------------------------------------------
# MethylStore
# ---------------------------------------------------------------------------


class MethylStore:
    """Per-sample Zarr store for TAPS/WGBS methylation data.

    Stores dense ``(1, chrom_len)`` arrays for coverage, methylation_pct,
    n_methylated, and n_total per chromosome.  BAM provides coverage; bedGraph
    provides methylation values.

    Use :meth:`from_files` to create or :meth:`open` to read.
    """

    def __init__(
        self,
        store_path: Path | str,
        sample: str,
        chromsizes: dict[str, int],
        *,
        chunk_len: int,
        compressors: list,
        overwrite: bool = True,
    ) -> None:
        self.store_path = _normalize_path(store_path)
        self.sample = sample
        self.chromsizes = chromsizes
        self.chromosomes = sorted(chromsizes.keys())
        self.chunk_len = chunk_len
        self.compressors = compressors

        if overwrite:
            _delete_path(self.store_path)

        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        local_store = LocalStore(str(self.store_path))
        self.root = zarr.group(store=local_store, overwrite=True, zarr_format=3)
        self.meta = create_metadata_group(self.root, sample)
        self._init_arrays()
        self.root.attrs.update({
            "assay": "meth",
            "sample": sample,
            "chromsizes": chromsizes,
            "chunk_len": chunk_len,
        })

    def _init_arrays(self) -> None:
        for chrom, chrom_len in self.chromsizes.items():
            grp = self.root.require_group(chrom)
            for key, dtype, fill in [
                ("coverage",     np.uint32,  0),
                ("methyl_pct",   np.float32, np.nan),
                ("n_methylated", np.uint32,  0),
                ("n_total",      np.uint32,  0),
            ]:
                grp.require_array(
                    key,
                    shape=(1, chrom_len),
                    chunks=(1, self.chunk_len),
                    dtype=dtype,
                    compressors=self.compressors,
                    fill_value=fill,
                    overwrite=True,
                )

    def _write_coverage(
        self,
        bam_file: str,
        read_filter: bamnado.ReadFilter,
        use_fragment: bool,
    ) -> float:
        """Write BAM coverage to root[chrom]["coverage"]. Returns mean sparsity."""
        sparsity_vals = []
        for chrom, chrom_size in self.chromsizes.items():
            sig = bamnado.get_signal_for_chromosome(
                bam_path=bam_file,
                chromosome_name=chrom,
                bin_size=BIN_SIZE,
                scale_factor=1.0,
                use_fragment=use_fragment,
                ignore_scaffold_chromosomes=False,
                read_filter=read_filter,
            ).astype(np.uint32)
            if sig.shape[0] != chrom_size:
                sig = sig[:chrom_size] if sig.shape[0] > chrom_size else np.pad(
                    sig, (0, chrom_size - sig.shape[0])
                )
            self.root[chrom]["coverage"][0, :] = sig
            sparsity_vals.append(float(np.sum(sig == 0) / sig.size * 100))
        return float(np.mean(sparsity_vals)) if sparsity_vals else float("nan")

    def _write_methylation(self, by_chrom: dict[str, pd.DataFrame]) -> None:
        """Scatter sparse bedGraph data into dense arrays."""
        for chrom, chrom_size in self.chromsizes.items():
            if chrom not in by_chrom:
                continue
            df = by_chrom[chrom]
            positions = df["start"].values.astype(np.int64)

            self.root[chrom]["methyl_pct"][0, :] = _scatter_to_dense(
                positions, df["methylation_pct"].values, chrom_size, np.nan, np.float32
            )
            self.root[chrom]["n_methylated"][0, :] = _scatter_to_dense(
                positions, df["n_methylated"].values, chrom_size, 0, np.uint32
            )
            self.root[chrom]["n_total"][0, :] = _scatter_to_dense(
                positions, df["n_total"].values, chrom_size, 0, np.uint32
            )

    def _finalise(self, bam_file: str, sparsity: float) -> None:
        bam_hash, total_reads, mean_read_length = _collect_bam_stats(bam_file)
        self.meta["completed"][0] = True
        self.meta["total_reads"][0] = total_reads
        self.meta["mean_read_length"][0] = mean_read_length
        self.meta["sparsity"][0] = sparsity
        zarr.consolidate_metadata(str(self.store_path))
        logger.info(f"Completed {self.sample}: {total_reads:,} mapped reads")

    @classmethod
    def open(cls, store_path: str | Path, read_only: bool = True) -> "MethylStore":
        store_path = _normalize_path(store_path)
        if not store_path.exists():
            raise FileNotFoundError(f"Store not found: {store_path}")
        mode = "r" if read_only else "r+"
        root = zarr.open_group(str(store_path), mode=mode)
        obj = object.__new__(cls)
        obj.store_path = store_path
        obj.root = root
        attrs = dict(root.attrs)
        obj.sample = attrs.get("sample", "")
        obj.chromsizes = {str(k): int(v) for k, v in attrs.get("chromsizes", {}).items()}
        obj.chromosomes = sorted(obj.chromsizes.keys())
        obj.chunk_len = int(attrs.get("chunk_len", 65536))
        obj.meta = root.get("metadata")
        return obj

    @classmethod
    def from_files(
        cls,
        bam_path: str | Path,
        methyl_path: str | Path,
        store_path: Path | str,
        sample: str,
        chromsizes: str | Path | dict[str, int] | None = None,
        *,
        bam_filter: bamnado.ReadFilter | None = None,
        count_fragments: bool = False,
        paired: bool = True,
        chunk_len: int | None = None,
        construction_compression: str = DEFAULT_CONSTRUCTION_COMPRESSION,
        overwrite: bool = True,
        filter_chromosomes: bool = True,
        test: bool = False,
        test_chromosomes: list[str] | tuple[str, ...] | None = None,
        log_file: Path | None = None,
    ) -> "MethylStore":
        """Create a per-sample MethylStore from a BAM file and a MethylDackel bedGraph.

        Parameters
        ----------
        bam_path:
            Aligned BAM file (for coverage).
        methyl_path:
            MethylDackel CpG bedGraph (for methylation values).
        store_path:
            Output .zarr directory.
        sample:
            Sample name.
        chromsizes:
            Path to .chrom.sizes, dict, or None to infer from BAM header.
        paired:
            Whether the BAM contains paired-end reads. Set ``False`` for
            single-end libraries to disable proper-pair filtering and
            fragment-level coverage.
        """
        if log_file is not None:
            from quantnado.utils import setup_logging
            setup_logging(Path(log_file), verbose=False)

        bam_path = str(bam_path)

        if chromsizes is None:
            chromsizes = _get_chromsizes_from_bam(bam_path)

        chromsizes_dict = _parse_chromsizes(
            chromsizes,
            filter_chromosomes=filter_chromosomes,
            test=test,
            test_chromosomes=test_chromosomes,
        )

        logger.info(f"Reading methylation data from {methyl_path}")
        by_chrom = _read_bedgraph(
            methyl_path,
            filter_chromosomes=filter_chromosomes,
            test=test,
            test_chromosomes=test_chromosomes,
        )

        resolved_chunk_len = _resolve_chunk_len(chromsizes_dict, Path(store_path), chunk_len)
        compressors = _resolve_compressors(construction_compression)
        if bam_filter is not None:
            read_filter = _copy_read_filter(bam_filter)
            if not paired:
                read_filter.proper_pair = False
        else:
            read_filter = bamnado.ReadFilter(proper_pair=paired)

        use_fragment = bool(count_fragments)
        if use_fragment and not paired:
            logger.warning(
                "count_fragments=True requires paired-end reads; using read-level coverage"
            )
            use_fragment = False

        store = cls(
            store_path=store_path,
            sample=sample,
            chromsizes=chromsizes_dict,
            chunk_len=resolved_chunk_len,
            compressors=compressors,
            overwrite=overwrite,
        )
        store.root.attrs.update({
            "paired": bool(paired),
            "count_fragments": use_fragment,
        })

        logger.info(f"Writing BAM coverage for {sample}")
        sparsity = store._write_coverage(bam_path, read_filter, use_fragment)
        logger.info(f"Writing methylation data for {sample}")
        store._write_methylation(by_chrom)
        store._finalise(bam_path, sparsity)

        return store

    @property
    def completed(self) -> bool:
        if self.meta is None:
            return False
        return bool(self.meta["completed"][0])

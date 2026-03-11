from __future__ import annotations

import io
import shutil
from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from loguru import logger
from zarr.codecs import BloscCodec
from zarr.storage import LocalStore

from .core import BaseStore
from .constants import DEFAULT_CHUNK_LEN
from .store_bam import _compute_sample_hash, _to_str_list


def _read_bedgraph(path: Path | str, filter_chromosomes: bool = True) -> dict[str, pd.DataFrame]:
    """
    Read a CpG bedGraph file from MethylDackel, Bismark, or similar tools.

    Supported column layouts (after optional track/browser header lines):
        4-col: chrom, start, end, methylation_pct
        5-col: chrom, start, end, methylation_pct, coverage
        6-col: chrom, start, end, methylation_pct, n_unmethylated, n_methylated

    For files with fewer than 6 columns, n_unmethylated and n_methylated are
    inferred where possible (5-col: pct*cov and (1-pct)*cov) or set to 0.

    Returns
    -------
    dict mapping chromosome name -> DataFrame with columns:
        chrom, start, end, methylation_pct, n_unmethylated, n_methylated
    """
    path = Path(path)
    with open(path) as _fh:
        _data = "".join(
            line for line in _fh 
            if not (line.startswith(("track ", "browser ")) or "type=" in line)
        )
    df = pd.read_csv(io.StringIO(_data), sep="\t", header=None, dtype=str)
    df = df.reset_index(drop=True)

    n_cols = df.shape[1]
    if n_cols < 4:
        raise ValueError(
            f"bedGraph file {path.name} has only {n_cols} columns; expected at least 4"
        )

    df = df.iloc[:, :6]  # ignore extra columns beyond 6
    df.columns = range(df.shape[1])

    result = pd.DataFrame()
    result["chrom"] = df[0].astype(str)
    result["start"] = pd.to_numeric(df[1]).astype(np.int64)
    result["end"] = pd.to_numeric(df[2]).astype(np.int64)
    result["methylation_pct"] = pd.to_numeric(df[3], errors="coerce").astype(np.float32)

    if n_cols >= 6:
        result["n_unmethylated"] = pd.to_numeric(df[4], errors="coerce").fillna(0).astype(np.uint16)
        result["n_methylated"] = pd.to_numeric(df[5], errors="coerce").fillna(0).astype(np.uint16)
    elif n_cols == 5:
        # 5th column is total coverage; split by methylation_pct
        coverage = pd.to_numeric(df[4], errors="coerce").fillna(0)
        pct = result["methylation_pct"].fillna(0) / 100.0
        result["n_methylated"] = (pct * coverage).round().astype(np.uint16)
        result["n_unmethylated"] = ((1 - pct) * coverage).round().astype(np.uint16)
    else:
        result["n_unmethylated"] = np.uint16(0)
        result["n_methylated"] = np.uint16(0)

    if filter_chromosomes:
        result = result[result["chrom"].str.startswith("chr") & ~result["chrom"].str.contains("_")]

    return {chrom: grp.reset_index(drop=True) for chrom, grp in result.groupby("chrom")}


def _read_cxreport(path: Path | str, filter_chromosomes: bool = True) -> dict[str, pd.DataFrame]:
    """
    Read a biomodal evoC CXreport file, merging forward and reverse strand per CpG.

    Format (no header): chrom, pos(0-based), strand, n_mc, n_hmc, n_c, context,
    trinucleotide, total_coverage

    The + and - strand rows for each CpG are merged by summing their counts.
    The canonical position is the + strand C position.

    Returns
    -------
    dict mapping chromosome name -> DataFrame with columns:
        start (int64, 0-based), n_mc (uint16), n_hmc (uint16), n_c (uint16),
        methylation_pct (float32, NaN where total=0)
    """
    path = Path(path)
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["chrom", "pos", "strand", "n_mc", "n_hmc", "n_c", "context", "trinuc", "total"],
        dtype={
            "chrom": str,
            "pos": np.int64,
            "strand": str,
            "n_mc": np.int32,
            "n_hmc": np.int32,
            "n_c": np.int32,
            "context": str,
            "trinuc": str,
            "total": np.int32,
        },
    )

    if filter_chromosomes:
        df = df[df["chrom"].str.startswith("chr") & ~df["chrom"].str.contains("_")]

    # Canonical CpG position: + strand pos as-is, - strand pos - 1
    df["canonical_pos"] = np.where(df["strand"] == "+", df["pos"], df["pos"] - 1)

    # Merge both strands by summing counts
    agg = (
        df.groupby(["chrom", "canonical_pos"], sort=True)[["n_mc", "n_hmc", "n_c", "total"]]
        .sum()
        .reset_index()
    )
    agg = agg.rename(columns={"canonical_pos": "start"})
    agg["start"] = agg["start"].astype(np.int64)

    total = agg["total"].to_numpy(dtype=np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        meth_pct = np.where(total > 0, (agg["n_mc"] + agg["n_hmc"]) / total * 100.0, np.nan)
    agg["methylation_pct"] = meth_pct.astype(np.float32)

    agg["n_mc"] = agg["n_mc"].clip(0, 65535).astype(np.uint16)
    agg["n_hmc"] = agg["n_hmc"].clip(0, 65535).astype(np.uint16)
    agg["n_c"] = agg["n_c"].clip(0, 65535).astype(np.uint16)

    cols = ["start", "n_mc", "n_hmc", "n_c", "methylation_pct"]
    return {chrom: grp[cols].reset_index(drop=True) for chrom, grp in agg.groupby("chrom")}


def _read_split_cxreport(
    mc_path: Path | str | None,
    hmc_path: Path | str | None,
    filter_chromosomes: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Read one or both 7-column split CXreport files (mc and/or hmC) and merge
    them into per-CpG DataFrames.

    Format (no header): chrom, pos(0-based), strand, n_mod, n_not_mod,
    context, trinucleotide

    When both files are provided they must have identical row order.
    total = n_mod_mc + n_not_mod_mc  (same as hmc counterpart).
    n_c = total - n_mc - n_hmc.

    When only one file is provided the other modification is assumed to be 0:
    n_c = n_not_mod from the provided file.

    Returns
    -------
    dict mapping chromosome name -> DataFrame with columns:
        start (int64, 0-based), n_mc (uint16), n_hmc (uint16), n_c (uint16),
        methylation_pct (float32, NaN where total=0)
    """
    if mc_path is None and hmc_path is None:
        raise ValueError("At least one of mc_path or hmc_path must be provided")

    _cols = ["chrom", "pos", "strand", "n_mod", "n_not_mod", "context", "trinuc"]
    _dtypes = {
        "chrom": str,
        "pos": np.int64,
        "strand": str,
        "n_mod": np.int32,
        "n_not_mod": np.int32,
        "context": str,
        "trinuc": str,
    }

    if mc_path is not None and hmc_path is not None:
        mc = pd.read_csv(Path(mc_path), sep="\t", header=None, names=_cols, dtype=_dtypes)
        hmc = pd.read_csv(Path(hmc_path), sep="\t", header=None, names=_cols, dtype=_dtypes)
        df = pd.DataFrame(
            {
                "chrom": mc["chrom"],
                "pos": mc["pos"],
                "strand": mc["strand"],
                "n_mc": mc["n_mod"].astype(np.int32),
                "n_hmc": hmc["n_mod"].astype(np.int32),
                "total": (mc["n_mod"] + mc["n_not_mod"]).astype(np.int32),
            }
        )
    elif mc_path is not None:
        mc = pd.read_csv(Path(mc_path), sep="\t", header=None, names=_cols, dtype=_dtypes)
        df = pd.DataFrame(
            {
                "chrom": mc["chrom"],
                "pos": mc["pos"],
                "strand": mc["strand"],
                "n_mc": mc["n_mod"].astype(np.int32),
                "n_hmc": np.zeros(len(mc), dtype=np.int32),
                "total": (mc["n_mod"] + mc["n_not_mod"]).astype(np.int32),
            }
        )
    else:
        hmc = pd.read_csv(Path(hmc_path), sep="\t", header=None, names=_cols, dtype=_dtypes)
        df = pd.DataFrame(
            {
                "chrom": hmc["chrom"],
                "pos": hmc["pos"],
                "strand": hmc["strand"],
                "n_mc": np.zeros(len(hmc), dtype=np.int32),
                "n_hmc": hmc["n_mod"].astype(np.int32),
                "total": (hmc["n_mod"] + hmc["n_not_mod"]).astype(np.int32),
            }
        )

    df["n_c"] = (df["total"] - df["n_mc"] - df["n_hmc"]).clip(lower=0).astype(np.int32)

    if filter_chromosomes:
        df = df[df["chrom"].str.startswith("chr") & ~df["chrom"].str.contains("_")]

    df["canonical_pos"] = np.where(df["strand"] == "+", df["pos"], df["pos"] - 1)

    agg = (
        df.groupby(["chrom", "canonical_pos"], sort=True)[["n_mc", "n_hmc", "n_c", "total"]]
        .sum()
        .reset_index()
        .rename(columns={"canonical_pos": "start"})
    )
    agg["start"] = agg["start"].astype(np.int64)

    total = agg["total"].to_numpy(dtype=np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        meth_pct = np.where(total > 0, (agg["n_mc"] + agg["n_hmc"]) / total * 100.0, np.nan)
    agg["methylation_pct"] = meth_pct.astype(np.float32)

    agg["n_mc"] = agg["n_mc"].clip(0, 65535).astype(np.uint16)
    agg["n_hmc"] = agg["n_hmc"].clip(0, 65535).astype(np.uint16)
    agg["n_c"] = agg["n_c"].clip(0, 65535).astype(np.uint16)

    cols = ["start", "n_mc", "n_hmc", "n_c", "methylation_pct"]
    return {chrom: grp[cols].reset_index(drop=True) for chrom, grp in agg.groupby("chrom")}


class MethylStore(BaseStore):
    """
    Zarr-backed methylation store for CpG-level data from MethylDackel bedGraph files.

    Data is stored sparsely - only positions with coverage across any sample are retained.

    Per-chromosome zarr layout::

        <chrom>/
            positions         int64[n_cpg]          CpG start positions (0-based)
            methylation_pct   float32[n_samples, n_cpg]  NaN where no coverage
            n_methylated      uint16[n_samples, n_cpg]
            n_unmethylated    uint16[n_samples, n_cpg]

    Example
    -------
    >>> store = MethylStore.from_bedgraph_files(
    ...     methyldackel_files=["sample1.bedGraph", "sample2.bedGraph"],
    ...     store_path="methylation.zarr",
    ... )
    >>> xr_dict = store.to_xarray(variable="methylation_pct")
    >>> region = store.extract_region("chr21:5000000-6000000")
    """

    def __init__(
        self,
        store_path: Path | str,
        sample_names: list[str],
        *,
        overwrite: bool = True,
        resume: bool = False,
        read_only: bool = False,
        mc_hmc_split: bool = False,
    ) -> None:
        self.path = Path(store_path)
        self.store_path = self._normalize_path(self.path)

        # Initialize BaseStore attributes
        if self.store_path.exists() and not overwrite:
            self.root = zarr.open_group(str(self.store_path), mode="r" if read_only else "r+")
            self._init_common_attributes(sample_names)
        else:
            self.sample_names = [str(s) for s in sample_names]
            self._setup_sample_lookup()
            self.completed_mask_raw = np.zeros(len(self.sample_names), dtype=bool)
            self._metadata_cache = None

        self.n_samples = len(self.sample_names)
        self.sample_hash = _compute_sample_hash(self.sample_names)
        self.compressor = BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")
        self.read_only = read_only
        self._mc_hmc_split_init = mc_hmc_split  # stored before _init_store is called

        if self.n_samples == 0:
            raise ValueError("sample_names must not be empty")

        if self.store_path.exists():
            if overwrite:
                if read_only:
                    raise ValueError("Cannot overwrite store in read-only mode.")
                logger.warning(f"Deleting existing store at: {self.store_path}")
                if self.store_path.is_dir():
                    shutil.rmtree(self.store_path)
                else:
                    self.store_path.unlink()
                self._init_store()
            elif resume:
                self._load_existing()
                self._validate_sample_names()
            else:
                raise FileExistsError(
                    f"Store already exists at {self.store_path}; set overwrite=True or resume=True"
                )
        else:
            if read_only:
                raise FileNotFoundError(
                    f"Store does not exist at {self.store_path} (read_only=True)"
                )
            self._init_store()

    @classmethod
    def open(cls, store_path: str | Path, read_only: bool = True) -> "MethylStore":
        """Open an existing MethylStore for reading (default) or writing."""
        store_path = cls._normalize_path(store_path)
        if not store_path.exists():
            raise FileNotFoundError(f"Store does not exist at {store_path}")
        group = zarr.open_group(str(store_path), mode="r" if read_only else "r+")
        try:
            sample_names = list(group.attrs["sample_names"])
        except KeyError as e:
            raise ValueError(f"Missing required attribute in store: {e}")
        return cls(
            store_path=store_path,
            sample_names=sample_names,
            overwrite=False,
            resume=True,
            read_only=read_only,
        )

    @staticmethod
    def _normalize_path(path: Path | str) -> Path:
        path = Path(path)
        if not str(path).endswith(".zarr"):
            path = path.with_suffix(".zarr")
        return path

    def _check_writable(self) -> None:
        if getattr(self, "read_only", False):
            raise RuntimeError(
                "Store is in read-only mode. Reopen with read_only=False to allow modifications."
            )

    def _init_store(self) -> None:
        store = LocalStore(str(self.store_path))
        self.root = zarr.group(store=store, overwrite=True, zarr_format=3)
        self.meta = self.root.create_group("metadata")
        self.meta.create_array(
            name="completed",
            shape=(self.n_samples,),
            dtype=bool,
            fill_value=False,
            overwrite=True,
        )
        self.root.attrs.update(
            {
                "sample_names": self.sample_names,
                "sample_names_hash": self.sample_hash,
                "n_samples": self.n_samples,
                "store_type": "methylation",
                "has_mc_hmc_split": self._mc_hmc_split_init,
            }
        )
        logger.info(f"Initialized MethylStore at {self.store_path}")

    def _load_existing(self) -> None:
        store = LocalStore(str(self.store_path))
        self.root = zarr.open_group(store=store, mode="a")
        self.meta = self.root["metadata"]
        logger.info(f"Resuming existing MethylStore at {self.store_path}")

    def _validate_sample_names(self) -> None:
        stored = self.root.attrs.get("sample_names")
        if stored is None:
            raise ValueError("Existing store missing sample_names attribute")
        if [str(s) for s in stored] != self.sample_names:
            raise ValueError("sample_names mismatch; refusing to resume to prevent corruption")

    @property
    def completed_mask(self) -> np.ndarray:
        return self.meta["completed"][:].astype(bool)

    @property
    def chromosomes(self) -> list[str]:
        return [k for k in self.root.keys() if k != "metadata"]

    @property
    def has_mc_hmc_split(self) -> bool:
        """True if this store was built from evoC CXreport files (separate mC/hmC arrays)."""
        return bool(self.root.attrs.get("has_mc_hmc_split", False))

    def _init_chrom_arrays(self, chrom: str, positions: np.ndarray) -> None:
        """Create zarr arrays for a chromosome given the union of CpG positions."""
        n_pos = len(positions)
        grp = self.root.require_group(chrom)
        grp.create_array(
            name="positions", shape=(n_pos,), dtype=np.int64, fill_value=0, overwrite=True
        )
        grp["positions"][:] = positions

        chunk_len = min(DEFAULT_CHUNK_LEN, max(1, n_pos))

        if self.has_mc_hmc_split:
            arrays = [
                ("methylation_pct", np.float32, np.nan),
                ("n_mc", np.uint16, 0),
                ("n_hmc", np.uint16, 0),
                ("n_c", np.uint16, 0),
            ]
        else:
            arrays = [
                ("methylation_pct", np.float32, np.nan),
                ("n_methylated", np.uint16, 0),
                ("n_unmethylated", np.uint16, 0),
            ]

        for name, dtype, fill in arrays:
            grp.create_array(
                name=name,
                shape=(self.n_samples, n_pos),
                chunks=(1, chunk_len),
                dtype=dtype,
                compressors=[self.compressor],
                fill_value=fill,
                overwrite=True,
            )

    @classmethod
    def from_bedgraph_files(
        cls,
        methyldackel_files: list[str | Path],
        store_path: Path | str,
        sample_names: list[str] | None = None,
        metadata: pd.DataFrame | Path | str | None = None,
        *,
        filter_chromosomes: bool = True,
        overwrite: bool = True,
        resume: bool = False,
        sample_column: str = "sample_id",
    ) -> "MethylStore":
        """
        Create a MethylStore from MethylDackel bedGraph files.

        Each file corresponds to one sample. CpG positions are unioned across all
        samples per chromosome; positions missing in a sample are filled with NaN
        (methylation_pct) or 0 (counts).

        Parameters
        ----------
        methyldackel_files : list of str or Path
            Paths to MethylDackel bedGraph files.
        store_path : Path or str
            Output Zarr store path.
        sample_names : list of str, optional
            Sample names aligned with ``methyldackel_files``. Defaults to file stems.
        metadata : DataFrame, Path, or str, optional
            Sample metadata CSV to attach.
        filter_chromosomes : bool, default True
            Keep only canonical chromosomes (chr* without underscores).
        overwrite : bool, default True
            Overwrite existing store.
        resume : bool, default False
            Resume an existing store.
        sample_column : str, default "sample_id"
            Column in metadata matching sample names.
        """
        methyldackel_files = [Path(f) for f in methyldackel_files]
        if sample_names is None:
            sample_names = [f.stem for f in methyldackel_files]
        if len(sample_names) != len(methyldackel_files):
            raise ValueError("sample_names length must match methyldackel_files length")

        store = cls(
            store_path=store_path,
            sample_names=sample_names,
            overwrite=overwrite,
            resume=resume,
        )

        # Read all files into memory (single pass)
        logger.info("Reading bedGraph files...")
        all_file_data: list[dict[str, pd.DataFrame]] = []
        for path in methyldackel_files:
            logger.info(f"  {path.name}")
            all_file_data.append(_read_bedgraph(path, filter_chromosomes=filter_chromosomes))

        # Collect all chromosomes seen across all files
        all_chroms: set[str] = set()
        for fd in all_file_data:
            all_chroms.update(fd.keys())

        logger.info(f"Building union positions and writing store ({len(all_chroms)} chroms)...")
        for chrom in sorted(all_chroms):
            # (sample_idx, DataFrame) pairs for this chromosome
            chrom_samples = [(i, fd[chrom]) for i, fd in enumerate(all_file_data) if chrom in fd]
            # Union of CpG start positions across all samples
            all_positions = np.unique(
                np.concatenate([df["start"].values for _, df in chrom_samples])
            )
            store._init_chrom_arrays(chrom, all_positions)

            pos_to_idx: dict[int, int] = {int(p): i for i, p in enumerate(all_positions)}

            for sample_idx, df in chrom_samples:
                indices = np.array([pos_to_idx[int(p)] for p in df["start"].values])
                store.root[chrom]["methylation_pct"][sample_idx, indices] = df[
                    "methylation_pct"
                ].values
                store.root[chrom]["n_methylated"][sample_idx, indices] = df["n_methylated"].values
                store.root[chrom]["n_unmethylated"][sample_idx, indices] = df[
                    "n_unmethylated"
                ].values

            logger.info(f"  {chrom}: {len(all_positions)} CpG sites")

        # Mark all samples complete
        store.meta["completed"][:] = True
        store.root.attrs["chromosomes"] = sorted(all_chroms)

        if metadata is not None:
            if isinstance(metadata, (str, Path)):
                metadata_df = pd.read_csv(metadata)
            else:
                metadata_df = metadata
            store.set_metadata(metadata_df, sample_column=sample_column)

        return store

    @classmethod
    def from_cxreport_files(
        cls,
        cxreport_files: list[str | Path],
        store_path: Path | str,
        sample_names: list[str] | None = None,
        metadata: pd.DataFrame | Path | str | None = None,
        *,
        filter_chromosomes: bool = True,
        overwrite: bool = True,
        resume: bool = False,
        sample_column: str = "sample_id",
    ) -> "MethylStore":
        """
        Create a MethylStore from biomodal evoC CXreport files.

        Each file corresponds to one sample. The + and - strand rows for each
        CpG are merged by summing read counts. Stores separate mC and hmC
        arrays (``n_mc``, ``n_hmc``, ``n_c``) alongside ``methylation_pct``
        ((mC + hmC) / total coverage).

        Parameters
        ----------
        cxreport_files : list of str or Path
            Paths to ``.CXreport.txt.gz`` files from the biomodal pipeline.
        store_path : Path or str
            Output Zarr store path.
        sample_names : list of str, optional
            Sample names aligned with ``cxreport_files``. Defaults to file stems
            up to the first ``.``.
        metadata : DataFrame, Path, or str, optional
            Sample metadata CSV to attach.
        filter_chromosomes : bool, default True
            Keep only canonical chromosomes (``chr*`` without underscores).
        overwrite : bool, default True
            Overwrite existing store.
        resume : bool, default False
            Resume an existing store.
        sample_column : str, default "sample_id"
            Column in ``metadata`` matching sample names.
        """
        cxreport_files = [Path(f) for f in cxreport_files]
        if sample_names is None:
            sample_names = [f.name.split(".")[0] for f in cxreport_files]
        if len(sample_names) != len(cxreport_files):
            raise ValueError("sample_names length must match cxreport_files length")

        store = cls(
            store_path=store_path,
            sample_names=sample_names,
            overwrite=overwrite,
            resume=resume,
            mc_hmc_split=True,
        )

        logger.info("Reading CXreport files...")
        all_file_data: list[dict[str, pd.DataFrame]] = []
        for path in cxreport_files:
            logger.info(f"  {path.name}")
            all_file_data.append(_read_cxreport(path, filter_chromosomes=filter_chromosomes))

        all_chroms: set[str] = set()
        for fd in all_file_data:
            all_chroms.update(fd.keys())

        logger.info(f"Building union positions and writing store ({len(all_chroms)} chroms)...")
        for chrom in sorted(all_chroms):
            chrom_samples = [(i, fd[chrom]) for i, fd in enumerate(all_file_data) if chrom in fd]
            all_positions = np.unique(
                np.concatenate([df["start"].values for _, df in chrom_samples])
            )
            store._init_chrom_arrays(chrom, all_positions)

            pos_to_idx: dict[int, int] = {int(p): i for i, p in enumerate(all_positions)}

            for sample_idx, df in chrom_samples:
                indices = np.array([pos_to_idx[int(p)] for p in df["start"].values])
                store.root[chrom]["methylation_pct"][sample_idx, indices] = df[
                    "methylation_pct"
                ].values
                store.root[chrom]["n_mc"][sample_idx, indices] = df["n_mc"].values
                store.root[chrom]["n_hmc"][sample_idx, indices] = df["n_hmc"].values
                store.root[chrom]["n_c"][sample_idx, indices] = df["n_c"].values

            logger.info(f"  {chrom}: {len(all_positions)} CpG sites")

        store.meta["completed"][:] = True
        store.root.attrs["chromosomes"] = sorted(all_chroms)

        if metadata is not None:
            if isinstance(metadata, (str, Path)):
                metadata_df = pd.read_csv(metadata)
            else:
                metadata_df = metadata
            store.set_metadata(metadata_df, sample_column=sample_column)

        return store

    @classmethod
    def from_split_cxreport_files(
        cls,
        mc_files: list[str | Path] | None = None,
        hmc_files: list[str | Path] | None = None,
        store_path: Path | str = "methylation.zarr",
        sample_names: list[str] | None = None,
        metadata: pd.DataFrame | Path | str | None = None,
        *,
        filter_chromosomes: bool = True,
        overwrite: bool = True,
        resume: bool = False,
        sample_column: str = "sample_id",
    ) -> "MethylStore":
        """
        Create a MethylStore from 7-column split CXreport files.

        At least one of ``mc_files`` or ``hmc_files`` must be provided. When
        both are given they must be the same length and aligned per sample.
        This format is produced by seqnado and newer biomodal pipeline outputs
        (``*.num_mc_cxreport.txt.gz`` / ``*.num_hmc_cxreport.txt.gz``).

        Parameters
        ----------
        mc_files : list of str or Path, optional
            Paths to ``num_mc_cxreport.txt.gz`` files, one per sample.
        hmc_files : list of str or Path, optional
            Paths to ``num_hmc_cxreport.txt.gz`` files, same order as mc_files.
        store_path : Path or str
            Output Zarr store path.
        sample_names : list of str, optional
            Sample names. Defaults to filename prefix up to first ``.``.
        metadata : DataFrame, Path, or str, optional
            Sample metadata CSV to attach.
        filter_chromosomes : bool, default True
            Keep only canonical chromosomes (``chr*`` without underscores).
        overwrite : bool, default True
            Overwrite existing store.
        resume : bool, default False
            Resume an existing store.
        sample_column : str, default "sample_id"
            Column in ``metadata`` matching sample names.
        """
        mc_files = [Path(f) for f in (mc_files or [])]
        hmc_files = [Path(f) for f in (hmc_files or [])]
        if not mc_files and not hmc_files:
            raise ValueError("Provide at least one of mc_files or hmc_files")
        if mc_files and hmc_files and len(mc_files) != len(hmc_files):
            raise ValueError("mc_files and hmc_files must have the same length when both provided")

        # Determine sample count and default names from whichever list is non-empty
        ref_files = mc_files if mc_files else hmc_files
        n_samples = len(ref_files)
        if sample_names is None:
            sample_names = [f.name.split(".")[0] for f in ref_files]
        if len(sample_names) != n_samples:
            raise ValueError("sample_names length must match number of samples")

        store = cls(
            store_path=store_path,
            sample_names=sample_names,
            overwrite=overwrite,
            resume=resume,
            mc_hmc_split=True,
        )

        logger.info("Reading split CXreport files...")
        all_file_data: list[dict[str, pd.DataFrame]] = []
        mc_iter = mc_files if mc_files else [None] * n_samples
        hmc_iter = hmc_files if hmc_files else [None] * n_samples
        for mc_path, hmc_path in zip(mc_iter, hmc_iter):
            label = " + ".join(p.name for p in [mc_path, hmc_path] if p is not None)
            logger.info(f"  {label}")
            all_file_data.append(
                _read_split_cxreport(mc_path, hmc_path, filter_chromosomes=filter_chromosomes)
            )

        all_chroms: set[str] = set()
        for fd in all_file_data:
            all_chroms.update(fd.keys())

        logger.info(f"Building union positions and writing store ({len(all_chroms)} chroms)...")
        for chrom in sorted(all_chroms):
            chrom_samples = [(i, fd[chrom]) for i, fd in enumerate(all_file_data) if chrom in fd]
            all_positions = np.unique(
                np.concatenate([df["start"].values for _, df in chrom_samples])
            )
            store._init_chrom_arrays(chrom, all_positions)

            pos_to_idx: dict[int, int] = {int(p): i for i, p in enumerate(all_positions)}

            for sample_idx, df in chrom_samples:
                indices = np.array([pos_to_idx[int(p)] for p in df["start"].values])
                store.root[chrom]["methylation_pct"][sample_idx, indices] = df[
                    "methylation_pct"
                ].values
                store.root[chrom]["n_mc"][sample_idx, indices] = df["n_mc"].values
                store.root[chrom]["n_hmc"][sample_idx, indices] = df["n_hmc"].values
                store.root[chrom]["n_c"][sample_idx, indices] = df["n_c"].values

            logger.info(f"  {chrom}: {len(all_positions)} CpG sites")

        store.meta["completed"][:] = True
        store.root.attrs["chromosomes"] = sorted(all_chroms)

        if metadata is not None:
            if isinstance(metadata, (str, Path)):
                metadata_df = pd.read_csv(metadata)
            else:
                metadata_df = metadata
            store.set_metadata(metadata_df, sample_column=sample_column)

        return store

    @classmethod
    def from_mixed_files(
        cls,
        methyldackel_files: list[str | Path] | None = None,
        mc_files: list[str | Path] | None = None,
        hmc_files: list[str | Path] | None = None,
        store_path: Path | str = "methylation.zarr",
        methyldackel_sample_names: list[str] | None = None,
        mc_hmc_sample_names: list[str] | None = None,
        metadata: pd.DataFrame | Path | str | None = None,
        *,
        filter_chromosomes: bool = True,
        overwrite: bool = True,
        resume: bool = False,
        sample_column: str = "sample_id",
    ) -> "MethylStore":
        """
        Create a MethylStore combining bedGraph (e.g. TAPS) and split CXreport
        (e.g. evoC) samples into a single store using the extended mC/hmC schema.

        Bedgraph samples are treated as undifferentiated modC:
        ``n_mc = n_methylated``, ``n_hmc = 0``, ``n_c = n_unmethylated``.
        """
        methyldackel_files = [Path(f) for f in (methyldackel_files or [])]
        mc_files = [Path(f) for f in (mc_files or [])]
        hmc_files = [Path(f) for f in (hmc_files or [])]

        if not methyldackel_files and not mc_files and not hmc_files:
            raise ValueError("Provide methyldackel_files and/or mc_files/hmc_files")
        if mc_files and hmc_files and len(mc_files) != len(hmc_files):
            raise ValueError("mc_files and hmc_files must have the same length when both provided")

        if methyldackel_sample_names is None:
            methyldackel_sample_names = [f.stem for f in methyldackel_files]
        # Default sample names from whichever cx list is non-empty
        ref_cx = mc_files if mc_files else hmc_files
        if mc_hmc_sample_names is None:
            mc_hmc_sample_names = [f.name.split(".")[0] for f in ref_cx]

        all_sample_names = list(methyldackel_sample_names) + list(mc_hmc_sample_names)
        if len(set(all_sample_names)) != len(all_sample_names):
            raise ValueError("Duplicate sample names across bedgraph and CXreport files")

        store = cls(
            store_path=store_path,
            sample_names=all_sample_names,
            overwrite=overwrite,
            resume=resume,
            mc_hmc_split=True,
        )

        # Read all files
        logger.info("Reading bedGraph files (as undifferentiated modC)...")
        bg_data: list[dict[str, pd.DataFrame]] = []
        for path in methyldackel_files:
            logger.info(f"  {path.name}")
            raw = _read_bedgraph(path, filter_chromosomes=filter_chromosomes)
            # Convert to extended schema: n_mc = n_methylated, n_hmc = 0, n_c = n_unmethylated
            converted: dict[str, pd.DataFrame] = {}
            for chrom, df in raw.items():
                ext = pd.DataFrame(
                    {
                        "start": df["start"],
                        "n_mc": df["n_methylated"],
                        "n_hmc": np.zeros(len(df), dtype=np.uint16),
                        "n_c": df["n_unmethylated"],
                        "methylation_pct": df["methylation_pct"],
                    }
                )
                converted[chrom] = ext
            bg_data.append(converted)

        logger.info("Reading split CXreport files...")
        cx_data: list[dict[str, pd.DataFrame]] = []
        n_cx = len(ref_cx)
        mc_iter = mc_files if mc_files else [None] * n_cx
        hmc_iter = hmc_files if hmc_files else [None] * n_cx
        for mc_path, hmc_path in zip(mc_iter, hmc_iter):
            label = " + ".join(p.name for p in [mc_path, hmc_path] if p is not None)
            logger.info(f"  {label}")
            cx_data.append(
                _read_split_cxreport(mc_path, hmc_path, filter_chromosomes=filter_chromosomes)
            )

        # bg samples are indices 0..n_bg-1; cx samples are n_bg..n_bg+n_cx-1
        n_bg = len(bg_data)
        all_file_data = [(i, fd) for i, fd in enumerate(bg_data)] + [
            (n_bg + i, fd) for i, fd in enumerate(cx_data)
        ]

        all_chroms: set[str] = set()
        for _, fd in all_file_data:
            all_chroms.update(fd.keys())

        logger.info(f"Building union positions and writing store ({len(all_chroms)} chroms)...")
        for chrom in sorted(all_chroms):
            chrom_samples = [(idx, fd[chrom]) for idx, fd in all_file_data if chrom in fd]
            all_positions = np.unique(
                np.concatenate([df["start"].values for _, df in chrom_samples])
            )
            store._init_chrom_arrays(chrom, all_positions)
            pos_to_idx = {int(p): i for i, p in enumerate(all_positions)}

            for sample_idx, df in chrom_samples:
                indices = np.array([pos_to_idx[int(p)] for p in df["start"].values])
                store.root[chrom]["methylation_pct"][sample_idx, indices] = df[
                    "methylation_pct"
                ].values
                store.root[chrom]["n_mc"][sample_idx, indices] = df["n_mc"].values
                store.root[chrom]["n_hmc"][sample_idx, indices] = df["n_hmc"].values
                store.root[chrom]["n_c"][sample_idx, indices] = df["n_c"].values

            logger.info(f"  {chrom}: {len(all_positions)} CpG sites")

        store.meta["completed"][:] = True
        store.root.attrs["chromosomes"] = sorted(all_chroms)

        if metadata is not None:
            if isinstance(metadata, (str, Path)):
                metadata_df = pd.read_csv(metadata)
            else:
                metadata_df = metadata
            store.set_metadata(metadata_df, sample_column=sample_column)

        return store

    # ---- Metadata ----

    def set_metadata(self, metadata: pd.DataFrame, sample_column: str = "sample_id") -> None:
        """Store metadata columns from a DataFrame."""
        self._check_writable()
        if sample_column not in metadata.columns:
            raise ValueError(f"Sample column '{sample_column}' not found in metadata")
        meta_subset = metadata.copy()
        meta_subset[sample_column] = meta_subset[sample_column].astype(str)
        meta_subset = meta_subset.set_index(sample_column)
        for col in meta_subset.columns:
            values = _to_str_list(
                meta_subset[col].reindex(self.sample_names, fill_value="").tolist()
            )
            self.root.attrs[f"metadata_{col}"] = values
            logger.info(f"Updated metadata column: {col}")
    # ---- Data access ----

    def get_positions(self, chrom: str) -> np.ndarray:
        """Return CpG positions (0-based start) for a chromosome."""
        return self.root[chrom]["positions"][:]

    def to_xarray(
        self,
        chromosomes: list[str] | None = None,
        variable: str = "methylation_pct",
    ) -> dict[str, xr.DataArray]:
        """
        Extract data as per-chromosome Xarray DataArrays (lazy dask-backed).

        Parameters
        ----------
        chromosomes : list[str], optional
            Chromosomes to extract. Defaults to all.
        variable : str, default "methylation_pct"
            Which array to extract: ``"methylation_pct"``, ``"n_methylated"``,
            or ``"n_unmethylated"``.

        Returns
        -------
        dict[str, xr.DataArray]
            Each DataArray has dims ``(sample, position)`` where ``position``
            holds the actual genomic coordinates (0-based CpG starts).
        """
        if self.has_mc_hmc_split:
            valid = {"methylation_pct", "n_mc", "n_hmc", "n_c"}
        else:
            valid = {"methylation_pct", "n_methylated", "n_unmethylated"}
        if variable not in valid:
            raise ValueError(f"variable must be one of {valid}, got {variable!r}")

        chroms = chromosomes if chromosomes is not None else self.chromosomes
        invalid = set(chroms) - set(self.chromosomes)
        if invalid:
            raise ValueError(f"Chromosomes not in store: {invalid}")

        metadata_df = self.get_metadata()
        result: dict[str, xr.DataArray] = {}
        for chrom in chroms:
            positions = self.get_positions(chrom)
            zarr_arr = self.root[chrom][variable]
            dask_arr = da.from_array(zarr_arr, chunks=(1, DEFAULT_CHUNK_LEN))

            coords: dict = {"sample": self.sample_names, "position": positions}
            for col in metadata_df.columns:
                if col != "sample_id":
                    coords[col] = ("sample", metadata_df[col].values)

            result[chrom] = xr.DataArray(
                dask_arr,
                dims=("sample", "position"),
                coords=coords,
                attrs={"variable": variable, "chromosome": chrom},
            )
        return result

    def count_features(
        self,
        *,
        gtf_file=None,
        bed_file=None,
        ranges_df=None,
        feature_type: str = "gene",
        feature_id_col: str | list[str] | None = None,
        strand: str | None = None,
        integerize: bool = False,
    ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
        """
        Aggregate methylation data over genomic features.

        For each feature, finds all CpG sites within it and computes a full set
        of methylation statistics across those sites per sample.

        Parameters
        ----------
        gtf_file : str or Path, optional
            Path to GTF file for feature extraction.
        bed_file : str or Path, optional
            Path to BED file with genomic ranges.
        ranges_df : DataFrame, optional
            Pre-parsed genomic ranges with contig/start/end columns.
        feature_type : str, default "gene"
            GTF feature type to extract (e.g., "gene", "transcript", "exon").
        feature_id_col : str or list[str], optional
            Column(s) to use as row index of the output DataFrames.
            For GTF inputs defaults to the first available of: gene_id, transcript_id,
            gene_name, transcript_name.
        strand : str, optional
            If "+" or "-", restrict to features on that strand.
        integerize : bool, default False
            If True, round n_methylated/n_unmethylated/n_cpg_covered to int64.

        Returns
        -------
        stats : dict[str, DataFrame]
            Dictionary of features × samples DataFrames:

            - ``"n_methylated"``     — sum of methylated reads across CpG sites
            - ``"n_unmethylated"``   — sum of unmethylated reads across CpG sites
            - ``"n_cpg_covered"``    — number of CpG sites with any read coverage
            - ``"methylation_ratio"``— n_methylated / (n_methylated + n_unmethylated);
                                       NaN where total depth is zero
            - ``"methylation_pct"``  — mean methylation_pct across covered CpG sites;
                                       NaN where no site is covered
        feature_metadata : DataFrame
            Per-feature metadata (contig, strand, start, end, range_length,
            n_cpg_total — total CpG sites regardless of coverage).
        """
        from quantnado.analysis.features import extract_feature_ranges, load_gtf

        # ── Resolve ranges ────────────────────────────────────────────────────
        if ranges_df is None and bed_file is None:
            if gtf_file is None:
                raise TypeError("Provide ranges_df, bed_file, or gtf_file")
            gtf_source = load_gtf(gtf_file, feature_types=None)
            ranges_df = pd.DataFrame(extract_feature_ranges(gtf_source, feature_type=feature_type))
            ranges_df = ranges_df.rename(
                columns={"Chromosome": "contig", "Start": "start", "End": "end"}
            )
            if feature_id_col is None:
                for candidate in ("gene_id", "transcript_id", "gene_name", "transcript_name"):
                    if candidate in ranges_df.columns:
                        feature_id_col = candidate
                        break
        elif bed_file is not None:
            ranges_df = pd.read_csv(
                bed_file,
                sep="\t",
                header=None,
                usecols=[0, 1, 2],
                names=["contig", "start", "end"],
            )

        ranges_df = ranges_df.reset_index(drop=True)

        # Normalize PyRanges Strand column
        if "Strand" in ranges_df.columns and "strand" not in ranges_df.columns:
            ranges_df = ranges_df.rename(columns={"Strand": "strand"})
        if strand is not None and "strand" in ranges_df.columns:
            ranges_df = ranges_df[ranges_df["strand"] == strand].reset_index(drop=True)

        contig_col = next(
            (c for c in ("contig", "Chromosome", "chrom") if c in ranges_df.columns), None
        )
        start_col = next((c for c in ("start", "Start") if c in ranges_df.columns), "start")
        end_col = next((c for c in ("end", "End") if c in ranges_df.columns), "end")

        # ── Pre-load all three variables per chromosome ───────────────────────
        # methylation_pct has NaN where no coverage; counts have fill=0
        chrom_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        if contig_col is not None:
            for chrom in ranges_df[contig_col].unique():
                if chrom in self.chromosomes:
                    if self.has_mc_hmc_split:
                        n_meth = self.root[chrom]["n_mc"][:].astype(np.float64) + self.root[chrom][
                            "n_hmc"
                        ][:].astype(np.float64)
                        n_unmeth = self.root[chrom]["n_c"][:].astype(np.float64)
                    else:
                        n_meth = self.root[chrom]["n_methylated"][:].astype(np.float64)
                        n_unmeth = self.root[chrom]["n_unmethylated"][:].astype(np.float64)
                    chrom_data[chrom] = (
                        self.get_positions(chrom),
                        n_meth,
                        n_unmeth,
                        self.root[chrom]["methylation_pct"][:].astype(np.float64),
                    )

        # ── Aggregate per feature ─────────────────────────────────────────────
        rows_meth = []
        rows_unmeth = []
        rows_covered = []
        rows_ratio = []
        rows_pct = []
        n_cpg_total = []

        for _, row in ranges_df.iterrows():
            chrom = row[contig_col] if contig_col else None
            feat_start, feat_end = int(row[start_col]), int(row[end_col])

            if chrom is None or chrom not in chrom_data:
                rows_meth.append(np.zeros(self.n_samples))
                rows_unmeth.append(np.zeros(self.n_samples))
                rows_covered.append(np.zeros(self.n_samples))
                rows_ratio.append(np.full(self.n_samples, np.nan))
                rows_pct.append(np.full(self.n_samples, np.nan))
                n_cpg_total.append(0)
                continue

            positions, n_meth, n_unmeth, meth_pct = chrom_data[chrom]
            lo = int(np.searchsorted(positions, feat_start, side="left"))
            hi = int(np.searchsorted(positions, feat_end, side="left"))
            n_cpg_total.append(hi - lo)

            if lo == hi:
                rows_meth.append(np.zeros(self.n_samples))
                rows_unmeth.append(np.zeros(self.n_samples))
                rows_covered.append(np.zeros(self.n_samples))
                rows_ratio.append(np.full(self.n_samples, np.nan))
                rows_pct.append(np.full(self.n_samples, np.nan))
                continue

            s_meth = n_meth[:, lo:hi]  # (n_samples, n_sites)
            s_unmeth = n_unmeth[:, lo:hi]
            s_pct = meth_pct[:, lo:hi]

            sum_meth = s_meth.sum(axis=1)
            sum_unmeth = s_unmeth.sum(axis=1)
            total_reads = sum_meth + sum_unmeth
            covered = (~np.isnan(s_pct)).sum(axis=1).astype(np.float64)

            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = np.where(total_reads > 0, sum_meth / total_reads, np.nan)
                mean_pct = np.nanmean(s_pct, axis=1)
                mean_pct = np.where(covered > 0, mean_pct, np.nan)

            rows_meth.append(sum_meth)
            rows_unmeth.append(sum_unmeth)
            rows_covered.append(covered)
            rows_ratio.append(ratio)
            rows_pct.append(mean_pct)

        idx = ranges_df.index

        def _df(rows):
            return pd.DataFrame(rows, columns=self.sample_names, index=idx)

        stats = {
            "n_methylated": _df(rows_meth),
            "n_unmethylated": _df(rows_unmeth),
            "n_cpg_covered": _df(rows_covered),
            "methylation_ratio": _df(rows_ratio),
            "methylation_pct": _df(rows_pct),
        }

        # ── Build feature metadata ────────────────────────────────────────────
        feature_metadata = pd.DataFrame(
            {
                "start": ranges_df[start_col].values,
                "end": ranges_df[end_col].values,
                "range_length": (ranges_df[end_col] - ranges_df[start_col]).values,
                "n_cpg_total": n_cpg_total,
            },
            index=idx,
        )
        if contig_col is not None:
            feature_metadata.insert(0, "contig", ranges_df[contig_col].values)
        if "strand" in ranges_df.columns:
            feature_metadata.insert(
                feature_metadata.columns.get_loc("start"), "strand", ranges_df["strand"].values
            )

        # Apply feature_id_col as shared index across all outputs
        id_cols = [feature_id_col] if isinstance(feature_id_col, str) else (feature_id_col or [])
        if id_cols and all(c in ranges_df.columns for c in id_cols):
            for col in reversed(id_cols):
                if col not in feature_metadata.columns:
                    feature_metadata.insert(0, col, ranges_df[col].values)
            new_index = (
                feature_metadata[id_cols[0]].values
                if len(id_cols) == 1
                else pd.MultiIndex.from_frame(feature_metadata[id_cols])
            )
            for df in stats.values():
                df.index = new_index
            feature_metadata.index = new_index

        if integerize:
            for key in ("n_methylated", "n_unmethylated", "n_cpg_covered"):
                stats[key] = stats[key].round().astype("int64")

        return stats, feature_metadata

    def extract_region(
        self,
        region: str | None = None,
        chrom: str | None = None,
        start: int | None = None,
        end: int | None = None,
        variable: str = "methylation_pct",
        samples: list[str] | list[int] | None = None,
        as_xarray: bool = True,
    ) -> xr.DataArray | np.ndarray:
        """
        Extract methylation data for a genomic region.

        Parameters
        ----------
        region : str, optional
            Region string, e.g. ``"chr21:5000000-6000000"``.
        chrom, start, end : optional
            Alternative to ``region``. ``start``/``end`` are 0-based half-open.
        variable : str, default "methylation_pct"
            Which variable to return.
        samples : list, optional
            Sample names or integer indices. Defaults to all.
        as_xarray : bool, default True
            Return an xr.DataArray; if False return np.ndarray.
        """
        from ..utils import parse_genomic_region

        if region is not None and chrom is not None:
            raise ValueError("Specify either 'region' or 'chrom', not both")
        if region is not None:
            chrom, parsed_start, parsed_end = parse_genomic_region(region)
            if parsed_start is not None:
                start = parsed_start
            if parsed_end is not None:
                end = parsed_end
        if chrom is None:
            raise ValueError("Must specify either 'region' or 'chrom'")
        if chrom not in self.chromosomes:
            raise ValueError(f"Chromosome '{chrom}' not in store. Available: {self.chromosomes}")

        positions = self.get_positions(chrom)
        mask = np.ones(len(positions), dtype=bool)
        if start is not None:
            mask &= positions >= start
        if end is not None:
            mask &= positions < end
        pos_indices = np.where(mask)[0]
        region_positions = positions[pos_indices]

        if samples is None:
            sample_indices = np.arange(self.n_samples)
            sample_names_out = list(self.sample_names)
        else:
            sample_indices_list = []
            sample_names_out = []
            for s in samples:
                if isinstance(s, str):
                    if s not in self.sample_names:
                        raise ValueError(f"Sample '{s}' not found in store")
                    idx = self.sample_names.index(s)
                else:
                    idx = int(s)
                sample_indices_list.append(idx)
                sample_names_out.append(self.sample_names[idx])
            sample_indices = np.array(sample_indices_list)

        data = self.root[chrom][variable][np.ix_(sample_indices, pos_indices)]

        if not as_xarray:
            return np.array(data)

        metadata_df = self.get_metadata()
        metadata_subset = metadata_df.iloc[sample_indices]
        coords: dict = {"sample": sample_names_out, "position": region_positions}
        for col in metadata_subset.columns:
            if col != "sample_id":
                coords[col] = ("sample", np.asarray(metadata_subset[col]))

        return xr.DataArray(
            da.from_array(data, chunks=(1, -1)),
            dims=("sample", "position"),
            coords=coords,
            attrs={
                "variable": variable,
                "chromosome": chrom,
                "start": start,
                "end": end,
            },
        )

    def extract(
        self,
        intervals_path: "str | None" = None,
        ranges_df: "pd.DataFrame | None" = None,
        feature_type: "str | None" = None,
        gtf_path: "str | None" = None,
        variable: str = "methylation_pct",
        upstream: "int | None" = None,
        downstream: "int | None" = None,
        fixed_width: "int | None" = None,
        anchor: str = "midpoint",
        bin_size: "int | None" = 50,
        samples: "list[str] | None" = None,
    ) -> xr.DataArray:
        """
        Extract methylation signal over genomic intervals, binned into fixed windows.

        Bins sparse CpG methylation values into fixed-width windows around an anchor
        point (TSS, midpoint, etc.) and returns a ``(interval, bin, sample)`` DataArray
        compatible with :func:`metaplot` and :func:`tornadoplot`.

        Parameters
        ----------
        intervals_path : str, optional
            Path to a BED or GTF file with genomic intervals.
        ranges_df : DataFrame, optional
            Pre-loaded ranges with columns Chromosome, Start, End (and optionally Strand).
        feature_type : str, optional
            Feature type to extract from GTF: ``"gene"``, ``"transcript"``, ``"promoter"``.
            Requires ``gtf_path``.
        gtf_path : str, optional
            Path to GTF file (used with ``feature_type``).
        variable : str, default "methylation_pct"
            Which methylation variable to bin (``"methylation_pct"``, ``"n_methylated"``,
            ``"n_unmethylated"``).
        upstream : int, optional
            Bases upstream of anchor to include. Cannot be combined with ``fixed_width``.
        downstream : int, optional
            Bases downstream of anchor to include. Cannot be combined with ``fixed_width``.
        fixed_width : int, optional
            Symmetric window total width. Cannot be combined with ``upstream``/``downstream``.
        anchor : str, default "midpoint"
            Anchor point: ``"midpoint"``, ``"start"`` (5' end, strand-aware),
            or ``"end"`` (3' end, strand-aware).
        bin_size : int, optional, default 50
            Bin size in bp. Bins CpG sites into windows of this size.
            ``None`` returns one value per base position (slow for large windows).
        samples : list of str, optional
            Subset of sample names to include. Defaults to all samples.

        Returns
        -------
        xr.DataArray
            Dimensions ``(interval, bin, sample)``. Position coordinate contains bp
            offsets from the anchor (negative = upstream). Bins with no CpG sites
            are ``NaN``.

        Examples
        --------
        >>> binned_meth = meth.extract(
        ...     feature_type="transcript",
        ...     gtf_path="genes.gtf",
        ...     upstream=1000,
        ...     downstream=1000,
        ...     anchor="start",
        ...     bin_size=50,
        ... )
        >>> ax = metaplot(binned_meth, modality="methylation", title="CpG methylation at TSS")
        """
        from quantnado.analysis.reduce import _log_chromosome_overlap, _resolve_ranges
        from quantnado.dataset.enums import AnchorPoint

        # Resolve window
        if upstream is not None or downstream is not None:
            if fixed_width is not None:
                raise ValueError("Cannot specify both fixed_width and upstream/downstream")
            _upstream = upstream if upstream is not None else 0
            _downstream = downstream if downstream is not None else 0
        elif fixed_width is not None:
            _upstream = fixed_width // 2
            _downstream = fixed_width - _upstream
        else:
            raise ValueError("Must specify upstream/downstream or fixed_width")

        _total_width = _upstream + _downstream
        if bin_size is not None and _total_width % bin_size != 0:
            raise ValueError(
                f"Total window ({_total_width}) must be divisible by bin_size ({bin_size})"
            )
        n_bins = _total_width // bin_size if bin_size is not None else _total_width

        # Resolve ranges and samples
        ranges_df, start_col, end_col, contig_col = _resolve_ranges(
            ranges_df, intervals_path, feature_type, gtf_path, "Start", "End", "Chromosome"
        )
        if samples is None:
            sample_indices = np.arange(self.n_samples)
            sample_names_out = list(self.sample_names)
        else:
            sample_indices_list, sample_names_out = [], []
            for s in samples:
                idx = self.sample_names.index(s) if isinstance(s, str) else int(s)
                sample_indices_list.append(idx)
                sample_names_out.append(self.sample_names[idx])
            sample_indices = np.array(sample_indices_list)
        n_samples_out = len(sample_indices)

        _log_chromosome_overlap(
            set(ranges_df[contig_col].unique()), set(self.chromosomes), "intervals"
        )

        anchor_enum = AnchorPoint(anchor) if isinstance(anchor, str) else anchor
        has_strand = "Strand" in ranges_df.columns

        all_matrices: list[np.ndarray] = []
        meta_starts, meta_ends, meta_contigs, meta_strands = [], [], [], []

        for chrom, group in ranges_df.groupby(contig_col, observed=True):
            if chrom not in self.chromosomes:
                continue

            cpg_pos = self.get_positions(chrom)  # (n_cpg,)
            n_cpg = len(cpg_pos)
            meth_data = np.array(
                self.root[chrom][variable][np.ix_(sample_indices, np.arange(n_cpg))]
            )  # (n_samples, n_cpg)

            starts = np.asarray(group[start_col], dtype=np.int64)
            ends = np.asarray(group[end_col], dtype=np.int64)
            strands = np.asarray(group["Strand"], dtype=object) if has_strand else None

            if anchor_enum == AnchorPoint.MIDPOINT:
                anchor_pos = (starts + ends) // 2
            elif anchor_enum == AnchorPoint.START:
                anchor_pos = np.where(strands == "-", ends, starts) if has_strand else starts
            else:  # END
                anchor_pos = np.where(strands == "-", starts, ends) if has_strand else ends

            win_starts = anchor_pos - _upstream
            win_ends = anchor_pos + _downstream

            for i in range(len(starts)):
                ws, we = int(win_starts[i]), int(win_ends[i])
                strand = strands[i] if has_strand else "+"
                cpg_mask = (cpg_pos >= ws) & (cpg_pos < we)
                cpg_idx = np.where(cpg_mask)[0]

                mat = np.full((n_bins, n_samples_out), np.nan, dtype=np.float32)
                if cpg_idx.size > 0:
                    rel = cpg_pos[cpg_idx] - ws  # [0, _total_width)
                    if has_strand and strand == "-":
                        rel = _total_width - 1 - rel
                    bins = (rel * n_bins // _total_width).clip(0, n_bins - 1)
                    vals = meth_data[:, cpg_idx]  # (n_samples, n_cpg_in_window)
                    for s_idx in range(n_samples_out):
                        sv = vals[s_idx]
                        valid = ~np.isnan(sv)
                        if valid.any():
                            b_valid = bins[valid]
                            sv_valid = sv[valid]
                            sums = np.bincount(b_valid, weights=sv_valid, minlength=n_bins)
                            counts = np.bincount(b_valid, minlength=n_bins)
                            mat[:, s_idx] = np.where(counts > 0, sums / counts, np.nan)

                all_matrices.append(mat)
                meta_starts.append(int(starts[i]))
                meta_ends.append(int(ends[i]))
                meta_contigs.append(chrom)
                meta_strands.append(str(strand))

        if not all_matrices:
            raise ValueError("No intervals found on chromosomes present in methylation store")

        data_arr = np.stack(all_matrices, axis=0)  # (n_intervals, n_bins, n_samples)

        if bin_size is not None:
            pos_values = np.arange(n_bins, dtype=np.int64) * bin_size - _upstream
            pos_dim = "bin"
        else:
            pos_values = np.arange(-_upstream, _downstream, dtype=np.int64)
            pos_dim = "relative_position"

        coords: dict = {
            pos_dim: pos_values,
            "sample": sample_names_out,
            "start": ("interval", np.array(meta_starts)),
            "end": ("interval", np.array(meta_ends)),
            "contig": ("interval", np.array(meta_contigs)),
        }
        if has_strand:
            coords["strand"] = ("interval", np.array(meta_strands))

        return xr.DataArray(
            data_arr,
            dims=("interval", pos_dim, "sample"),
            coords=coords,
            attrs={
                "variable": variable,
                "upstream": _upstream,
                "downstream": _downstream,
                "bin_size": bin_size,
            },
        )

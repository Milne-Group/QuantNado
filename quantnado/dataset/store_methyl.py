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
from .store_bam import _compute_sample_hash, _to_str_list


def _read_bedgraph(path: Path | str, filter_chromosomes: bool = True) -> dict[str, pd.DataFrame]:
    """Read a CpG bedGraph file from MethylDackel or similar tools.

    Returns dict mapping chromosome -> DataFrame with columns:
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
        raise ValueError(f"bedGraph file {path.name} has only {n_cols} columns; expected at least 4")

    df = df.iloc[:, :6]
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
    """Read a biomodal evoC CXreport file.

    Returns dict mapping chromosome -> DataFrame with columns:
        start (int64, 0-based), n_mc (uint16), n_hmc (uint16), n_c (uint16), methylation_pct (float32)
    """
    _cols = ["chrom", "pos", "strand", "mod_level", "n_mod", "n_not_mod", "context"]
    _dtypes = {
        "chrom": str, "pos": np.int64, "strand": str, "mod_level": str,
        "n_mod": np.int32, "n_not_mod": np.int32, "context": str,
    }
    df = pd.read_csv(Path(path), sep="\t", header=None, names=_cols, dtype=_dtypes, comment="#")

    if filter_chromosomes:
        df = df[df["chrom"].str.startswith("chr") & ~df["chrom"].str.contains("_")]

    df["canonical_pos"] = np.where(df["strand"] == "+", df["pos"], df["pos"] - 1)
    df["is_mc"] = df["mod_level"].str.lower().isin(["c", "mc", "5mc", "5mc_5hmc"])
    df["is_hmc"] = df["mod_level"].str.lower().isin(["hmc", "5hmc", "5mc_5hmc"])

    agg = (
        df.groupby(["chrom", "canonical_pos"], sort=True)
        .apply(lambda g: pd.Series({
            "n_mc": int(g.loc[g["is_mc"], "n_mod"].sum()),
            "n_hmc": int(g.loc[g["is_hmc"], "n_mod"].sum()),
            "total": int(g["n_mod"].sum() + g["n_not_mod"].sum()),
        }))
        .reset_index()
        .rename(columns={"canonical_pos": "start"})
    )
    agg["start"] = agg["start"].astype(np.int64)
    agg["n_c"] = (agg["total"] - agg["n_mc"] - agg["n_hmc"]).clip(lower=0).astype(np.int32)

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
    """Read one or both 7-column split CXreport files and merge into per-CpG DataFrames."""
    if mc_path is None and hmc_path is None:
        raise ValueError("At least one of mc_path or hmc_path must be provided")

    _cols = ["chrom", "pos", "strand", "n_mod", "n_not_mod", "context", "trinuc"]
    _dtypes = {
        "chrom": str, "pos": np.int64, "strand": str,
        "n_mod": np.int32, "n_not_mod": np.int32, "context": str, "trinuc": str,
    }

    if mc_path is not None and hmc_path is not None:
        mc = pd.read_csv(Path(mc_path), sep="\t", header=None, names=_cols, dtype=_dtypes)
        hmc = pd.read_csv(Path(hmc_path), sep="\t", header=None, names=_cols, dtype=_dtypes)
        df = pd.DataFrame({
            "chrom": mc["chrom"], "pos": mc["pos"], "strand": mc["strand"],
            "n_mc": mc["n_mod"].astype(np.int32),
            "n_hmc": hmc["n_mod"].astype(np.int32),
            "total": (mc["n_mod"] + mc["n_not_mod"]).astype(np.int32),
        })
    elif mc_path is not None:
        mc = pd.read_csv(Path(mc_path), sep="\t", header=None, names=_cols, dtype=_dtypes)
        df = pd.DataFrame({
            "chrom": mc["chrom"], "pos": mc["pos"], "strand": mc["strand"],
            "n_mc": mc["n_mod"].astype(np.int32),
            "n_hmc": np.zeros(len(mc), dtype=np.int32),
            "total": (mc["n_mod"] + mc["n_not_mod"]).astype(np.int32),
        })
    else:
        hmc = pd.read_csv(Path(hmc_path), sep="\t", header=None, names=_cols, dtype=_dtypes)
        df = pd.DataFrame({
            "chrom": hmc["chrom"], "pos": hmc["pos"], "strand": hmc["strand"],
            "n_mc": np.zeros(len(hmc), dtype=np.int32),
            "n_hmc": hmc["n_mod"].astype(np.int32),
            "total": (hmc["n_mod"] + hmc["n_not_mod"]).astype(np.int32),
        })

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
    Zarr-backed methylation store using a flat sparse layout.

    All CpG sites across all chromosomes are stored in a single flat array,
    sorted by chromosome then position. Chromosome offsets are stored in
    ``root.attrs["contig_offsets"]`` for O(1) chromosome lookup.

    Store layout::

        root/
        ├── contig          (n_sites,)              uint8  — index into contig_list
        ├── position        (n_sites,)              uint32 — 0-based
        ├── methylation_pct (n_sites, n_samples)    float32  NaN where no coverage
        ├── n_methylated    (n_sites, n_samples)    uint16
        └── n_total         (n_sites, n_samples)    uint16
        └── metadata/
            └── completed   (n_samples,)            bool

        root.attrs:
            sample_names, n_samples, store_type, has_mc_hmc_split,
            contig_list (list of chromosome names, index matches `contig` array),
            contig_offsets (dict: chrom -> [start_row, end_row])

    For mc/hmC split stores, ``n_methylated`` stores mC counts, and additional
    ``n_hmc`` and ``n_c`` arrays replace the default schema.

    Example
    -------
    >>> store = MethylStore.from_bedgraph_files(
    ...     methyldackel_files=["s1.bedGraph", "s2.bedGraph"],
    ...     store_path="methylation.zarr",
    ... )
    >>> region = store.extract_region("chr1:1000000-2000000")
    """

    CHUNK_LEN = 65536

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
        self._mc_hmc_split_init = mc_hmc_split

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
                raise FileNotFoundError(f"Store does not exist at {self.store_path} (read_only=True)")
            self._init_store()

    @classmethod
    def open(cls, store_path: str | Path, read_only: bool = True) -> "MethylStore":
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
            raise RuntimeError("Store is in read-only mode. Reopen with read_only=False to allow modifications.")

    def _init_store(self) -> None:
        store = LocalStore(str(self.store_path))
        self.root = zarr.group(store=store, overwrite=True, zarr_format=3)
        self.meta = self.root.create_group("metadata")
        self.meta.create_array("completed", shape=(self.n_samples,), dtype=bool, fill_value=False, overwrite=True)
        self.root.attrs.update({
            "sample_names": self.sample_names,
            "sample_names_hash": self.sample_hash,
            "n_samples": self.n_samples,
            "store_type": "methylation",
            "has_mc_hmc_split": self._mc_hmc_split_init,
            "metadata_data_type": ["methylation"] * self.n_samples,
        })
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
        return list(self.root.attrs.get("contig_list", []))

    @property
    def has_mc_hmc_split(self) -> bool:
        return bool(self.root.attrs.get("has_mc_hmc_split", False))

    def _write_flat_store(
        self,
        all_file_data: list[dict[str, pd.DataFrame]],
        mc_hmc_split: bool,
    ) -> None:
        """Build flat sparse arrays from per-sample per-chromosome DataFrames.

        Parameters
        ----------
        all_file_data : list of dict[chrom -> DataFrame]
            One dict per sample. For bedgraph: columns start, methylation_pct,
            n_methylated, n_unmethylated. For cxreport: columns start, n_mc,
            n_hmc, n_c, methylation_pct.
        mc_hmc_split : bool
            If True, create n_mc/n_hmc/n_c arrays instead of n_methylated/n_unmethylated.
        """
        # Collect all chromosomes across all samples (sorted)
        all_chroms: list[str] = sorted(
            {chrom for fd in all_file_data for chrom in fd.keys()}
        )

        # Build union of positions per chromosome, then concatenate into flat arrays
        contig_list: list[str] = []
        contig_offsets: dict[str, list[int]] = {}
        all_contig_idx: list[np.ndarray] = []
        all_positions: list[np.ndarray] = []

        for chrom_idx, chrom in enumerate(all_chroms):
            chrom_positions = np.unique(
                np.concatenate([
                    fd[chrom]["start"].values
                    for fd in all_file_data
                    if chrom in fd
                ])
            ).astype(np.uint32)

            start_row = sum(len(a) for a in all_positions)
            end_row = start_row + len(chrom_positions)
            contig_list.append(chrom)
            contig_offsets[chrom] = [start_row, end_row]
            all_contig_idx.append(np.full(len(chrom_positions), chrom_idx, dtype=np.uint8))
            all_positions.append(chrom_positions)

        n_sites = sum(len(a) for a in all_positions)
        contig_arr = np.concatenate(all_contig_idx)
        position_arr = np.concatenate(all_positions)

        chunk = min(self.CHUNK_LEN, max(1, n_sites))

        # Create flat coordinate arrays
        self.root.create_array(
            "contig", shape=(n_sites,), dtype=np.uint8, fill_value=0, overwrite=True,
            chunks=(chunk,), compressors=[self.compressor],
        )
        self.root.create_array(
            "position", shape=(n_sites,), dtype=np.uint32, fill_value=0, overwrite=True,
            chunks=(chunk,), compressors=[self.compressor],
        )
        self.root["contig"][:] = contig_arr
        self.root["position"][:] = position_arr

        # Create data arrays
        if mc_hmc_split:
            data_arrays = [
                ("methylation_pct", np.float32, np.nan),
                ("n_mc", np.uint16, 0),
                ("n_hmc", np.uint16, 0),
                ("n_c", np.uint16, 0),
            ]
        else:
            data_arrays = [
                ("methylation_pct", np.float32, np.nan),
                ("n_methylated", np.uint16, 0),
                ("n_total", np.uint16, 0),
            ]

        for name, dtype, fill in data_arrays:
            self.root.create_array(
                name,
                shape=(n_sites, self.n_samples),
                chunks=(chunk, self.n_samples),
                dtype=dtype,
                compressors=[self.compressor],
                fill_value=fill,
                overwrite=True,
            )

        # Fill in per-sample data
        for sample_idx, fd in enumerate(all_file_data):
            for chrom in all_chroms:
                if chrom not in fd:
                    continue
                df = fd[chrom]
                row_start, row_end = contig_offsets[chrom]
                # positions for this chrom in the flat array
                chrom_positions = position_arr[row_start:row_end]
                pos_to_flat = {int(p): row_start + i for i, p in enumerate(chrom_positions)}

                flat_indices = np.array([pos_to_flat[int(p)] for p in df["start"].values])

                self.root["methylation_pct"][flat_indices, sample_idx] = df["methylation_pct"].values

                if mc_hmc_split:
                    self.root["n_mc"][flat_indices, sample_idx] = df["n_mc"].values
                    self.root["n_hmc"][flat_indices, sample_idx] = df["n_hmc"].values
                    self.root["n_c"][flat_indices, sample_idx] = df["n_c"].values
                else:
                    self.root["n_methylated"][flat_indices, sample_idx] = df["n_methylated"].values
                    # n_total = n_methylated + n_unmethylated
                    n_total = (df["n_methylated"].values.astype(np.uint32) +
                               df["n_unmethylated"].values.astype(np.uint32)).clip(0, 65535).astype(np.uint16)
                    self.root["n_total"][flat_indices, sample_idx] = n_total

        # Store coordinate metadata
        self.root.attrs.update({
            "contig_list": contig_list,
            "contig_offsets": contig_offsets,
            "n_sites": n_sites,
            "has_mc_hmc_split": mc_hmc_split,
            "chromosomes": contig_list,
        })
        self.meta["completed"][:] = True
        logger.info(f"Wrote {n_sites} sites across {len(contig_list)} chromosomes")

    # ── Public factory methods ──────────────────────────────────────────────

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
        """Create a MethylStore from MethylDackel bedGraph files."""
        methyldackel_files = [Path(f) for f in methyldackel_files]
        if sample_names is None:
            sample_names = [f.stem for f in methyldackel_files]
        if len(sample_names) != len(methyldackel_files):
            raise ValueError("sample_names length must match methyldackel_files length")

        store = cls(store_path=store_path, sample_names=sample_names, overwrite=overwrite, resume=resume)

        logger.info("Reading bedGraph files...")
        all_file_data: list[dict[str, pd.DataFrame]] = []
        for path in methyldackel_files:
            logger.info(f"  {path.name}")
            all_file_data.append(_read_bedgraph(path, filter_chromosomes=filter_chromosomes))

        store._write_flat_store(all_file_data, mc_hmc_split=False)

        if metadata is not None:
            if isinstance(metadata, (str, Path)):
                metadata = pd.read_csv(metadata)
            store.set_metadata(metadata, sample_column=sample_column)

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
        """Create a MethylStore from biomodal evoC CXreport files."""
        cxreport_files = [Path(f) for f in cxreport_files]
        if sample_names is None:
            sample_names = [f.name.split(".")[0] for f in cxreport_files]
        if len(sample_names) != len(cxreport_files):
            raise ValueError("sample_names length must match cxreport_files length")

        store = cls(store_path=store_path, sample_names=sample_names, overwrite=overwrite, resume=resume, mc_hmc_split=True)

        logger.info("Reading CXreport files...")
        all_file_data: list[dict[str, pd.DataFrame]] = []
        for path in cxreport_files:
            logger.info(f"  {path.name}")
            all_file_data.append(_read_cxreport(path, filter_chromosomes=filter_chromosomes))

        store._write_flat_store(all_file_data, mc_hmc_split=True)

        if metadata is not None:
            if isinstance(metadata, (str, Path)):
                metadata = pd.read_csv(metadata)
            store.set_metadata(metadata, sample_column=sample_column)

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
        """Create a MethylStore from 7-column split CXreport files (mC/hmC)."""
        mc_files = [Path(f) for f in (mc_files or [])]
        hmc_files = [Path(f) for f in (hmc_files or [])]
        if not mc_files and not hmc_files:
            raise ValueError("Provide at least one of mc_files or hmc_files")
        if mc_files and hmc_files and len(mc_files) != len(hmc_files):
            raise ValueError("mc_files and hmc_files must have the same length when both provided")

        ref_files = mc_files if mc_files else hmc_files
        n_samples = len(ref_files)
        if sample_names is None:
            sample_names = [f.name.split(".")[0] for f in ref_files]
        if len(sample_names) != n_samples:
            raise ValueError("sample_names length must match number of samples")

        store = cls(store_path=store_path, sample_names=sample_names, overwrite=overwrite, resume=resume, mc_hmc_split=True)

        logger.info("Reading split CXreport files...")
        all_file_data: list[dict[str, pd.DataFrame]] = []
        mc_iter = mc_files if mc_files else [None] * n_samples
        hmc_iter = hmc_files if hmc_files else [None] * n_samples
        for mc_path, hmc_path in zip(mc_iter, hmc_iter):
            label = " + ".join(p.name for p in [mc_path, hmc_path] if p is not None)
            logger.info(f"  {label}")
            all_file_data.append(_read_split_cxreport(mc_path, hmc_path, filter_chromosomes=filter_chromosomes))

        store._write_flat_store(all_file_data, mc_hmc_split=True)

        if metadata is not None:
            if isinstance(metadata, (str, Path)):
                metadata = pd.read_csv(metadata)
            store.set_metadata(metadata, sample_column=sample_column)

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
        """Create a MethylStore combining bedGraph and split CXreport samples."""
        methyldackel_files = [Path(f) for f in (methyldackel_files or [])]
        mc_files = [Path(f) for f in (mc_files or [])]
        hmc_files = [Path(f) for f in (hmc_files or [])]

        if not methyldackel_files and not mc_files and not hmc_files:
            raise ValueError("Provide methyldackel_files and/or mc_files/hmc_files")
        if mc_files and hmc_files and len(mc_files) != len(hmc_files):
            raise ValueError("mc_files and hmc_files must have the same length when both provided")

        if methyldackel_sample_names is None:
            methyldackel_sample_names = [f.stem for f in methyldackel_files]
        ref_cx = mc_files if mc_files else hmc_files
        if mc_hmc_sample_names is None:
            mc_hmc_sample_names = [f.name.split(".")[0] for f in ref_cx]

        all_sample_names = list(methyldackel_sample_names) + list(mc_hmc_sample_names)
        if len(set(all_sample_names)) != len(all_sample_names):
            raise ValueError("Duplicate sample names across bedgraph and CXreport files")

        store = cls(store_path=store_path, sample_names=all_sample_names, overwrite=overwrite, resume=resume, mc_hmc_split=True)

        logger.info("Reading bedGraph files (as undifferentiated modC)...")
        bg_data: list[dict[str, pd.DataFrame]] = []
        for path in methyldackel_files:
            logger.info(f"  {path.name}")
            raw = _read_bedgraph(path, filter_chromosomes=filter_chromosomes)
            converted: dict[str, pd.DataFrame] = {}
            for chrom, df in raw.items():
                converted[chrom] = pd.DataFrame({
                    "start": df["start"],
                    "n_mc": df["n_methylated"],
                    "n_hmc": np.zeros(len(df), dtype=np.uint16),
                    "n_c": df["n_unmethylated"],
                    "methylation_pct": df["methylation_pct"],
                })
            bg_data.append(converted)

        logger.info("Reading split CXreport files...")
        cx_data: list[dict[str, pd.DataFrame]] = []
        n_cx = len(ref_cx)
        mc_iter = mc_files if mc_files else [None] * n_cx
        hmc_iter = hmc_files if hmc_files else [None] * n_cx
        for mc_path, hmc_path in zip(mc_iter, hmc_iter):
            label = " + ".join(p.name for p in [mc_path, hmc_path] if p is not None)
            logger.info(f"  {label}")
            cx_data.append(_read_split_cxreport(mc_path, hmc_path, filter_chromosomes=filter_chromosomes))

        store._write_flat_store(bg_data + cx_data, mc_hmc_split=True)

        if metadata is not None:
            if isinstance(metadata, (str, Path)):
                metadata = pd.read_csv(metadata)
            store.set_metadata(metadata, sample_column=sample_column)

        return store

    # ── Metadata ────────────────────────────────────────────────────────────

    def set_metadata(self, metadata: pd.DataFrame, sample_column: str = "sample_id") -> None:
        self._check_writable()
        if sample_column not in metadata.columns:
            raise ValueError(f"Sample column '{sample_column}' not found in metadata")
        meta_subset = metadata.copy()
        meta_subset[sample_column] = meta_subset[sample_column].astype(str)
        meta_subset = meta_subset.set_index(sample_column)
        for col in meta_subset.columns:
            values = _to_str_list(meta_subset[col].reindex(self.sample_names, fill_value="").tolist())
            self.root.attrs[f"metadata_{col}"] = values

    # ── Data access ──────────────────────────────────────────────────────────

    def _contig_row_range(self, chrom: str) -> tuple[int, int]:
        """Return (start_row, end_row) for a chromosome in the flat arrays."""
        offsets = self.root.attrs.get("contig_offsets", {})
        if chrom not in offsets:
            raise ValueError(f"Chromosome '{chrom}' not in store. Available: {self.chromosomes}")
        start, end = offsets[chrom]
        return int(start), int(end)

    def get_positions(self, chrom: str) -> np.ndarray:
        """Return CpG positions (0-based) for a chromosome."""
        start, end = self._contig_row_range(chrom)
        return self.root["position"][start:end]

    def to_xarray(
        self,
        chromosomes: list[str] | None = None,
        variable: str = "methylation_pct",
    ) -> dict[str, xr.DataArray]:
        """Extract data as per-chromosome lazy Xarray DataArrays.

        Parameters
        ----------
        chromosomes : list[str], optional
            Chromosomes to extract. Defaults to all.
        variable : str, default "methylation_pct"
            Which array to extract.

        Returns
        -------
        dict[str, xr.DataArray]
            Each DataArray has dims ``(sample, position)`` with genomic coordinates.
        """
        if self.has_mc_hmc_split:
            valid = {"methylation_pct", "n_mc", "n_hmc", "n_c"}
        else:
            valid = {"methylation_pct", "n_methylated", "n_total"}
        if variable not in valid:
            raise ValueError(f"variable must be one of {valid}, got {variable!r}")

        chroms = chromosomes if chromosomes is not None else self.chromosomes
        invalid = set(chroms) - set(self.chromosomes)
        if invalid:
            raise ValueError(f"Chromosomes not in store: {invalid}")

        metadata_df = self.get_metadata()
        result: dict[str, xr.DataArray] = {}
        for chrom in chroms:
            start_row, end_row = self._contig_row_range(chrom)
            positions = self.root["position"][start_row:end_row]
            zarr_arr = self.root[variable]
            # Slice the flat array for this chrom: shape (n_sites_chrom, n_samples)
            dask_arr = da.from_zarr(zarr_arr)[start_row:end_row, :]
            # Transpose to (sample, position)
            dask_arr = dask_arr.T

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

    def extract_region(
        self,
        region: str | None = None,
        chrom: str | None = None,
        start: int | None = None,
        end: int | None = None,
        variable: str = "methylation_pct",
        as_xarray: bool = True,
        samples: list[str] | list[int] | None = None,
    ) -> xr.DataArray | np.ndarray:
        """Extract methylation data for a genomic region.

        Returns array with dims ``(sample, position)`` where position holds
        actual genomic coordinates of CpG sites within the region.
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

        # Resolve sample indices
        if samples is None:
            sample_indices = list(range(self.n_samples))
            sample_names_out = self.sample_names
        else:
            sample_indices = []
            sample_names_out = []
            for s in samples:
                if isinstance(s, str):
                    if s not in self._sample_name_to_idx:
                        raise ValueError(f"Sample '{s}' not found in store")
                    idx = self._sample_name_to_idx[s]
                else:
                    idx = int(s)
                sample_indices.append(idx)
                sample_names_out.append(self.sample_names[idx])

        row_start, row_end = self._contig_row_range(chrom)
        positions = self.root["position"][row_start:row_end]

        # Filter to requested position range
        if start is not None or end is not None:
            mask = np.ones(len(positions), dtype=bool)
            if start is not None:
                mask &= positions >= start
            if end is not None:
                mask &= positions < end
            indices = np.where(mask)[0]
            positions = positions[indices]
            flat_indices = row_start + indices
        else:
            flat_indices = np.arange(row_start, row_end)

        if len(flat_indices) == 0:
            empty = np.full((len(sample_indices), 0), np.nan)
            if not as_xarray:
                return empty
            return xr.DataArray(empty, dims=("sample", "position"),
                                coords={"sample": sample_names_out, "position": positions})

        data = self.root[variable][flat_indices, :][:, sample_indices]  # (n_sites, n_sel)
        data_T = data.T  # (n_sel, n_sites)

        if not as_xarray:
            return data_T

        return xr.DataArray(
            data_T,
            dims=("sample", "position"),
            coords={"sample": sample_names_out, "position": positions},
            attrs={"variable": variable, "chromosome": chrom},
        )

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
        """Aggregate methylation over genomic features."""
        from quantnado.analysis.features import extract_feature_ranges, load_gtf

        if ranges_df is None and bed_file is None:
            if gtf_file is None:
                raise TypeError("Provide ranges_df, bed_file, or gtf_file")
            gtf_source = load_gtf(gtf_file, feature_types=None)
            ranges_df = pd.DataFrame(extract_feature_ranges(gtf_source, feature_type=feature_type))
            ranges_df = ranges_df.rename(columns={"Chromosome": "contig", "Start": "start", "End": "end"})
            if feature_id_col is None:
                for candidate in ("gene_id", "transcript_id", "gene_name", "transcript_name"):
                    if candidate in ranges_df.columns:
                        feature_id_col = candidate
                        break
        elif bed_file is not None:
            ranges_df = pd.read_csv(bed_file, sep="\t", header=None, usecols=[0, 1, 2],
                                    names=["contig", "start", "end"])

        ranges_df = ranges_df.reset_index(drop=True)

        if "Strand" in ranges_df.columns and "strand" not in ranges_df.columns:
            ranges_df = ranges_df.rename(columns={"Strand": "strand"})
        if strand is not None and "strand" in ranges_df.columns:
            ranges_df = ranges_df[ranges_df["strand"] == strand].reset_index(drop=True)

        contig_col = next((c for c in ("contig", "Chromosome", "chrom") if c in ranges_df.columns), None)
        start_col = next((c for c in ("start", "Start") if c in ranges_df.columns), "start")
        end_col = next((c for c in ("end", "End") if c in ranges_df.columns), "end")

        results: dict[str, list] = {
            "n_methylated": [], "n_unmethylated": [], "n_cpg_covered": [],
            "methylation_ratio": [], "methylation_pct": [], "n_cpg_total": [],
        }
        feature_meta_rows: list[dict] = []
        feature_ids: list = list(range(len(ranges_df)))

        if feature_id_col is not None:
            if isinstance(feature_id_col, list):
                feature_ids = [
                    "|".join(str(ranges_df.loc[i, c]) for c in feature_id_col)
                    for i in range(len(ranges_df))
                ]
            else:
                feature_ids = ranges_df[feature_id_col].tolist()

        for _, row in ranges_df.iterrows():
            chrom = str(row[contig_col]) if contig_col else ""
            feat_start = int(row[start_col])
            feat_end = int(row[end_col])

            if chrom not in self.chromosomes:
                for key in results:
                    results[key].append([np.nan] * self.n_samples)
                feature_meta_rows.append({
                    "contig": chrom, "start": feat_start, "end": feat_end,
                    "strand": row.get("strand", ".") if "strand" in row.index else ".",
                    "range_length": feat_end - feat_start, "n_cpg_total": 0,
                })
                continue

            xr_data = self.extract_region(chrom=chrom, start=feat_start, end=feat_end, variable="methylation_pct")
            xr_nm = self.extract_region(chrom=chrom, start=feat_start, end=feat_end, variable="n_methylated" if not self.has_mc_hmc_split else "n_mc")
            xr_nu = self.extract_region(chrom=chrom, start=feat_start, end=feat_end, variable="n_total" if not self.has_mc_hmc_split else "n_c")

            n_cpg = xr_data.shape[1] if hasattr(xr_data, "shape") else 0
            feature_meta_rows.append({
                "contig": chrom, "start": feat_start, "end": feat_end,
                "strand": row.get("strand", ".") if "strand" in row.index else ".",
                "range_length": feat_end - feat_start, "n_cpg_total": n_cpg,
            })

            if n_cpg == 0:
                for key in results:
                    results[key].append([np.nan] * self.n_samples)
                continue

            nm_arr = np.asarray(xr_nm)  # (n_samples, n_sites)
            nu_arr = np.asarray(xr_nu)
            pct_arr = np.asarray(xr_data)

            n_m_sum = np.nansum(nm_arr, axis=1)
            n_u_sum = np.nansum(nu_arr, axis=1)
            covered = np.sum(~np.isnan(pct_arr), axis=1).astype(float)
            total_depth = n_m_sum + n_u_sum
            ratio = np.where(total_depth > 0, n_m_sum / total_depth, np.nan)
            mean_pct = np.where(covered > 0, np.nanmean(pct_arr, axis=1), np.nan)

            if integerize:
                n_m_sum = n_m_sum.astype(np.int64)
                n_u_sum = n_u_sum.astype(np.int64)
                covered = covered.astype(np.int64)

            results["n_methylated"].append(n_m_sum.tolist())
            results["n_unmethylated"].append(n_u_sum.tolist())
            results["n_cpg_covered"].append(covered.tolist())
            results["methylation_ratio"].append(ratio.tolist())
            results["methylation_pct"].append(mean_pct.tolist())
            results["n_cpg_total"].append(n_cpg)

        feature_metadata = pd.DataFrame(feature_meta_rows, index=feature_ids)
        n_cpg_total_col = feature_metadata.pop("n_cpg_total")

        stats: dict[str, pd.DataFrame] = {}
        for key in ("n_methylated", "n_unmethylated", "n_cpg_covered", "methylation_ratio", "methylation_pct"):
            stats[key] = pd.DataFrame(results[key], index=feature_ids, columns=self.sample_names)
        feature_metadata["n_cpg_total"] = n_cpg_total_col

        return stats, feature_metadata

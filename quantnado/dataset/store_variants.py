from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import re
import shutil
import warnings

import numpy as np
import pandas as pd
import zarr
from zarr.storage import LocalStore
from zarr.codecs import BloscCodec
from loguru import logger
import xarray as xr
import dask.array as da

from .core import BaseStore
from .utils import _compute_sample_hash
from quantnado.utils import estimate_chunk_len, is_network_fs

# Genotype encoding
GT_MISSING: np.int8 = np.int8(-1)
GT_HOM_REF: np.int8 = np.int8(0)
GT_HET: np.int8 = np.int8(1)
GT_HOM_ALT: np.int8 = np.int8(2)


@contextmanager
def _suppress_allel_warnings():
    """Suppress scikit-allel warnings (malformed VCF headers, etc.)."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        yield


def _callset_to_chrom_dfs(
    callset: dict[str, np.ndarray] | None,
    filter_chromosomes: bool = True,
) -> dict[str, pd.DataFrame]:
    if callset is None:
        return {}

    chroms: np.ndarray = callset["variants/CHROM"]
    pos: np.ndarray = callset["variants/POS"]
    refs: np.ndarray = callset["variants/REF"]
    alts: np.ndarray = callset["variants/ALT"]
    qual: np.ndarray = callset["variants/QUAL"]
    gt: np.ndarray | None = callset.get("calldata/GT")
    ad: np.ndarray | None = callset.get("calldata/AD")
    dp: np.ndarray | None = callset.get("variants/DP")
    mq: np.ndarray | None = callset.get("variants/MQ")
    is_indel: np.ndarray | None = callset.get("variants/INDEL")
    variant_id: np.ndarray | None = callset.get("variants/ID")

    if filter_chromosomes:
        mask = np.array([c.startswith("chr") and "_" not in c for c in chroms])
        chroms = chroms[mask]
        pos = pos[mask]
        refs = refs[mask]
        alts = alts[mask]
        qual = qual[mask]
        if gt is not None:
            gt = gt[mask]
        if ad is not None:
            ad = ad[mask]
        if dp is not None:
            dp = dp[mask]
        if mq is not None:
            mq = mq[mask]
        if is_indel is not None:
            is_indel = is_indel[mask]
        if variant_id is not None:
            variant_id = variant_id[mask]

    n = len(pos)
    if n == 0:
        return {}

    if gt is not None:
        # allel may return (n_variants, n_samples, ploidy) or (n_variants, ploidy) for single-sample VCFs
        gt0 = gt[:, 0, :] if gt.ndim == 3 else gt
        missing = np.any(gt0 < 0, axis=1)
        genotype = np.clip((gt0 > 0).sum(axis=1), 0, 2).astype(np.int8)
        genotype[missing] = GT_MISSING
    else:
        genotype = np.full(n, GT_MISSING, dtype=np.int8)

    if ad is not None:
        # allel may return (n_variants, n_samples, 2) or (n_variants, 2) for single-sample VCFs
        ad0 = ad[:, 0, :] if ad.ndim == 3 else ad
        ad_ref = np.where(ad0[:, 0] >= 0, ad0[:, 0], 0).clip(0, 65535).astype(np.uint16)
        ad_alt = np.where(ad0[:, 1] >= 0, ad0[:, 1], 0).clip(0, 65535).astype(np.uint16)
    else:
        ad_ref = np.zeros(n, dtype=np.uint16)
        ad_alt = np.zeros(n, dtype=np.uint16)

    dp_vals = (
        np.where(dp >= 0, dp, 0).clip(0, 65535).astype(np.uint16)
        if dp is not None
        else np.zeros(n, dtype=np.uint16)
    )
    mq_vals = (
        np.where(mq >= 0, mq, 0).clip(0, 255).astype(np.uint8)
        if mq is not None
        else np.zeros(n, dtype=np.uint8)
    )
    indel_vals = is_indel.astype(bool) if is_indel is not None else np.zeros(n, dtype=bool)
    id_vals = (
        np.where(variant_id != "", variant_id, ".")
        if variant_id is not None
        else np.full(n, ".", dtype=object)
    )

    df = pd.DataFrame(
        {
            "chrom": chroms,
            "pos": pos.astype(np.int64),
            "ref": refs,
            "alt": alts[:, 0] if alts.ndim == 2 else alts,
            "qual": qual.astype(np.float32),
            "genotype": genotype,
            "ad_ref": ad_ref,
            "ad_alt": ad_alt,
            "dp": dp_vals,
            "mq": mq_vals,
            "is_indel": indel_vals,
            "variant_id": id_vals,
        }
    )
    return {chrom: grp.reset_index(drop=True) for chrom, grp in df.groupby("chrom")}


def _extract_chromsizes_from_headers(headers: list[str]) -> dict[str, int]:
    """Extract chromosome sizes from VCF header lines.

    Parses lines like: ##contig=<ID=chr1,length=248956422>
    """
    chromsizes = {}
    for line in headers:
        if line.startswith("##contig="):
            match = re.search(r"ID=([^,>]+).*length=(\d+)", line)
            if match:
                chrom_id = match.group(1)
                length = int(match.group(2))
                chromsizes[chrom_id] = length
    return chromsizes


def _read_vcf(
    path: Path | str,
    filter_chromosomes: bool = True,
) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    """Read variants from a single-sample VCF/VCF.gz file using scikit-allel.

    Requests all known optional fields; allel silently omits any that are
    absent from the VCF so no header pre-scan is needed.

    Returns
    -------
    tuple
        (dict mapping chromosome -> DataFrame, dict mapping chromosome -> chromsize from header)
        DataFrame columns: pos (int64, 1-based), ref (str), alt (str), qual (float32),
        genotype (int8: -1 missing, 0 hom_ref, 1 het, 2 hom_alt),
        ad_ref (uint16), ad_alt (uint16), dp (uint16), mq (uint8), is_indel (bool), variant_id (str)
    """
    try:
        import allel
    except ImportError as e:
        raise ImportError(
            "scikit-allel is required to read VCF files: pip install scikit-allel"
        ) from e

    path = Path(path)

    # Extract chromsizes from VCF header using allel
    header_chromsizes: dict[str, int] = {}
    try:
        with _suppress_allel_warnings():
            header = allel.read_vcf_headers(input=str(path))
        header_chromsizes = _extract_chromsizes_from_headers(header.headers)
    except Exception as e:
        logger.warning(f"Could not read chromsizes from {path.name}: {e}")

    fields = [
        "CHROM",
        "POS",
        "REF",
        "ALT",
        "QUAL",
        "ID",
        "variants/DP",
        "variants/MQ",
        "variants/INDEL",
        "calldata/GT",
        "calldata/AD",
    ]
    numbers = {"ALT": 1, "AD": 2}
    with _suppress_allel_warnings():
        callset = allel.read_vcf(str(path), fields=fields, numbers=numbers)

    chrom_dfs = _callset_to_chrom_dfs(callset, filter_chromosomes=filter_chromosomes)
    return chrom_dfs, header_chromsizes


class VariantStore(BaseStore):
    """
    Zarr-backed SNP/variant store using a flat sparse layout.

    All variants across all chromosomes are stored in flat arrays sorted by
    chromosome then position. Chromosome offsets are stored in
    ``root.attrs["contig_offsets"]`` for O(1) chromosome lookup.

    Store layout::

        root/
        ├── contig          (n_variants,)              uint8  — index into contig_list
        ├── position        (n_variants,)              int64  — 1-based genomic position
        ├── ref_alleles     (n_variants,)              str    — reference allele strings
        ├── alt_alleles     (n_variants,)              str    — alt allele strings
        ├── genotype        (n_variants, n_samples)    int8   -1=missing,0=hom_ref,1=het,2=hom_alt
        ├── allele_depth_ref (n_variants, n_samples)   uint16
        ├── allele_depth_alt (n_variants, n_samples)   uint16
        └── qual            (n_variants, n_samples)    float32
        └── metadata/
            └── completed   (n_samples,)               bool

        root.attrs:
            sample_names, n_samples, store_type,
            contig_list (list of chromosome names),
            contig_offsets (dict: chrom -> [start_row, end_row])
            chromsizes (dict: chrom -> chromsize)

    Example
    -------
    >>> store = VariantStore.from_vcf_files(
    ...     vcf_files=["sample1.vcf.gz", "sample2.vcf.gz"],
    ...     store_path="variants.zarr",
    ... )
    >>> xr_dict = store.to_xarray(variable="genotype")
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
        _shared_root: "zarr.Group | None" = None,
    ) -> None:
        self.path = Path(store_path)
        self.store_path = self._normalize_path(self.path)

        if _shared_root is not None:
            self.sample_names = [str(s) for s in sample_names]
            self._setup_sample_lookup()
            self.completed_mask_raw = np.zeros(len(self.sample_names), dtype=bool)
            self._metadata_cache = None
            self.n_samples = len(self.sample_names)
            self.sample_hash = _compute_sample_hash(self.sample_names)
            self.compressor = BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")
            self.read_only = read_only
            if self.n_samples == 0:
                raise ValueError("sample_names must not be empty")
            self.root = _shared_root
            self.meta = self.root.require_group("metadata")
            self._init_store_in_group()
            return

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
    def open(cls, store_path: str | Path, read_only: bool = True) -> "VariantStore":
        store_path = cls._normalize_path(store_path)
        if not store_path.exists():
            raise FileNotFoundError(f"Store does not exist at {store_path}")
        group = zarr.open_group(str(store_path), mode="r" if read_only else "r+")
        try:
            stored_names = list(group.attrs["sample_names"])
        except KeyError as e:
            raise ValueError(f"Missing required attribute in store: {e}")
        return cls(
            store_path=store_path,
            sample_names=stored_names,
            overwrite=False,
            resume=True,
            read_only=read_only,
        )

    def _init_store(self) -> None:
        store = LocalStore(str(self.store_path))
        self.root = zarr.group(store=store, overwrite=True, zarr_format=3)
        self.meta = self.root.create_group("metadata")
        self.meta.create_array(
            "completed", shape=(self.n_samples,), dtype=bool, fill_value=False, overwrite=True
        )
        self.root.attrs.update(
            {
                "sample_names": self.sample_names,
                "sample_names_hash": self.sample_hash,
                "n_samples": self.n_samples,
                "store_type": "variants",
                "metadata_data_type": ["variants"] * self.n_samples,
            }
        )
        logger.info(f"Initialized VariantStore at {self.store_path}")

    def _init_store_in_group(self) -> None:
        """Initialise variant metadata inside an already-open shared zarr group."""
        self.meta.create_array(
            "completed", shape=(self.n_samples,), dtype=bool, fill_value=False, overwrite=True
        )
        self.root.attrs.update(
            {
                "sample_names": self.sample_names,
                "sample_names_hash": self.sample_hash,
                "n_samples": self.n_samples,
                "store_type": "variants",
                "metadata_data_type": ["variants"] * self.n_samples,
            }
        )
        logger.info(f"Initialized VariantStore (shared root) at {self.store_path}")

    @property
    def completed_mask(self) -> np.ndarray:
        return self.meta["completed"][:].astype(bool)

    @property
    def chromosomes(self) -> list[str]:
        return list(self.root.attrs.get("contig_list", []))

    def _write_flat_store(
        self, all_file_data: list[dict[str, pd.DataFrame]], chromsizes: dict[str, int] | None = None
    ) -> None:
        """Build flat sparse arrays from per-sample per-chromosome DataFrames."""
        n_samples = len(all_file_data)
        single_sample = n_samples == 1

        all_chroms: list[str] = sorted({chrom for fd in all_file_data for chrom in fd.keys()})

        contig_list: list[str] = []
        contig_offsets: dict[str, list[int]] = {}
        calculated_chromsizes: dict[str, int] = {}
        all_contig_idx: list[np.ndarray] = []
        all_positions: list[np.ndarray] = []
        all_refs: list = []
        all_alts: list = []
        all_indels: list = []
        all_ids: list = []

        for chrom_idx, chrom in enumerate(all_chroms):
            if single_sample:
                fd0 = all_file_data[0]
                if chrom not in fd0:
                    continue
                df0 = fd0[chrom]
                chrom_positions = df0["pos"].values.astype(np.int64)
                # Vectorised: no dict, no itertuples — VCF is sorted by position.
                all_refs.extend(df0["ref"].tolist())
                all_alts.extend(df0["alt"].tolist())
                all_indels.extend(df0["is_indel"].tolist())
                all_ids.extend(df0["variant_id"].tolist())
            else:
                chrom_positions = np.unique(
                    np.concatenate([fd[chrom]["pos"].values for fd in all_file_data if chrom in fd])
                ).astype(np.int64)
                # Build ref/alt lookup from first sample per position
                first_sample = next((fd[chrom] for fd in all_file_data if chrom in fd), None)
                if first_sample is not None:
                    pos_to_ref = dict(zip(first_sample["pos"].values.astype(int), first_sample["ref"].astype(str)))
                    pos_to_alt = dict(zip(first_sample["pos"].values.astype(int), first_sample["alt"].astype(str)))
                    pos_to_indel = dict(zip(first_sample["pos"].values.astype(int), first_sample["is_indel"].astype(bool)))
                    pos_to_id = dict(zip(first_sample["pos"].values.astype(int), first_sample["variant_id"].astype(str)))
                    all_refs.extend(pos_to_ref[int(p)] for p in chrom_positions)
                    all_alts.extend(pos_to_alt[int(p)] for p in chrom_positions)
                    all_indels.extend(pos_to_indel[int(p)] for p in chrom_positions)
                    all_ids.extend(pos_to_id[int(p)] for p in chrom_positions)

            start_row = sum(len(a) for a in all_positions)
            end_row = start_row + len(chrom_positions)
            contig_list.append(chrom)
            contig_offsets[chrom] = [start_row, end_row]

            # Use provided chromsizes if available, otherwise calculate from VCF
            if chromsizes and chrom in chromsizes:
                calculated_chromsizes[chrom] = int(chromsizes[chrom])
            else:
                chrom_size = chrom_positions.max() + 1 if len(chrom_positions) > 0 else 0
                calculated_chromsizes[chrom] = int(chrom_size)

            all_contig_idx.append(np.full(len(chrom_positions), chrom_idx, dtype=np.uint8))
            all_positions.append(chrom_positions)

        n_variants = sum(len(a) for a in all_positions)
        contig_arr = np.concatenate(all_contig_idx)
        position_arr = np.concatenate(all_positions)

        chunk = min(
            estimate_chunk_len(
                total_positions=n_variants,
                dtype_bytes=4,
                n_samples=1,
                fs_is_network=is_network_fs(str(self.store_path)),
            )["chunk_len"],
            max(1, n_variants),
        )

        self.root.create_array(
            "contig",
            shape=(n_variants,),
            dtype=np.uint8,
            fill_value=0,
            overwrite=True,
            chunks=(chunk,),
            compressors=[self.compressor],
        )
        self.root.create_array(
            "position",
            shape=(n_variants,),
            dtype=np.int64,
            fill_value=0,
            overwrite=True,
            chunks=(chunk,),
            compressors=[self.compressor],
        )
        self.root["contig"][:] = contig_arr
        self.root["position"][:] = position_arr

        for name, dtype, fill in [
            ("genotype", np.int8, -1),
            ("allele_depth_ref", np.uint16, 0),
            ("allele_depth_alt", np.uint16, 0),
            ("coverage", np.uint16, 0),
            ("mapping_quality", np.uint8, 0),
            ("qual", np.float32, np.nan),
        ]:
            self.root.create_array(
                name,
                shape=(n_variants, self.n_samples),
                chunks=(chunk, self.n_samples),
                dtype=dtype,
                compressors=[self.compressor],
                fill_value=fill,
                overwrite=True,
            )

        # Fill per-sample data. Build each column in memory then write once —
        # avoids scatter-write RMW cycles that dominate Zarr async overhead.
        for sample_idx, fd in enumerate(all_file_data):
            gt_col = np.full(n_variants, -1, dtype=np.int8)
            adr_col = np.zeros(n_variants, dtype=np.uint16)
            ada_col = np.zeros(n_variants, dtype=np.uint16)
            dp_col = np.zeros(n_variants, dtype=np.uint16)
            mq_col = np.zeros(n_variants, dtype=np.uint8)
            qual_col = np.full(n_variants, np.nan, dtype=np.float32)

            for chrom in all_chroms:
                if chrom not in fd:
                    continue
                df = fd[chrom]
                row_start, row_end = contig_offsets[chrom]

                if single_sample:
                    indices = slice(row_start, row_end)
                else:
                    chrom_positions = position_arr[row_start:row_end]
                    indices = row_start + np.searchsorted(
                        chrom_positions, df["pos"].values.astype(np.int64)
                    )

                gt_col[indices] = df["genotype"].values.astype(np.int8)
                adr_col[indices] = df["ad_ref"].values.clip(0, 65535).astype(np.uint16)
                ada_col[indices] = df["ad_alt"].values.clip(0, 65535).astype(np.uint16)
                dp_col[indices] = df["dp"].values.clip(0, 65535).astype(np.uint16)
                mq_col[indices] = df["mq"].values.clip(0, 255).astype(np.uint8)
                qual_col[indices] = df["qual"].values.astype(np.float32)

            self.root["genotype"][:, sample_idx] = gt_col
            self.root["allele_depth_ref"][:, sample_idx] = adr_col
            self.root["allele_depth_alt"][:, sample_idx] = ada_col
            self.root["coverage"][:, sample_idx] = dp_col
            self.root["mapping_quality"][:, sample_idx] = mq_col
            self.root["qual"][:, sample_idx] = qual_col

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*does not have a Zarr V3 specification")
            self.root.create_array(
                "ref_alleles",
                shape=(n_variants,),
                dtype="|S12",
                fill_value=b"",
                overwrite=True,
                chunks=(chunk,),
                compressors=[self.compressor],
            )
            self.root.create_array(
                "alt_alleles",
                shape=(n_variants,),
                dtype="|S12",
                fill_value=b"",
                overwrite=True,
                chunks=(chunk,),
                compressors=[self.compressor],
            )
            self.root.create_array(
                "variant_id",
                shape=(n_variants,),
                dtype="|S24",
                fill_value=b".",
                overwrite=True,
                chunks=(chunk,),
                compressors=[self.compressor],
            )
        self.root["ref_alleles"][:] = np.array(all_refs, dtype="|S12")
        self.root["alt_alleles"][:] = np.array(all_alts, dtype="|S12")
        self.root.create_array(
            "is_indel",
            shape=(n_variants,),
            dtype=bool,
            fill_value=False,
            overwrite=True,
            chunks=(chunk,),
        )
        self.root["is_indel"][:] = np.array(all_indels, dtype=bool)
        self.root["variant_id"][:] = np.array(all_ids, dtype="|S24")

        self.root.attrs.update(
            {
                "contig_list": contig_list,
                "contig_offsets": contig_offsets,
                "chromsizes": calculated_chromsizes,
                "n_variants": n_variants,
                "chromosomes": contig_list,
            }
        )
        self.meta["completed"][:] = True
        logger.info(f"Wrote {n_variants} variants across {len(contig_list)} chromosomes")

    @classmethod
    def from_vcf_files(
        cls,
        vcf_files: list[str | Path],
        store_path: Path | str,
        sample_names: list[str] | None = None,
        metadata: pd.DataFrame | Path | str | None = None,
        *,
        filter_chromosomes: bool = True,
        overwrite: bool = True,
        resume: bool = False,
        sample_column: str = "sample_id",
        _shared_root=None,
    ) -> "VariantStore":
        """Create a VariantStore from per-sample VCF.gz files.

        Chromsizes are extracted from VCF header ##contig= lines and required.
        """
        vcf_files = [Path(f) for f in vcf_files]
        if sample_names is None:
            sample_names = [f.name.split(".")[0] for f in vcf_files]
        if len(sample_names) != len(vcf_files):
            raise ValueError("sample_names length must match vcf_files length")

        store = cls(
            store_path=store_path,
            sample_names=sample_names,
            overwrite=overwrite,
            resume=resume,
            _shared_root=_shared_root,
        )

        logger.info("Reading VCF files...")
        all_file_data: list[dict[str, pd.DataFrame]] = []
        reference_chromsizes: dict[str, int] | None = None
        for i, path in enumerate(vcf_files):
            logger.info(f"  {path.name}")
            chrom_dfs, header_chromsizes = _read_vcf(path, filter_chromosomes=filter_chromosomes)
            all_file_data.append(chrom_dfs)
            if i == 0:
                reference_chromsizes = header_chromsizes
            elif header_chromsizes and header_chromsizes != reference_chromsizes:
                logger.warning(f"Chromsize mismatch in {path.name}—using first sample's chromsizes")

        # Verify chromsizes were extracted from first sample
        if not reference_chromsizes:
            raise ValueError(
                "VCF files must contain ##contig= header lines with length information"
            )
        store._write_flat_store(all_file_data, chromsizes=reference_chromsizes)

        if metadata is not None:
            if isinstance(metadata, (str, Path)):
                metadata = pd.read_csv(metadata)
            store.set_metadata(metadata, sample_column=sample_column)

        return store

    # ── Metadata ────────────────────────────────────────────────────────────

    # ── Data access ──────────────────────────────────────────────────────────

    def get_positions(self, chrom: str) -> np.ndarray:
        """Return variant positions (1-based) for a chromosome."""
        start, end = self._contig_row_range(chrom)
        return self.root["position"][start:end]

    def get_alleles(self, chrom: str) -> tuple[list[str], list[str]]:
        """Return (ref, alt) allele lists for a chromosome, aligned with get_positions."""
        start, end = self._contig_row_range(chrom)
        if "ref_alleles" in self.root:
            refs = [
                v.decode() if isinstance(v, bytes) else v
                for v in self.root["ref_alleles"][start:end].tolist()
            ]
            alts = [
                v.decode() if isinstance(v, bytes) else v
                for v in self.root["alt_alleles"][start:end].tolist()
            ]
        else:
            # Backwards compatibility: old stores kept alleles in attrs
            refs = self.root.attrs.get("ref_alleles", [])[start:end]
            alts = self.root.attrs.get("alt_alleles", [])[start:end]
        return list(refs), list(alts)

    def to_xarray(
        self,
        chromosomes: list[str] | None = None,
        variable: str = "genotype",
        sparse: bool = False,
    ) -> dict[str, xr.DataArray]:
        """Extract variant data as per-chromosome lazy Xarray DataArrays.

        Returns DataArrays with dims ``(sample, position)``.

        Parameters
        ----------
        sparse:
            If True, each dask chunk is backed by ``sparse.COO``. Beneficial for
            rare-variant datasets where most (variant, sample) pairs are hom_ref.
        """
        valid = {
            "genotype",
            "allele_depth_ref",
            "allele_depth_alt",
            "coverage",
            "mapping_quality",
            "qual",
        }
        if variable not in valid:
            raise ValueError(f"variable must be one of {valid}, got {variable!r}")

        chroms = chromosomes if chromosomes is not None else self.chromosomes
        invalid = set(chroms) - set(self.chromosomes)
        if invalid:
            raise ValueError(f"Chromosomes not in store: {invalid}")

        metadata_df = self.metadata
        result: dict[str, xr.DataArray] = {}
        for chrom in chroms:
            start_row, end_row = self._contig_row_range(chrom)
            positions = self.root["position"][start_row:end_row]
            zarr_arr = self.root[variable]
            # (n_variants_chrom, n_samples) → transpose to (sample, position)
            dask_arr = da.from_zarr(zarr_arr)[start_row:end_row, :].T
            if sparse:
                import sparse as sp

                dask_arr = dask_arr.map_blocks(sp.COO, dtype=dask_arr.dtype)

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
        variable: str = "genotype",
        samples: list[str] | list[int] | None = None,
        as_xarray: bool = True,
    ) -> xr.DataArray | np.ndarray:
        """Extract variant data for a genomic region.

        Coordinates are 1-based (VCF convention), end is inclusive.
        Returns array with dims ``(sample, position)``.
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

        row_start, row_end = self._contig_row_range(chrom)
        positions = self.root["position"][row_start:row_end]

        # Binary search on sorted positions array (1-based, end inclusive).
        lo = int(np.searchsorted(positions, start, side="left")) if start is not None else 0
        hi = (
            int(np.searchsorted(positions, end, side="right"))
            if end is not None
            else len(positions)
        )
        pos_indices = np.arange(lo, hi)
        region_positions = positions[lo:hi]
        flat_indices = row_start + pos_indices

        if samples is None:
            sample_indices = np.arange(self.n_samples)
            sample_names_out = list(self.sample_names)
        else:
            sample_indices_list = []
            sample_names_out = []
            for s in samples:
                if isinstance(s, str):
                    if s not in self._sample_name_to_idx:
                        raise ValueError(f"Sample '{s}' not found in store")
                    idx = self._sample_name_to_idx[s]
                else:
                    idx = int(s)
                sample_indices_list.append(idx)
                sample_names_out.append(self.sample_names[idx])
            sample_indices = np.array(sample_indices_list)

        # (n_variants_region, n_sel_samples) → transpose to (n_sel_samples, n_variants_region)
        data = self.root[variable][np.ix_(flat_indices, sample_indices)].T

        if not as_xarray:
            return np.array(data)

        metadata_df = self.metadata
        metadata_subset = metadata_df.iloc[sample_indices]
        coords: dict = {"sample": sample_names_out, "position": region_positions}
        for col in metadata_subset.columns:
            if col != "sample_id":
                coords[col] = ("sample", np.asarray(metadata_subset[col]))

        return xr.DataArray(
            da.from_array(data, chunks=(1, -1)),
            dims=("sample", "position"),
            coords=coords,
            attrs={"variable": variable, "chromosome": chrom, "start": start, "end": end},
        )

from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import zarr
from zarr.storage import LocalStore
from zarr.codecs import BloscCodec
from loguru import logger
import xarray as xr
import dask.array as da

from .core import BaseStore
from .constants import DEFAULT_CHUNK_LEN
from .store_bam import _compute_sample_hash, _to_str_list

# Genotype encoding
GT_MISSING: np.int8 = np.int8(-1)
GT_HOM_REF: np.int8 = np.int8(0)
GT_HET: np.int8 = np.int8(1)
GT_HOM_ALT: np.int8 = np.int8(2)


def _read_vcf(
    path: Path | str,
    filter_chromosomes: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Read variants from a single-sample VCF/VCF.gz file.

    Only the first sample in the VCF is used (the convention for per-sample VCFs).
    Multi-allelic sites are simplified to the first alternate allele.

    Returns
    -------
    dict mapping chromosome name -> DataFrame with columns:
        pos (int64, 1-based), ref (str), alt (str), qual (float32),
        genotype (int8: -1 missing, 0 hom_ref, 1 het, 2 hom_alt),
        ad_ref (int32), ad_alt (int32)
    """
    try:
        import pysam
    except ImportError as e:
        raise ImportError("pysam is required to read VCF files: pip install pysam") from e

    path = Path(path)
    records: list[dict] = []

    with pysam.VariantFile(str(path)) as vcf:
        samples = list(vcf.header.samples)
        if not samples:
            raise ValueError(f"VCF has no samples: {path}")
        sample_name = samples[0]

        for rec in vcf.fetch():
            chrom = rec.chrom
            if filter_chromosomes:
                if not chrom.startswith("chr") or "_" in chrom:
                    continue

            alts = rec.alts or (".",)
            alt = alts[0]

            samp = rec.samples[sample_name]
            gt = samp.get("GT", (None, None))
            if gt is None or None in gt:
                gt_int = GT_MISSING
            else:
                n_alt = sum(1 for a in gt if a is not None and a > 0)
                gt_int = np.int8(min(n_alt, 2))  # cap at 2

            ad = samp.get("AD", None)
            ad_ref = int(ad[0]) if ad and len(ad) > 0 else 0
            ad_alt = int(ad[1]) if ad and len(ad) > 1 else 0

            records.append(
                {
                    "chrom": chrom,
                    "pos": np.int64(rec.pos),  # 1-based
                    "ref": str(rec.ref),
                    "alt": str(alt),
                    "qual": np.float32(rec.qual) if rec.qual is not None else np.float32(np.nan),
                    "genotype": gt_int,
                    "ad_ref": np.int32(ad_ref),
                    "ad_alt": np.int32(ad_alt),
                }
            )

    if not records:
        return {}
    df = pd.DataFrame(records)
    return {chrom: grp.reset_index(drop=True) for chrom, grp in df.groupby("chrom")}


class VariantStore(BaseStore):
    """
    Zarr-backed SNP/variant store from per-sample VCF.gz files.

    Data is stored sparsely - only variant positions are retained.
    Variant positions are unioned across all samples; positions not called
    in a sample are filled with genotype ``-1`` (missing) and depths ``0``.

    Per-chromosome zarr layout::

        <chrom>/
            positions        int64[n_variants]           1-based genomic positions
            genotype         int8[n_samples, n_variants] -1 missing, 0 hom_ref, 1 het, 2 hom_alt
            allele_depth_ref int32[n_samples, n_variants]
            allele_depth_alt int32[n_samples, n_variants]
            qual             float32[n_samples, n_variants]

    Alleles (ref/alt) are stored as lists in each chromosome group's attributes,
    indexed by position order. Retrieve with :meth:`get_alleles`.

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
        """Open an existing VariantStore for reading (default) or writing."""
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
                "store_type": "variants",
            }
        )
        logger.info(f"Initialized VariantStore at {self.store_path}")

    def _load_existing(self) -> None:
        store = LocalStore(str(self.store_path))
        self.root = zarr.open_group(store=store, mode="a")
        self.meta = self.root["metadata"]
        logger.info(f"Resuming existing VariantStore at {self.store_path}")

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

    def _init_chrom_arrays(
        self,
        chrom: str,
        positions: np.ndarray,
        refs: list[str],
        alts: list[str],
    ) -> None:
        """Create zarr arrays for a chromosome. ref/alt stored as group attrs."""
        n_var = len(positions)
        grp = self.root.require_group(chrom)

        grp.create_array(
            name="positions", shape=(n_var,), dtype=np.int64, fill_value=0, overwrite=True
        )
        grp["positions"][:] = positions

        # Store ref/alt strings as JSON-serialisable lists in group attrs
        grp.attrs["ref"] = refs
        grp.attrs["alt"] = alts

        chunk_len = min(DEFAULT_CHUNK_LEN, max(1, n_var))
        for name, dtype, fill in [
            ("genotype", np.int8, -1),
            ("allele_depth_ref", np.int32, 0),
            ("allele_depth_alt", np.int32, 0),
            ("qual", np.float32, np.nan),
        ]:
            grp.create_array(
                name=name,
                shape=(self.n_samples, n_var),
                chunks=(1, chunk_len),
                dtype=dtype,
                compressors=[self.compressor],
                fill_value=fill,
                overwrite=True,
            )

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
    ) -> "VariantStore":
        """
        Create a VariantStore from per-sample VCF.gz files.

        Each file is treated as a single-sample VCF; the first sample in each
        file is used. Variant positions are unioned across all samples per
        chromosome; positions not called in a sample receive genotype ``-1``
        (missing) and allele depths of ``0``.

        Parameters
        ----------
        vcf_files : list of str or Path
            Paths to VCF/VCF.gz files (one per sample).
        store_path : Path or str
            Output Zarr store path.
        sample_names : list of str, optional
            Sample names aligned with ``vcf_files``. Defaults to the part of
            the filename before the first ``.`` (e.g. ``snp.vcf.gz`` → ``snp``).
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
        vcf_files = [Path(f) for f in vcf_files]
        if sample_names is None:
            # Use stem up to first dot so "snp.vcf.gz" -> "snp"
            sample_names = [f.name.split(".")[0] for f in vcf_files]
        if len(sample_names) != len(vcf_files):
            raise ValueError("sample_names length must match vcf_files length")

        store = cls(
            store_path=store_path,
            sample_names=sample_names,
            overwrite=overwrite,
            resume=resume,
        )

        logger.info("Reading VCF files...")
        all_file_data: list[dict[str, pd.DataFrame]] = []
        for path in vcf_files:
            logger.info(f"  {path.name}")
            all_file_data.append(_read_vcf(path, filter_chromosomes=filter_chromosomes))

        all_chroms: set[str] = set()
        for fd in all_file_data:
            all_chroms.update(fd.keys())

        logger.info(f"Building union positions and writing store ({len(all_chroms)} chroms)...")
        for chrom in sorted(all_chroms):
            chrom_samples = [
                (i, fd[chrom]) for i, fd in enumerate(all_file_data) if chrom in fd
            ]
            all_positions = np.unique(
                np.concatenate([df["pos"].values for _, df in chrom_samples])
            )

            # Aggregate ref/alt from the first sample that reports each position
            pos_to_ref: dict[int, str] = {}
            pos_to_alt: dict[int, str] = {}
            for _, df in chrom_samples:
                for row in df.itertuples(index=False):
                    p = int(row.pos)
                    if p not in pos_to_ref:
                        pos_to_ref[p] = str(row.ref)
                        pos_to_alt[p] = str(row.alt)

            refs = [pos_to_ref[int(p)] for p in all_positions]
            alts = [pos_to_alt[int(p)] for p in all_positions]

            store._init_chrom_arrays(chrom, all_positions, refs, alts)

            pos_to_idx: dict[int, int] = {int(p): i for i, p in enumerate(all_positions)}

            for sample_idx, df in chrom_samples:
                indices = np.array([pos_to_idx[int(p)] for p in df["pos"].values])
                store.root[chrom]["genotype"][sample_idx, indices] = (
                    df["genotype"].values.astype(np.int8)
                )
                store.root[chrom]["allele_depth_ref"][sample_idx, indices] = (
                    df["ad_ref"].values.astype(np.int32)
                )
                store.root[chrom]["allele_depth_alt"][sample_idx, indices] = (
                    df["ad_alt"].values.astype(np.int32)
                )
                store.root[chrom]["qual"][sample_idx, indices] = (
                    df["qual"].values.astype(np.float32)
                )

            logger.info(f"  {chrom}: {len(all_positions)} variants")

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

    def set_metadata(
        self, metadata: pd.DataFrame, sample_column: str = "sample_id"
    ) -> None:
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
        """Return variant positions (1-based) for a chromosome."""
        return self.root[chrom]["positions"][:]

    def get_alleles(self, chrom: str) -> tuple[list[str], list[str]]:
        """
        Return ``(ref, alt)`` allele lists for a chromosome.

        Each list is aligned with :meth:`get_positions`.
        """
        grp_attrs = self.root[chrom].attrs
        return list(grp_attrs["ref"]), list(grp_attrs["alt"])

    def to_xarray(
        self,
        chromosomes: list[str] | None = None,
        variable: str = "genotype",
    ) -> dict[str, xr.DataArray]:
        """
        Extract variant data as per-chromosome Xarray DataArrays (lazy dask-backed).

        Parameters
        ----------
        chromosomes : list[str], optional
            Chromosomes to extract. Defaults to all.
        variable : str, default "genotype"
            Which array to extract: ``"genotype"``, ``"allele_depth_ref"``,
            ``"allele_depth_alt"``, or ``"qual"``.

        Returns
        -------
        dict[str, xr.DataArray]
            Each DataArray has dims ``(sample, position)`` where ``position``
            holds the actual 1-based genomic coordinates.
        """
        valid = {"genotype", "allele_depth_ref", "allele_depth_alt", "qual"}
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
        """
        Extract variant data for a genomic region.

        Parameters
        ----------
        region : str, optional
            Region string, e.g. ``"chr21:5000000-6000000"``.
        chrom, start, end : optional
            Alternative to ``region``. Coordinates are 1-based (VCF convention)
            and inclusive on both ends.
        variable : str, default "genotype"
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
            mask &= positions <= end  # inclusive for 1-based VCF coords
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

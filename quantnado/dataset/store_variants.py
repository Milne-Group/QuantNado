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
from .store_bam import _compute_sample_hash, _to_str_list

# Genotype encoding
GT_MISSING: np.int8 = np.int8(-1)
GT_HOM_REF: np.int8 = np.int8(0)
GT_HET: np.int8 = np.int8(1)
GT_HOM_ALT: np.int8 = np.int8(2)

CHUNK_LEN = 65536


def _read_vcf(
    path: Path | str,
    filter_chromosomes: bool = True,
) -> dict[str, pd.DataFrame]:
    """Read variants from a single-sample VCF/VCF.gz file.

    Returns dict mapping chromosome -> DataFrame with columns:
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
                gt_int = np.int8(min(n_alt, 2))

            ad = samp.get("AD", None)
            ad_ref = int(ad[0]) if ad and len(ad) > 0 else 0
            ad_alt = int(ad[1]) if ad and len(ad) > 1 else 0

            records.append({
                "chrom": chrom,
                "pos": np.int64(rec.pos),  # 1-based
                "ref": str(rec.ref),
                "alt": str(alt),
                "qual": np.float32(rec.qual) if rec.qual is not None else np.float32(np.nan),
                "genotype": gt_int,
                "ad_ref": np.int32(ad_ref),
                "ad_alt": np.int32(ad_alt),
            })

    if not records:
        return {}
    df = pd.DataFrame(records)
    return {chrom: grp.reset_index(drop=True) for chrom, grp in df.groupby("chrom")}


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
        ├── ref             stored in root.attrs["ref_alleles"] (list of str)
        ├── alt             stored in root.attrs["alt_alleles"] (list of str)
        ├── genotype        (n_variants, n_samples)    int8   -1=missing,0=hom_ref,1=het,2=hom_alt
        ├── allele_depth_ref (n_variants, n_samples)   int32
        ├── allele_depth_alt (n_variants, n_samples)   int32
        └── qual            (n_variants, n_samples)    float32
        └── metadata/
            └── completed   (n_samples,)               bool

        root.attrs:
            sample_names, n_samples, store_type,
            contig_list (list of chromosome names),
            contig_offsets (dict: chrom -> [start_row, end_row]),
            ref_alleles, alt_alleles (flat lists aligned with position array)

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
                raise FileNotFoundError(f"Store does not exist at {self.store_path} (read_only=True)")
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
            "store_type": "variants",
            "metadata_data_type": ["variants"] * self.n_samples,
        })
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
        return list(self.root.attrs.get("contig_list", []))

    def _write_flat_store(self, all_file_data: list[dict[str, pd.DataFrame]]) -> None:
        """Build flat sparse arrays from per-sample per-chromosome DataFrames."""
        all_chroms: list[str] = sorted(
            {chrom for fd in all_file_data for chrom in fd.keys()}
        )

        contig_list: list[str] = []
        contig_offsets: dict[str, list[int]] = {}
        all_contig_idx: list[np.ndarray] = []
        all_positions: list[np.ndarray] = []
        all_refs: list[str] = []
        all_alts: list[str] = []

        for chrom_idx, chrom in enumerate(all_chroms):
            chrom_positions = np.unique(
                np.concatenate([fd[chrom]["pos"].values for fd in all_file_data if chrom in fd])
            ).astype(np.int64)

            # Collect ref/alt from first sample that reports each position
            pos_to_ref: dict[int, str] = {}
            pos_to_alt: dict[int, str] = {}
            for fd in all_file_data:
                if chrom not in fd:
                    continue
                for row in fd[chrom].itertuples(index=False):
                    p = int(row.pos)
                    if p not in pos_to_ref:
                        pos_to_ref[p] = str(row.ref)
                        pos_to_alt[p] = str(row.alt)

            start_row = sum(len(a) for a in all_positions)
            end_row = start_row + len(chrom_positions)
            contig_list.append(chrom)
            contig_offsets[chrom] = [start_row, end_row]
            all_contig_idx.append(np.full(len(chrom_positions), chrom_idx, dtype=np.uint8))
            all_positions.append(chrom_positions)
            all_refs.extend(pos_to_ref[int(p)] for p in chrom_positions)
            all_alts.extend(pos_to_alt[int(p)] for p in chrom_positions)

        n_variants = sum(len(a) for a in all_positions)
        contig_arr = np.concatenate(all_contig_idx)
        position_arr = np.concatenate(all_positions)

        chunk = min(CHUNK_LEN, max(1, n_variants))

        self.root.create_array("contig", shape=(n_variants,), dtype=np.uint8, fill_value=0,
                               overwrite=True, chunks=(chunk,), compressors=[self.compressor])
        self.root.create_array("position", shape=(n_variants,), dtype=np.int64, fill_value=0,
                               overwrite=True, chunks=(chunk,), compressors=[self.compressor])
        self.root["contig"][:] = contig_arr
        self.root["position"][:] = position_arr

        for name, dtype, fill in [
            ("genotype", np.int8, -1),
            ("allele_depth_ref", np.int32, 0),
            ("allele_depth_alt", np.int32, 0),
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

        # Fill per-sample data
        for sample_idx, fd in enumerate(all_file_data):
            for chrom in all_chroms:
                if chrom not in fd:
                    continue
                df = fd[chrom]
                row_start, row_end = contig_offsets[chrom]
                chrom_positions = position_arr[row_start:row_end]
                pos_to_flat = {int(p): row_start + i for i, p in enumerate(chrom_positions)}
                flat_indices = np.array([pos_to_flat[int(p)] for p in df["pos"].values])

                self.root["genotype"][flat_indices, sample_idx] = df["genotype"].values.astype(np.int8)
                self.root["allele_depth_ref"][flat_indices, sample_idx] = df["ad_ref"].values.astype(np.int32)
                self.root["allele_depth_alt"][flat_indices, sample_idx] = df["ad_alt"].values.astype(np.int32)
                self.root["qual"][flat_indices, sample_idx] = df["qual"].values.astype(np.float32)

        self.root.attrs.update({
            "contig_list": contig_list,
            "contig_offsets": contig_offsets,
            "n_variants": n_variants,
            "chromosomes": contig_list,
            "ref_alleles": all_refs,
            "alt_alleles": all_alts,
        })
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
    ) -> "VariantStore":
        """Create a VariantStore from per-sample VCF.gz files."""
        vcf_files = [Path(f) for f in vcf_files]
        if sample_names is None:
            sample_names = [f.name.split(".")[0] for f in vcf_files]
        if len(sample_names) != len(vcf_files):
            raise ValueError("sample_names length must match vcf_files length")

        store = cls(store_path=store_path, sample_names=sample_names, overwrite=overwrite, resume=resume)

        logger.info("Reading VCF files...")
        all_file_data: list[dict[str, pd.DataFrame]] = []
        for path in vcf_files:
            logger.info(f"  {path.name}")
            all_file_data.append(_read_vcf(path, filter_chromosomes=filter_chromosomes))

        store._write_flat_store(all_file_data)

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
        offsets = self.root.attrs.get("contig_offsets", {})
        if chrom not in offsets:
            raise ValueError(f"Chromosome '{chrom}' not in store. Available: {self.chromosomes}")
        start, end = offsets[chrom]
        return int(start), int(end)

    def get_positions(self, chrom: str) -> np.ndarray:
        """Return variant positions (1-based) for a chromosome."""
        start, end = self._contig_row_range(chrom)
        return self.root["position"][start:end]

    def get_alleles(self, chrom: str) -> tuple[list[str], list[str]]:
        """Return (ref, alt) allele lists for a chromosome, aligned with get_positions."""
        start, end = self._contig_row_range(chrom)
        refs = self.root.attrs.get("ref_alleles", [])[start:end]
        alts = self.root.attrs.get("alt_alleles", [])[start:end]
        return list(refs), list(alts)

    def to_xarray(
        self,
        chromosomes: list[str] | None = None,
        variable: str = "genotype",
    ) -> dict[str, xr.DataArray]:
        """Extract variant data as per-chromosome lazy Xarray DataArrays.

        Returns DataArrays with dims ``(sample, position)``.
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
            start_row, end_row = self._contig_row_range(chrom)
            positions = self.root["position"][start_row:end_row]
            zarr_arr = self.root[variable]
            # (n_variants_chrom, n_samples) → transpose to (sample, position)
            dask_arr = da.from_zarr(zarr_arr)[start_row:end_row, :].T

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

        mask = np.ones(len(positions), dtype=bool)
        if start is not None:
            mask &= positions >= start
        if end is not None:
            mask &= positions <= end  # inclusive for 1-based VCF coords
        pos_indices = np.where(mask)[0]
        region_positions = positions[pos_indices]
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
            attrs={"variable": variable, "chromosome": chrom, "start": start, "end": end},
        )

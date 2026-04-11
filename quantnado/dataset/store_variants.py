"""VariantStore — per-sample Zarr store for VCF/SNP data.

Store layout::

    {SampleName}.zarr/
    ├── {chrom}/
    │   ├── GT   (1, chrom_len)  int8    — genotype: -1=missing, 0=hom_ref, 1=het, 2=hom_alt
    │   ├── DP   (1, chrom_len)  uint16  — read depth
    │   ├── AF   (1, chrom_len)  float32 — alt allele frequency (AD_alt / DP)
    │   └── MQ   (1, chrom_len)  uint8   — mapping quality
    ├── metadata/
    │   ├── completed        bool    (1,)
    │   ├── total_reads      int64   (1,)
    │   ├── mean_read_length float32 (1,)
    │   └── sparsity         float32 (1,)
    └── root.attrs: assay="snp", sample, chromsizes, chunk_len

VCF positions are 1-based; array indices are 0-based (position - 1).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pysam
import zarr
from loguru import logger
from zarr.storage import LocalStore

from .metadata import _parse_chromsizes, create_metadata_group
from .store_bam import (
    _delete_path,
    _normalize_path,
    _resolve_chunk_len,
    _resolve_compressors,
    DEFAULT_CONSTRUCTION_COMPRESSION,
)

# Genotype encoding
GT_MISSING: np.int8 = np.int8(-1)
GT_HOM_REF: np.int8 = np.int8(0)
GT_HET: np.int8 = np.int8(1)
GT_HOM_ALT: np.int8 = np.int8(2)


# ---------------------------------------------------------------------------
# VCF reading helpers
# ---------------------------------------------------------------------------


def _read_vcf(
    path: Path | str,
    filter_chromosomes: bool = True,
    test: bool = False,
    test_chromosomes: list[str] | tuple[str, ...] | None = None,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, int]]:
    """Read a single-sample VCF/VCF.gz with pysam (htslib).

    Returns
    -------
    (chrom_data, header_chromsizes)
        chrom_data  : dict[chrom -> dict[field -> 1D array]]
                      fields: GT (int8), DP (uint16), AF (float32), MQ (uint8)
                      pos is 1-based (matching VCF POS)
        header_chromsizes : dict[chrom -> int] from ##contig header lines
    """
    path = Path(path)

    if test or test_chromosomes:
        allowed_chroms: set[str] | None = set(
            _parse_chromsizes({}, test=test, test_chromosomes=test_chromosomes)
        )
    else:
        allowed_chroms = None

    with pysam.VariantFile(str(path)) as vcf:
        # Chromsizes from header contigs
        header_chromsizes: dict[str, int] = {
            c.name: c.length
            for c in vcf.header.contigs.values()
            if c.length is not None
        }

        # Chromosomes to process
        if filter_chromosomes:
            target_chroms = [
                c for c in vcf.header.contigs
                if c.startswith("chr") and "_" not in c
            ]
        else:
            target_chroms = list(vcf.header.contigs)

        if allowed_chroms is not None:
            target_chroms = [c for c in target_chroms if c in allowed_chroms]

        # Check if file is tabix-indexed (enables fast per-chrom fetch)
        is_indexed = Path(str(path) + ".tbi").exists() or Path(str(path) + ".csi").exists()

        chrom_data: dict[str, dict[str, np.ndarray]] = {}

        if is_indexed:
            # Fast path: fetch per chromosome (skips irrelevant contigs entirely)
            for chrom in target_chroms:
                records = _collect_chrom_records(vcf, chrom, fetch=True)
                if records:
                    chrom_data[chrom] = _records_to_arrays(records)
        else:
            # Sequential scan: one pass, bucket by chromosome
            buckets: dict[str, list] = {c: [] for c in target_chroms}
            target_set = set(target_chroms)
            for rec in vcf.fetch(contig=None):
                c = rec.chrom
                if c in target_set:
                    buckets[c].append(_extract_record(rec))
            for chrom, recs in buckets.items():
                if recs:
                    chrom_data[chrom] = _records_to_arrays(recs)

    return chrom_data, header_chromsizes


def _extract_record(rec: "pysam.VariantRecord") -> tuple:
    """Extract (pos_1based, gt, dp, af, mq) from a pysam VCF record."""
    pos = rec.pos + 1  # pysam pos is 0-based; VCF POS is 1-based

    # GT from first sample
    try:
        sample = rec.samples[0]
        alleles = sample.allele_indices  # tuple of ints, None=missing
        if alleles is None or any(a is None for a in alleles):
            gt = GT_MISSING
        else:
            gt = np.int8(min(sum(a > 0 for a in alleles), 2))
    except (KeyError, IndexError):
        gt = GT_MISSING

    # DP
    try:
        dp_val = sample["DP"]
        dp = np.uint16(max(0, min(dp_val, 65535))) if dp_val is not None and dp_val >= 0 else np.uint16(0)
    except (KeyError, TypeError):
        dp = np.uint16(0)

    # AF from AD (allele depths)
    try:
        ad = sample["AD"]
        if ad is not None and len(ad) >= 2 and ad[0] is not None and ad[1] is not None:
            ref_d = max(0, ad[0])
            alt_d = max(0, ad[1])
            total = ref_d + alt_d
            af = np.float32(alt_d / total) if total > 0 else np.float32(np.nan)
        else:
            af = np.float32(np.nan)
    except (KeyError, TypeError, IndexError):
        af = np.float32(np.nan)

    # MQ (INFO field)
    try:
        mq_val = rec.info["MQ"]
        mq = np.uint8(min(max(0, int(mq_val)), 255)) if mq_val is not None else np.uint8(0)
    except (KeyError, TypeError):
        mq = np.uint8(0)

    return pos, gt, dp, af, mq


def _collect_chrom_records(vcf: "pysam.VariantFile", chrom: str, fetch: bool) -> list:
    records = []
    try:
        it = vcf.fetch(chrom) if fetch else vcf.fetch(contig=None)
        for rec in it:
            if not fetch and rec.chrom != chrom:
                continue
            records.append(_extract_record(rec))
    except (ValueError, KeyError):
        pass
    return records


def _records_to_arrays(records: list) -> dict[str, np.ndarray]:
    """Convert list of (pos, gt, dp, af, mq) tuples to dict of arrays."""
    pos_arr = np.array([r[0] for r in records], dtype=np.int64)
    gt_arr = np.array([r[1] for r in records], dtype=np.int8)
    dp_arr = np.array([r[2] for r in records], dtype=np.uint16)
    af_arr = np.array([r[3] for r in records], dtype=np.float32)
    mq_arr = np.array([r[4] for r in records], dtype=np.uint8)
    return {"pos": pos_arr, "GT": gt_arr, "DP": dp_arr, "AF": af_arr, "MQ": mq_arr}


# ---------------------------------------------------------------------------
# VariantStore
# ---------------------------------------------------------------------------


class VariantStore:
    """Per-sample Zarr store for VCF/SNP data.

    Stores dense ``(1, chrom_len)`` arrays for GT, DP, AF, MQ.
    VCF positions (1-based) are mapped to 0-based array indices.

    Use :meth:`from_vcf` to create or :meth:`open` to read.
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
            "assay": "snp",
            "sample": sample,
            "chromsizes": chromsizes,
            "chunk_len": chunk_len,
        })

    def _init_arrays(self) -> None:
        for chrom, chrom_len in self.chromsizes.items():
            grp = self.root.require_group(chrom)
            specs = [
                ("GT",  np.int8,    GT_MISSING),
                ("DP",  np.uint16,  0),
                ("AF",  np.float32, np.nan),
                ("MQ",  np.uint8,   0),
            ]
            for key, dtype, fill in specs:
                grp.require_array(
                    key,
                    shape=(1, chrom_len),
                    chunks=(1, self.chunk_len),
                    dtype=dtype,
                    compressors=self.compressors,
                    fill_value=fill,
                    overwrite=True,
                )

    def _write_variants(self, chrom_data: dict[str, dict[str, np.ndarray]]) -> float:
        """Scatter VCF data to dense arrays. Returns mean fraction of covered positions."""
        covered_fracs = []
        for chrom, chrom_len in self.chromsizes.items():
            if chrom not in chrom_data:
                continue
            data = chrom_data[chrom]
            pos_0based = (data["pos"] - 1).astype(np.int64)
            valid = (pos_0based >= 0) & (pos_0based < chrom_len)
            idx = pos_0based[valid]
            grp = self.root[chrom]

            gt_arr = np.full(chrom_len, GT_MISSING, dtype=np.int8)
            gt_arr[idx] = data["GT"][valid]
            grp["GT"][0, :] = gt_arr

            dp_arr = np.zeros(chrom_len, dtype=np.uint16)
            dp_arr[idx] = data["DP"][valid]
            grp["DP"][0, :] = dp_arr

            af_arr = np.full(chrom_len, np.nan, dtype=np.float32)
            af_arr[idx] = data["AF"][valid]
            grp["AF"][0, :] = af_arr

            mq_arr = np.zeros(chrom_len, dtype=np.uint8)
            mq_arr[idx] = data["MQ"][valid]
            grp["MQ"][0, :] = mq_arr

            covered_fracs.append(len(idx) / chrom_len)

        return float(np.mean(covered_fracs)) if covered_fracs else 0.0

    def _finalise(self, n_variants: int) -> None:
        self.meta["completed"][0] = True
        self.meta["total_reads"][0] = n_variants
        zarr.consolidate_metadata(str(self.store_path))
        logger.info(f"Completed {self.sample}: {n_variants:,} variants")

    @classmethod
    def open(cls, store_path: str | Path, read_only: bool = True) -> "VariantStore":
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
    def from_vcf(
        cls,
        vcf_path: str | Path,
        store_path: Path | str,
        sample: str,
        chromsizes: str | Path | dict[str, int] | None = None,
        *,
        chunk_len: int | None = None,
        construction_compression: str = DEFAULT_CONSTRUCTION_COMPRESSION,
        overwrite: bool = True,
        filter_chromosomes: bool = True,
        test: bool = False,
        test_chromosomes: list[str] | tuple[str, ...] | None = None,
        log_file: Path | None = None,
    ) -> "VariantStore":
        """Create a per-sample VariantStore zarr from a single-sample VCF.

        Parameters
        ----------
        vcf_path:
            Path to annotated VCF (.vcf or .vcf.gz).
        store_path:
            Output .zarr directory.
        sample:
            Sample name.
        chromsizes:
            Path to .chrom.sizes, dict, or None to infer from VCF ##contig headers.
        """
        if log_file is not None:
            from quantnado.utils import setup_logging
            setup_logging(Path(log_file), verbose=False)

        logger.info(f"Reading VCF: {vcf_path}")
        chrom_data, header_chromsizes = _read_vcf(
            vcf_path,
            filter_chromosomes=filter_chromosomes,
            test=test,
            test_chromosomes=test_chromosomes,
        )

        if chromsizes is None:
            if not header_chromsizes:
                raise ValueError(
                    "No chromsizes provided and VCF ##contig headers are missing. "
                    "Provide chromsizes explicitly."
                )
            chromsizes = header_chromsizes

        chromsizes_dict = _parse_chromsizes(
            chromsizes,
            filter_chromosomes=filter_chromosomes,
            test=test,
            test_chromosomes=test_chromosomes,
        )
        resolved_chunk_len = _resolve_chunk_len(chromsizes_dict, Path(store_path), chunk_len)
        compressors = _resolve_compressors(construction_compression)

        store = cls(
            store_path=store_path,
            sample=sample,
            chromsizes=chromsizes_dict,
            chunk_len=resolved_chunk_len,
            compressors=compressors,
            overwrite=overwrite,
        )

        store._write_variants(chrom_data)
        n_variants = sum(len(d["pos"]) for d in chrom_data.values())
        store._finalise(n_variants)

        return store

    @property
    def completed(self) -> bool:
        if self.meta is None:
            return False
        return bool(self.meta["completed"][0])

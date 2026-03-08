from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from .store_bam import BamStore, DEFAULT_CHUNK_LEN
from .store_methyl import MethylStore
from .store_variants import VariantStore


class MultiomicsStore:
    """
    Unified multi-modal genomic store combining coverage (BAM), methylation
    (bedGraph), and variant (VCF) data in a single directory.

    On-disk layout::

        <store_dir>/
            coverage.zarr/      (present if BAM files were provided)
            methylation.zarr/   (present if bedGraph files were provided)
            variants.zarr/      (present if VCF files were provided)

    Each sub-store is a self-contained Zarr archive; access them individually
    via :attr:`coverage`, :attr:`methylation`, and :attr:`variants`, or use the
    high-level helpers on this class.

    Example
    -------
    >>> ms = MultiomicsStore.from_files(
    ...     store_dir="dataset/",
    ...     bam_files=["atac.bam", "meth-rep1.bam"],
    ...     bedgraph_files=["meth-rep1.bedGraph", "meth-rep2.bedGraph"],
    ...     vcf_files=["snp.vcf.gz"],
    ... )
    >>> ms.modalities
    ['coverage', 'methylation', 'variants']
    >>> ms.coverage.sample_names
    ['atac', 'meth-rep1']
    >>> ms.methylation.to_xarray()
    >>> ms.variants.extract_region("chr21:5000000-6000000")
    """

    def __init__(self, store_dir: Path | str) -> None:
        self.store_dir = Path(store_dir)

        self.coverage: BamStore | None = None
        self.methylation: MethylStore | None = None
        self.variants: VariantStore | None = None

        cov_path = self.store_dir / "coverage.zarr"
        meth_path = self.store_dir / "methylation.zarr"
        var_path = self.store_dir / "variants.zarr"

        if cov_path.exists():
            self.coverage = BamStore.open(cov_path)
        if meth_path.exists():
            self.methylation = MethylStore.open(meth_path)
        if var_path.exists():
            self.variants = VariantStore.open(var_path)

        if not self.modalities:
            logger.warning(
                f"No modality stores found in {self.store_dir}. "
                "Run from_files() to populate the store."
            )

    # ---- Construction ----

    @classmethod
    def from_files(
        cls,
        store_dir: Path | str,
        bam_files: list[str | Path] | None = None,
        bedgraph_files: list[str | Path] | None = None,
        vcf_files: list[str | Path] | None = None,
        chromsizes: str | Path | dict[str, int] | None = None,
        metadata: pd.DataFrame | Path | str | None = None,
        *,
        bam_sample_names: list[str] | None = None,
        bedgraph_sample_names: list[str] | None = None,
        vcf_sample_names: list[str] | None = None,
        filter_chromosomes: bool = True,
        overwrite: bool = True,
        resume: bool = False,
        sample_column: str = "sample_id",
        chunk_len: int = DEFAULT_CHUNK_LEN,
        construction_compression: str = "default",
        local_staging: bool = False,
        staging_dir: "Path | str | None" = None,
        log_file: "Path | None" = None,
        max_workers: int = 1,
        chr_workers: int = 1,
        test: bool = False,
    ) -> "MultiomicsStore":
        """
        Create a MultiomicsStore from genomic data files.

        At least one of ``bam_files``, ``bedgraph_files``, or ``vcf_files``
        must be provided. Any omitted modality is simply absent from the store.

        Parameters
        ----------
        store_dir : Path or str
            Output directory. Created if it does not exist.
        bam_files : list of Path, optional
            BAM files for per-base coverage storage.
        bedgraph_files : list of Path, optional
            MethylDackel CpG bedGraph files for methylation storage.
        vcf_files : list of Path, optional
            VCF.gz files (one per sample) for variant storage.
        chromsizes : str, Path, or dict, optional
            Chromosome sizes for the coverage store. Extracted from the first
            BAM file if not provided.
        metadata : DataFrame, Path, or str, optional
            Sample metadata CSV attached to all sub-stores. Each sub-store's
            sample names are used to subset the metadata automatically.
        bam_sample_names : list of str, optional
            Override sample names for BAM files (default: file stems).
        bedgraph_sample_names : list of str, optional
            Override sample names for bedGraph files (default: file stems).
            Useful when MethylDackel embeds genome/suffix in the filename,
            e.g. ``meth-rep1_hg38_CpG_inverted.bedGraph`` → ``"meth-rep1"``.
        vcf_sample_names : list of str, optional
            Override sample names for VCF files (default: filename before first dot).
        filter_chromosomes : bool, default True
            Keep only canonical chromosomes (``chr*`` without underscores).
        overwrite : bool, default True
            Overwrite existing sub-stores.
        resume : bool, default False
            Resume processing an existing sub-store, skipping completed samples.
        sample_column : str, default "sample_id"
            Column in ``metadata`` matching sample names.
        chunk_len : int, default 65536
            Zarr chunk size for the position dimension (coverage store only).
        construction_compression : {"default", "fast", "none"}, default "default"
            Build-time compression profile for the coverage store.
        local_staging : bool, default False
            Build the coverage store under local scratch storage before publishing.
        staging_dir : str or Path, optional
            Scratch directory for local staging. Defaults to system temp dir.
        log_file : Path, optional
            Path to write BAM processing logs.
        max_workers : int, default 1
            Sample-level parallel workers for BAM processing.
        chr_workers : int, default 1
            Chromosome-level parallel workers within each sample thread.
            Total concurrent BAM reads = max_workers * chr_workers.
        test : bool, default False
            Restrict coverage to chr21/chr22/chrY (for testing).
        """
        if not any([bam_files, bedgraph_files, vcf_files]):
            raise ValueError("Provide at least one of bam_files, bedgraph_files, or vcf_files")

        store_dir = Path(store_dir)
        store_dir.mkdir(parents=True, exist_ok=True)

        if bam_files:
            logger.info(f"Building coverage store from {len(bam_files)} BAM file(s)...")
            BamStore.from_bam_files(
                bam_files=[str(f) for f in bam_files],
                store_path=store_dir / "coverage.zarr",
                chromsizes=chromsizes,
                metadata=metadata,
                filter_chromosomes=filter_chromosomes,
                overwrite=overwrite,
                resume=resume,
                sample_column=sample_column,
                chunk_len=chunk_len,
                construction_compression=construction_compression,
                local_staging=local_staging,
                staging_dir=staging_dir,
                log_file=log_file,
                max_workers=max_workers,
                chr_workers=chr_workers,
                test=test,
            )

        if bedgraph_files:
            logger.info(f"Building methylation store from {len(bedgraph_files)} bedGraph file(s)...")
            MethylStore.from_bedgraph_files(
                bedgraph_files=[str(f) for f in bedgraph_files],
                store_path=store_dir / "methylation.zarr",
                sample_names=bedgraph_sample_names,
                metadata=metadata,
                filter_chromosomes=filter_chromosomes,
                overwrite=overwrite,
                resume=resume,
                sample_column=sample_column,
            )

        if vcf_files:
            logger.info(f"Building variants store from {len(vcf_files)} VCF file(s)...")
            VariantStore.from_vcf_files(
                vcf_files=[str(f) for f in vcf_files],
                store_path=store_dir / "variants.zarr",
                sample_names=vcf_sample_names,
                metadata=metadata,
                filter_chromosomes=filter_chromosomes,
                overwrite=overwrite,
                resume=resume,
                sample_column=sample_column,
            )

        return cls(store_dir)

    @classmethod
    def open(cls, store_dir: str | Path) -> "MultiomicsStore":
        """Open an existing MultiomicsStore directory."""
        store_dir = Path(store_dir)
        if not store_dir.exists():
            raise FileNotFoundError(f"Store directory does not exist: {store_dir}")
        return cls(store_dir)

    # ---- Properties ----

    @property
    def modalities(self) -> list[str]:
        """List of available modalities in this store."""
        result = []
        if self.coverage is not None:
            result.append("coverage")
        if self.methylation is not None:
            result.append("methylation")
        if self.variants is not None:
            result.append("variants")
        return result

    @property
    def chromosomes(self) -> list[str]:
        """Sorted union of chromosome names across all modalities."""
        seen: set[str] = set()
        for store in self._active_stores():
            seen.update(store.chromosomes)
        return sorted(seen)

    @property
    def samples(self) -> dict[str, list[str]]:
        """
        Sample names available per modality.

        Returns
        -------
        dict
            Mapping modality name → list of sample names.
        """
        out: dict[str, list[str]] = {}
        if self.coverage is not None:
            out["coverage"] = list(self.coverage.sample_names)
        if self.methylation is not None:
            out["methylation"] = list(self.methylation.sample_names)
        if self.variants is not None:
            out["variants"] = list(self.variants.sample_names)
        return out

    @property
    def all_sample_names(self) -> list[str]:
        """Ordered union of sample names across all modalities."""
        seen: set[str] = set()
        result: list[str] = []
        for store in self._active_stores():
            for s in store.sample_names:
                if s not in seen:
                    result.append(s)
                    seen.add(s)
        return result

    def _active_stores(self):
        return [s for s in (self.coverage, self.methylation, self.variants) if s is not None]

    # ---- Metadata ----

    def set_metadata(
        self,
        metadata: pd.DataFrame,
        sample_column: str = "sample_id",
    ) -> None:
        """
        Attach metadata to all sub-stores.

        Each store is updated using the subset of ``metadata`` that matches its
        own sample names — samples not present in that store are ignored.
        """
        for store in self._active_stores():
            try:
                store.set_metadata(metadata, sample_column=sample_column)
            except Exception as e:
                logger.warning(f"Could not set metadata on {type(store).__name__}: {e}")

    def get_metadata(self) -> pd.DataFrame:
        """
        Combined metadata across all modalities.

        Returns a DataFrame indexed by ``sample_id`` with one row per unique
        sample. A ``modalities`` column lists which modalities each sample
        appears in.
        """
        frames: dict[str, pd.DataFrame] = {}
        for modality, store in [
            ("coverage", self.coverage),
            ("methylation", self.methylation),
            ("variants", self.variants),
        ]:
            if store is None:
                continue
            df = store.get_metadata().copy()
            df["modality"] = modality
            frames[modality] = df

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames.values(), ignore_index=False)

        # Build a modalities column showing which are available per sample
        sample_modalities: dict[str, list[str]] = {}
        for modality, df in frames.items():
            for sid in df.index:
                sample_modalities.setdefault(str(sid), []).append(modality)

        # Deduplicate: keep first occurrence per sample_id
        combined = combined[~combined.index.duplicated(keep="first")].drop(
            columns=["modality"], errors="ignore"
        )
        combined["modalities"] = combined.index.map(
            lambda s: ", ".join(sample_modalities.get(str(s), []))
        )
        return combined

    # ---- Summary ----

    def __repr__(self) -> str:
        lines = [f"MultiomicsStore at '{self.store_dir}'"]
        lines.append(f"  modalities : {self.modalities}")
        for modality, store in [
            ("coverage", self.coverage),
            ("methylation", self.methylation),
            ("variants", self.variants),
        ]:
            if store is not None:
                lines.append(
                    f"  {modality:<12}: {len(store.sample_names)} samples, "
                    f"{len(store.chromosomes)} chrom(s)"
                )
        return "\n".join(lines)

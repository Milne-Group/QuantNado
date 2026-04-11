"""Simple read interface for multiomics stores."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr
import zarr

from .store_coverage import BamStore
from .store_methyl import MethylStore
from .store_variants import VariantStore
from .combine_multiomics import combine_multiomics_stores
from loguru import logger


class MultiomicsStore:
    """
    Lightweight read interface for multiomics Zarr stores.

    Combines coverage (BAM), methylation (bedGraph), and variant (VCF) data
    stored in a single Zarr directory or as separate stores.
    """

    def __init__(self, store_dir: Path | str) -> None:
        """Open an existing multiomics store."""
        self.store_dir = Path(store_dir)

        self.coverage: BamStore | None = None
        self.methylation: MethylStore | None = None
        self.variants: VariantStore | None = None

        # Try to open each modality store
        cov_path = self.store_dir / "coverage.zarr"
        meth_path = self.store_dir / "methylation.zarr"
        var_path = self.store_dir / "variants.zarr"

        if cov_path.exists():
            self.coverage = BamStore.open(cov_path)
        if meth_path.exists():
            self.methylation = MethylStore.open(meth_path)
        if var_path.exists():
            self.variants = VariantStore.open(var_path)

        # Also try unified layout (all in one zarr root)
        if self.store_dir.exists():
            try:
                root = zarr.open_group(str(self.store_dir), mode="r")
                if not self.coverage and "coverage" in root:
                    self.coverage = BamStore.open(self.store_dir)
                if not self.methylation and ("methylation_pct" in root or "methyl_position" in root):
                    self.methylation = MethylStore.open(self.store_dir)
                if not self.variants and ("genotype" in root or "variant_position" in root):
                    self.variants = VariantStore.open(self.store_dir)
            except (OSError, ValueError):
                pass

    @classmethod
    def from_files(
        cls,
        store_dir: Path | str,
        bam_files: list[str | Path] | None = None,
        methyldackel_files: list[str | Path] | None = None,
        cxreport_files: list[str | Path] | None = None,
        mc_files: list[str | Path] | None = None,
        hmc_files: list[str | Path] | None = None,
        vcf_files: list[str | Path] | None = None,
        chromsizes: str | Path | dict[str, int] | None = None,
        metadata: pd.DataFrame | Path | str | None = None,
        bam_sample_names: list[str] | None = None,
        methyldackel_sample_names: list[str] | None = None,
        cxreport_sample_names: list[str] | None = None,
        mc_hmc_sample_names: list[str] | None = None,
        vcf_sample_names: list[str] | None = None,
        filter_chromosomes: bool = True,
        test: bool = False,
        overwrite: bool = True,
        resume: bool = False,
        chunk_len: int = 65536,
        n_workers: int = 1,
        sample_column: str = "sample_id",
        stranded: dict[str, bool] | list[str] | None = None,
        use_fragment: bool = False,
        read_filter: int | None = None,
        coverage_type: dict[str, str] | None = None,
        construction_compression: str = "zstd",
        local_staging: bool = False,
        staging_dir: str | Path | None = None,
        log_file: str | Path | None = None,
    ) -> "MultiomicsStore":
        """Create multiomics store from individual BAM, methylation, and VCF files.

        Creates separate stores for each data type, then combines them into a unified
        multiomics dataset.
        """
        store_dir = Path(store_dir)
        store_dir.mkdir(parents=True, exist_ok=True)

        individual_stores = {}

        # Create coverage store
        if bam_files:
            cov_path = store_dir / "coverage.zarr"
            logger.info(f"Creating coverage store at {cov_path}")
            BamStore.from_bam_files(
                bam_files=bam_files,
                store_path=cov_path,
                bam_sample_names=bam_sample_names,
                metadata=metadata,
                chromsizes=chromsizes,
                filter_chromosomes=filter_chromosomes,
                test=test,
                overwrite=overwrite,
                resume=resume,
                chunk_len=chunk_len,
                coverage_type=coverage_type,
                sample_column=sample_column,
                construction_compression=construction_compression,
                local_staging=local_staging,
                staging_dir=staging_dir,
                log_file=log_file,
            )
            individual_stores["coverage"] = [cov_path]

        # Create methylation store
        if methyldackel_files or cxreport_files or mc_files or hmc_files:
            meth_path = store_dir / "methylation.zarr"
            logger.info(f"Creating methylation store at {meth_path}")

            if methyldackel_files:
                MethylStore.from_bedgraph_files(
                    methyldackel_files=methyldackel_files,
                    store_path=meth_path,
                    sample_names=methyldackel_sample_names,
                    metadata=metadata,
                    filter_chromosomes=filter_chromosomes,
                    test=test,
                    overwrite=overwrite,
                    resume=resume,
                    sample_column=sample_column,
                )
            elif cxreport_files:
                MethylStore.from_cxreport_files(
                    cxreport_files=cxreport_files,
                    store_path=meth_path,
                    sample_names=cxreport_sample_names,
                    metadata=metadata,
                    filter_chromosomes=filter_chromosomes,
                    test=test,
                    overwrite=overwrite,
                    resume=resume,
                    sample_column=sample_column,
                )
            elif mc_files or hmc_files:
                MethylStore.from_split_cxreport_files(
                    mc_files=mc_files,
                    hmc_files=hmc_files,
                    store_path=meth_path,
                    sample_names=mc_hmc_sample_names,
                    metadata=metadata,
                    filter_chromosomes=filter_chromosomes,
                    test=test,
                    overwrite=overwrite,
                    resume=resume,
                    sample_column=sample_column,
                )

            individual_stores["methylation"] = [meth_path]

        # Create variant store
        if vcf_files:
            var_path = store_dir / "variants.zarr"
            logger.info(f"Creating variant store at {var_path}")
            VariantStore.from_vcf_files(
                vcf_files=vcf_files,
                store_path=var_path,
                sample_names=vcf_sample_names,
                metadata=metadata,
                filter_chromosomes=filter_chromosomes,
                test=test,
                overwrite=overwrite,
                resume=resume,
                sample_column=sample_column,
            )
            individual_stores["variants"] = [var_path]

        # Combine stores
        if individual_stores:
            combined_path = store_dir / "multiomics.zarr"
            logger.info(f"Combining stores into {combined_path}")
            combine_multiomics_stores(
                stores=individual_stores,
                output_path=combined_path,
                overwrite=overwrite,
            )

        # Open and return
        return cls(store_dir)

    @property
    def modalities(self) -> list[str]:
        """Available modalities: coverage, methylation, variants."""
        result = []
        if self.coverage is not None:
            result.append("coverage")
        if self.methylation is not None:
            result.append("methylation")
        if self.variants is not None:
            result.append("variants")
        return result

    @property
    def samples(self) -> dict[str, list[str]]:
        """Sample names per modality."""
        out: dict[str, list[str]] = {}
        if self.coverage is not None:
            out["coverage"] = list(self.coverage.sample_names)
        if self.methylation is not None:
            out["methylation"] = list(self.methylation.sample_names)
        if self.variants is not None:
            out["variants"] = list(self.variants.sample_names)
        return out

    @property
    def metadata(self) -> pd.DataFrame:
        """Union of metadata across all modalities."""
        frames: dict[str, pd.DataFrame] = {}
        for modality, store in [
            ("coverage", self.coverage),
            ("methylation", self.methylation),
            ("variants", self.variants),
        ]:
            if store is None:
                continue
            try:
                df = store.metadata.copy()
                frames[modality] = df
            except Exception:
                pass

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames.values(), ignore_index=False)
        return combined[~combined.index.duplicated(keep="first")]

    def to_xarray(self) -> xr.Dataset:
        """Combine all modalities into a single xarray Dataset."""
        datasets = []

        if self.coverage is not None:
            ds = self.coverage.to_xarray()
            if isinstance(ds, dict):
                # Multi-chromosome dict → combine
                ds = xr.merge([xr.Dataset({k: v}) for k, v in ds.items()])
            datasets.append(ds)

        if self.methylation is not None:
            ds = self.methylation.to_xarray()
            if isinstance(ds, dict):
                ds = xr.merge([xr.Dataset({k: v}) for k, v in ds.items()])
            datasets.append(ds)

        if self.variants is not None:
            ds = self.variants.to_xarray()
            if isinstance(ds, dict):
                ds = xr.merge([xr.Dataset({k: v}) for k, v in ds.items()])
            datasets.append(ds)

        if not datasets:
            raise ValueError("No modalities to combine")

        return xr.merge(datasets, join="override")

    def __repr__(self) -> str:
        lines = [f"MultiomicsStore at '{self.store_dir}'"]
        lines.append(f"  modalities : {self.modalities}")
        for modality in self.modalities:
            store = getattr(self, modality.split("_")[0])
            lines.append(f"  {modality:<12}: {len(store.sample_names)} samples")
        return "\n".join(lines)

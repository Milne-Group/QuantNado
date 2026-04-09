"""Simple read interface for multiomics stores."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr
import zarr

from .store_coverage import BamStore
from .store_methyl import MethylStore
from .store_variants import VariantStore


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

"""Integration tests for VariantStore using monkeypatched _read_vcf."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantnado.dataset.variants import VariantStore, GT_HET, GT_HOM_ALT, GT_HOM_REF, GT_MISSING


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _fake_vcf_data(chrom: str, positions: list[int], genotypes: list[int] | None = None):
    """Build a fake per-chromosome DataFrame matching _read_vcf output."""
    n = len(positions)
    gts = genotypes if genotypes is not None else [1] * n
    return pd.DataFrame({
        "chrom": [chrom] * n,
        "pos": np.array(positions, dtype=np.int64),
        "ref": ["A"] * n,
        "alt": ["T"] * n,
        "qual": np.full(n, 30.0, dtype=np.float32),
        "genotype": np.array(gts, dtype=np.int8),
        "ad_ref": np.full(n, 10, dtype=np.int32),
        "ad_alt": np.full(n, 5, dtype=np.int32),
    })


def _make_fake_reader(data: dict[str, dict[str, pd.DataFrame]]):
    def _fake(path, filter_chromosomes=True):
        stem = str(path)
        for key, val in data.items():
            if key in stem:
                return val
        return {}
    return _fake


@pytest.fixture
def two_sample_store(tmp_path, monkeypatch):
    """VariantStore with 2 samples on 2 chromosomes."""
    fake_data = {
        "s1": {
            "chr1": _fake_vcf_data("chr1", [100, 200, 300], genotypes=[1, 2, 0]),
            "chr2": _fake_vcf_data("chr2", [50, 150]),
        },
        "s2": {
            "chr1": _fake_vcf_data("chr1", [200, 300, 400], genotypes=[1, 0, 2]),
            "chr2": _fake_vcf_data("chr2", [50]),
        },
    }
    monkeypatch.setattr("quantnado.dataset.variants._read_vcf", _make_fake_reader(fake_data))

    store = VariantStore.from_vcf_files(
        vcf_files=[tmp_path / "s1.vcf.gz", tmp_path / "s2.vcf.gz"],
        store_path=tmp_path / "variants",
        sample_names=["s1", "s2"],
    )
    return store


# ---------------------------------------------------------------------------
# VariantStore construction
# ---------------------------------------------------------------------------

class TestVariantStoreConstruction:
    def test_empty_sample_names_raises(self, tmp_path):
        with pytest.raises(ValueError, match="empty"):
            VariantStore(tmp_path / "v", sample_names=[])

    def test_overwrite_false_raises_on_existing(self, tmp_path):
        VariantStore(tmp_path / "v", sample_names=["s1"])
        with pytest.raises(FileExistsError):
            VariantStore(tmp_path / "v", sample_names=["s1"], overwrite=False)

    def test_resume_stores_root(self, tmp_path):
        VariantStore(tmp_path / "v", sample_names=["s1"])
        store = VariantStore(tmp_path / "v", sample_names=["s1"], overwrite=False, resume=True)
        assert store.root is not None

    def test_normalize_path_adds_zarr_suffix(self, tmp_path):
        store = VariantStore(tmp_path / "mystore", sample_names=["s1"])
        assert str(store.store_path).endswith(".zarr")

    def test_sample_names_mismatch_raises_on_resume(self, tmp_path):
        VariantStore(tmp_path / "v", sample_names=["s1"])
        with pytest.raises(ValueError, match="mismatch"):
            VariantStore(tmp_path / "v", sample_names=["s2"], overwrite=False, resume=True)

    def test_read_only_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            VariantStore(tmp_path / "nonexistent", sample_names=["s1"], read_only=True)

    def test_read_only_overwrite_raises(self, tmp_path):
        VariantStore(tmp_path / "v", sample_names=["s1"])
        with pytest.raises(ValueError, match="read-only"):
            VariantStore(tmp_path / "v", sample_names=["s1"], overwrite=True, read_only=True)

    def test_sample_name_length_mismatch_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr("quantnado.dataset.variants._read_vcf", lambda *a, **k: {})
        with pytest.raises(ValueError, match="length"):
            VariantStore.from_vcf_files(
                vcf_files=[tmp_path / "s1.vcf.gz"],
                store_path=tmp_path / "v",
                sample_names=["a", "b"],
            )

    def test_overwrite_existing_store(self, tmp_path, monkeypatch):
        # Covers lines 151-156: delete dir + reinit when overwrite=True on existing
        monkeypatch.setattr(
            "quantnado.dataset.variants._read_vcf",
            lambda *a, **k: {"chr1": _fake_vcf_data("chr1", [100])},
        )
        VariantStore.from_vcf_files(
            vcf_files=[tmp_path / "s1.vcf.gz"],
            store_path=tmp_path / "v",
            sample_names=["s1"],
        )
        store2 = VariantStore.from_vcf_files(
            vcf_files=[tmp_path / "s1.vcf.gz"],
            store_path=tmp_path / "v",
            sample_names=["s1"],
            overwrite=True,
        )
        assert len(store2.chromosomes) == 1

    def test_validate_sample_names_missing_attr_raises(self, tmp_path):
        # Covers line 233: missing sample_names in root attrs during resume
        import zarr
        from zarr.storage import LocalStore
        store_path = tmp_path / "bad.zarr"
        root = zarr.group(store=LocalStore(str(store_path)), overwrite=True, zarr_format=3)
        meta = root.create_group("metadata")
        meta.create_array("completed", shape=(1,), dtype=bool, fill_value=False)
        with pytest.raises(ValueError, match="missing sample_names"):
            VariantStore(store_path, sample_names=["s1"], overwrite=False, resume=True)

    def test_from_vcf_with_metadata_df(self, tmp_path, monkeypatch):
        # Covers lines 394-398: passing metadata DataFrame to from_vcf_files
        monkeypatch.setattr(
            "quantnado.dataset.variants._read_vcf",
            lambda *a, **k: {"chr1": _fake_vcf_data("chr1", [100])},
        )
        md = pd.DataFrame({"sample_id": ["s1"], "condition": ["ctrl"]})
        store = VariantStore.from_vcf_files(
            vcf_files=[tmp_path / "s1.vcf.gz"],
            store_path=tmp_path / "v",
            sample_names=["s1"],
            metadata=md,
        )
        result = store.get_metadata()
        assert "condition" in result.columns


# ---------------------------------------------------------------------------
# VariantStore.open
# ---------------------------------------------------------------------------

class TestVariantStoreOpen:
    def test_open_existing(self, two_sample_store):
        store2 = VariantStore.open(two_sample_store.store_path)
        assert store2.sample_names == ["s1", "s2"]

    def test_open_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            VariantStore.open(tmp_path / "missing.zarr")

    def test_open_missing_sample_names_raises(self, tmp_path):
        # Covers lines 180-181: KeyError in open() when sample_names attr absent
        import zarr
        from zarr.storage import LocalStore
        store_path = tmp_path / "noattr.zarr"
        root = zarr.group(store=LocalStore(str(store_path)), overwrite=True, zarr_format=3)
        with pytest.raises(ValueError, match="Missing required attribute"):
            VariantStore.open(store_path)


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------

class TestVariantStoreDataAccess:
    def test_chromosomes_property(self, two_sample_store):
        assert set(two_sample_store.chromosomes) == {"chr1", "chr2"}

    def test_get_positions_chr1(self, two_sample_store):
        positions = two_sample_store.get_positions("chr1")
        # union of [100,200,300] and [200,300,400]
        assert list(positions) == [100, 200, 300, 400]

    def test_get_positions_chr2(self, two_sample_store):
        positions = two_sample_store.get_positions("chr2")
        assert list(positions) == [50, 150]

    def test_get_alleles(self, two_sample_store):
        refs, alts = two_sample_store.get_alleles("chr1")
        assert len(refs) == 4
        assert all(r == "A" for r in refs)

    def test_completed_mask_all_true(self, two_sample_store):
        assert two_sample_store.completed_mask.all()

    def test_to_xarray_returns_dict(self, two_sample_store):
        result = two_sample_store.to_xarray()
        assert isinstance(result, dict)
        assert "chr1" in result

    def test_to_xarray_genotype_dims(self, two_sample_store):
        result = two_sample_store.to_xarray(variable="genotype")
        da = result["chr1"]
        assert da.dims == ("sample", "position")
        assert da.shape == (2, 4)

    def test_to_xarray_invalid_variable_raises(self, two_sample_store):
        with pytest.raises(ValueError, match="variable"):
            two_sample_store.to_xarray(variable="bad")

    def test_to_xarray_invalid_chrom_raises(self, two_sample_store):
        with pytest.raises(ValueError, match="not in store"):
            two_sample_store.to_xarray(chromosomes=["chrX"])

    def test_to_xarray_subset_chromosomes(self, two_sample_store):
        result = two_sample_store.to_xarray(chromosomes=["chr1"])
        assert list(result.keys()) == ["chr1"]

    def test_to_xarray_allele_depth(self, two_sample_store):
        result = two_sample_store.to_xarray(variable="allele_depth_ref")
        da = result["chr1"]
        assert da.dtype == np.int32

    def test_genotype_missing_filled_with_minus_one(self, two_sample_store):
        # s2 does not have position 100 → should be -1 (GT_MISSING)
        result = two_sample_store.to_xarray(variable="genotype")
        gt_chr1 = result["chr1"].values  # shape (2, 4)
        # s2 (index 1) at position 100 (index 0) should be missing
        assert gt_chr1[1, 0] == GT_MISSING


# ---------------------------------------------------------------------------
# extract_region
# ---------------------------------------------------------------------------

class TestExtractRegion:
    def test_region_string(self, two_sample_store):
        result = two_sample_store.extract_region("chr1:150-350")
        assert isinstance(result, xr.DataArray)
        # positions 200, 300 in [150,350]
        assert result.shape[1] == 2

    def test_chrom_start_end(self, two_sample_store):
        result = two_sample_store.extract_region(chrom="chr1", start=200, end=300)
        assert result.shape == (2, 2)

    def test_as_numpy(self, two_sample_store):
        result = two_sample_store.extract_region("chr1", as_xarray=False)
        assert isinstance(result, np.ndarray)

    def test_subset_samples_by_name(self, two_sample_store):
        result = two_sample_store.extract_region("chr1", samples=["s1"])
        assert result.shape[0] == 1

    def test_subset_samples_by_index(self, two_sample_store):
        result = two_sample_store.extract_region("chr1", samples=[0])
        assert result.shape[0] == 1

    def test_unknown_sample_raises(self, two_sample_store):
        with pytest.raises(ValueError, match="not found"):
            two_sample_store.extract_region("chr1", samples=["ghost"])

    def test_both_region_and_chrom_raises(self, two_sample_store):
        with pytest.raises(ValueError, match="not both"):
            two_sample_store.extract_region(region="chr1:0-100", chrom="chr1")

    def test_missing_chrom_raises(self, two_sample_store):
        with pytest.raises(ValueError):
            two_sample_store.extract_region(chrom=None)

    def test_unknown_chrom_raises(self, two_sample_store):
        with pytest.raises(ValueError, match="not in store"):
            two_sample_store.extract_region(chrom="chrZ")


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

class TestVariantStoreMetadata:
    def test_set_and_get_metadata(self, two_sample_store):
        md = pd.DataFrame({"sample_id": ["s1", "s2"], "condition": ["ctrl", "treat"]})
        two_sample_store.set_metadata(md)
        result = two_sample_store.get_metadata()
        assert "condition" in result.columns

    def test_missing_sample_column_raises(self, two_sample_store):
        md = pd.DataFrame({"name": ["s1", "s2"]})
        with pytest.raises(ValueError, match="sample_id"):
            two_sample_store.set_metadata(md)

    def test_read_only_set_metadata_raises(self, two_sample_store):
        store_ro = VariantStore.open(two_sample_store.store_path, read_only=True)
        md = pd.DataFrame({"sample_id": ["s1", "s2"], "x": [1, 2]})
        with pytest.raises(RuntimeError, match="read-only"):
            store_ro.set_metadata(md)

"""Integration tests for MultiomicsStore."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantnado.dataset.store_methyl import MethylStore
from quantnado.dataset.store_multiomics import MultiomicsStore
from quantnado.dataset.store_variants import VariantStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_bedgraph_data(chrom: str, positions: list[int], pct: float = 75.0):
    n = len(positions)
    return pd.DataFrame({
        "chrom": [chrom] * n,
        "start": positions,
        "end": [p + 2 for p in positions],
        "methylation_pct": np.full(n, pct, dtype=np.float32),
        "n_unmethylated": np.ones(n, dtype=np.uint16),
        "n_methylated": np.full(n, 3, dtype=np.uint16),
    })


def _fake_vcf_data(chrom: str, positions: list[int]):
    n = len(positions)
    return pd.DataFrame({
        "chrom": [chrom] * n,
        "pos": np.array(positions, dtype=np.int64),
        "ref": ["A"] * n,
        "alt": ["T"] * n,
        "qual": np.full(n, 30.0, dtype=np.float32),
        "genotype": np.ones(n, dtype=np.int8),
        "ad_ref": np.full(n, 10, dtype=np.int32),
        "ad_alt": np.full(n, 5, dtype=np.int32),
    })


def _make_bg_reader(chrom="chr1", positions=None, pct=75.0):
    positions = positions or [100, 200, 300]
    data = {chrom: _fake_bedgraph_data(chrom, positions, pct)}
    return lambda path, filter_chromosomes=True: data


def _make_vcf_reader(chrom="chr1", positions=None):
    positions = positions or [100, 200]
    data = {chrom: _fake_vcf_data(chrom, positions)}
    return lambda path, filter_chromosomes=True: data


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def meth_only_store(tmp_path, monkeypatch):
    """MultiomicsStore with only a methylation sub-store."""
    monkeypatch.setattr(
        "quantnado.dataset.store_methyl._read_bedgraph",
        _make_bg_reader(),
    )
    store = MultiomicsStore.from_files(
        store_dir=tmp_path / "ms",
        methyldackel_files=[tmp_path / "s1.bedGraph"],
        bedgraph_sample_names=["s1"],
    )
    return store


@pytest.fixture
def meth_and_variant_store(tmp_path, monkeypatch):
    """MultiomicsStore with methylation and variants sub-stores."""
    monkeypatch.setattr(
        "quantnado.dataset.store_methyl._read_bedgraph",
        _make_bg_reader(),
    )
    monkeypatch.setattr(
        "quantnado.dataset.store_variants._read_vcf",
        _make_vcf_reader(),
    )
    store = MultiomicsStore.from_files(
        store_dir=tmp_path / "ms",
        methyldackel_files=[tmp_path / "s1.bedGraph"],
        bedgraph_sample_names=["s1"],
        vcf_files=[tmp_path / "v1.vcf.gz"],
        vcf_sample_names=["v1"],
    )
    return store


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestMultiomicsConstruction:
    def test_no_files_raises(self, tmp_path):
        with pytest.raises(ValueError, match="at least one"):
            MultiomicsStore.from_files(store_dir=tmp_path / "ms")

    def test_store_dir_created(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "quantnado.dataset.store_methyl._read_bedgraph",
            _make_bg_reader(),
        )
        store_dir = tmp_path / "new_dir"
        assert not store_dir.exists()
        MultiomicsStore.from_files(
            store_dir=store_dir,
            methyldackel_files=[tmp_path / "s1.bedGraph"],
            bedgraph_sample_names=["s1"],
        )
        assert store_dir.exists()

    def test_open_existing(self, meth_only_store, tmp_path):
        store2 = MultiomicsStore.open(meth_only_store.store_dir)
        assert "methylation" in store2.modalities

    def test_open_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            MultiomicsStore.open(tmp_path / "nonexistent")

    def test_empty_dir_warns_no_modalities(self, tmp_path):
        """Opening a directory with no sub-stores should log a warning but not raise."""
        (tmp_path / "empty_ms").mkdir()
        store = MultiomicsStore(tmp_path / "empty_ms")
        assert store.modalities == []


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

class TestMultiomicsProperties:
    def test_modalities_meth_only(self, meth_only_store):
        assert meth_only_store.modalities == ["methylation"]

    def test_modalities_meth_and_variants(self, meth_and_variant_store):
        assert set(meth_and_variant_store.modalities) == {"methylation", "variants"}

    def test_chromosomes_union(self, meth_and_variant_store):
        chroms = meth_and_variant_store.chromosomes
        assert "chr1" in chroms

    def test_samples_dict(self, meth_only_store):
        samples = meth_only_store.samples
        assert "methylation" in samples
        assert "s1" in samples["methylation"]

    def test_all_sample_names_deduped(self, meth_and_variant_store):
        names = meth_and_variant_store.all_sample_names
        assert len(names) == len(set(names))

    def test_repr_contains_modality(self, meth_only_store):
        r = repr(meth_only_store)
        assert "methylation" in r


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

class TestMultiomicsMetadata:
    def test_get_metadata_returns_dataframe(self, meth_only_store):
        result = meth_only_store.get_metadata()
        assert isinstance(result, pd.DataFrame)

    def test_get_metadata_modalities_column(self, meth_only_store):
        result = meth_only_store.get_metadata()
        assert "modalities" in result.columns

    def test_set_metadata_does_not_raise(self, meth_only_store):
        # Sub-stores are opened read-only; set_metadata catches and warns internally
        md = pd.DataFrame({"sample_id": ["s1"], "condition": ["ctrl"]})
        meth_only_store.set_metadata(md)  # should not raise

    def test_get_metadata_empty_store(self, tmp_path):
        (tmp_path / "empty").mkdir()
        store = MultiomicsStore(tmp_path / "empty")
        result = store.get_metadata()
        assert result.empty

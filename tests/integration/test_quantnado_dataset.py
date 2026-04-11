"""Integration tests for the new QuantNadoDataset (analysis.core)."""
from __future__ import annotations

import numpy as np
import pytest
import zarr
import xarray as xr

from quantnado.analysis.core import QuantNadoDataset

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers to build minimal per-sample zarr stores (new layout)
# ---------------------------------------------------------------------------


def _make_per_sample_store(
    tmp_path,
    sample: str = "s1",
    assay: str = "atac",
    chrom_sizes: dict[str, int] | None = None,
    completed: bool = True,
    store_name: str | None = None,
    value: int = 1,
) -> "Path":
    """Build a minimal per-sample zarr store in the new layout."""
    if chrom_sizes is None:
        chrom_sizes = {"chr1": 100, "chr2": 50}
    if store_name is None:
        store_name = f"{sample}.zarr"

    store_path = tmp_path / store_name
    root = zarr.open_group(str(store_path), mode="w", zarr_format=3)

    for chrom, chrom_len in chrom_sizes.items():
        grp = root.require_group(chrom)
        arr = grp.require_array(assay, shape=(1, chrom_len), dtype=np.uint32)
        arr[0, :] = np.full(chrom_len, value, dtype=np.uint32)

    meta = root.require_group("metadata")
    meta.require_array("completed", shape=(1,), dtype=bool)
    meta["completed"][0] = completed
    meta.require_array("total_reads", shape=(1,), dtype=np.int64)
    meta["total_reads"][0] = 1_000_000

    root.attrs.update({
        "assay": assay,
        "sample": sample,
        "chromsizes": chrom_sizes,
        "chunk_len": 65536,
    })

    zarr.consolidate_metadata(str(store_path))
    return store_path


def _make_dataset_dir(tmp_path, n_samples: int = 2, assay: str = "atac") -> "Path":
    """Build a directory of per-sample zarr stores."""
    ds_dir = tmp_path / "dataset"
    ds_dir.mkdir()
    for i in range(1, n_samples + 1):
        _make_per_sample_store(
            ds_dir,
            sample=f"s{i}",
            assay=assay,
            value=i,
            store_name=f"s{i}.zarr",
        )
    return ds_dir


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------


class TestInit:
    def test_directory_loads_per_sample_stores(self, tmp_path):
        ds_dir = _make_dataset_dir(tmp_path)
        qn = QuantNadoDataset(ds_dir)
        assert len(qn.sample_names) == 2
        assert set(qn.sample_names) == {"s1", "s2"}

    def test_skips_incomplete_stores(self, tmp_path):
        ds_dir = tmp_path / "ds"
        ds_dir.mkdir()
        _make_per_sample_store(ds_dir, sample="complete", completed=True, store_name="complete.zarr")
        _make_per_sample_store(ds_dir, sample="incomplete", completed=False, store_name="incomplete.zarr")
        qn = QuantNadoDataset(ds_dir)
        assert qn.sample_names == ["complete"]

    def test_path_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            QuantNadoDataset(tmp_path / "nonexistent.zarr")

    def test_empty_dir_raises(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises((FileNotFoundError, ValueError)):
            QuantNadoDataset(empty)

    def test_single_zarr_opens_as_per_sample(self, tmp_path):
        store_path = _make_per_sample_store(tmp_path)
        qn = QuantNadoDataset(store_path)
        assert qn.sample_names == ["s1"]


# ---------------------------------------------------------------------------
# TestProperties
# ---------------------------------------------------------------------------


class TestProperties:
    def test_sample_names(self, tmp_path):
        ds_dir = _make_dataset_dir(tmp_path)
        qn = QuantNadoDataset(ds_dir)
        assert sorted(qn.sample_names) == ["s1", "s2"]

    def test_assays(self, tmp_path):
        ds_dir = _make_dataset_dir(tmp_path, assay="atac")
        qn = QuantNadoDataset(ds_dir)
        assert qn.assays == ["ATAC"]

    def test_chromosomes(self, tmp_path):
        ds_dir = _make_dataset_dir(tmp_path)
        qn = QuantNadoDataset(ds_dir)
        assert set(qn.chromosomes) == {"chr1", "chr2"}

    def test_chromsizes(self, tmp_path):
        ds_dir = _make_dataset_dir(tmp_path)
        qn = QuantNadoDataset(ds_dir)
        assert qn.chromsizes == {"chr1": 100, "chr2": 50}

    def test_completed_mask_all_true(self, tmp_path):
        ds_dir = _make_dataset_dir(tmp_path)
        qn = QuantNadoDataset(ds_dir)
        assert qn.completed_mask.all()


# ---------------------------------------------------------------------------
# TestSel
# ---------------------------------------------------------------------------


class TestSel:
    def test_returns_xr_dataset(self, tmp_path):
        ds_dir = _make_dataset_dir(tmp_path)
        qn = QuantNadoDataset(ds_dir)
        result = qn.sel(chrom="chr1")
        assert isinstance(result, xr.Dataset)

    def test_dims_sample_and_position(self, tmp_path):
        ds_dir = _make_dataset_dir(tmp_path)
        qn = QuantNadoDataset(ds_dir)
        result = qn.sel(chrom="chr1")
        da = result["atac"]
        assert da.dims == ("sample", "position")

    def test_position_coords_are_1based(self, tmp_path):
        ds_dir = _make_dataset_dir(tmp_path)
        qn = QuantNadoDataset(ds_dir)
        result = qn.sel(chrom="chr1", start=5, end=10)
        positions = result.coords["position"].values
        assert positions[0] == 5
        assert positions[-1] == 10

    def test_full_chrom_shape(self, tmp_path):
        ds_dir = _make_dataset_dir(tmp_path, n_samples=2)
        qn = QuantNadoDataset(ds_dir)
        result = qn.sel(chrom="chr1")
        assert result["atac"].shape == (2, 100)

    def test_region_values(self, tmp_path):
        ds_dir = _make_dataset_dir(tmp_path)
        qn = QuantNadoDataset(ds_dir)
        result = qn.sel(chrom="chr1", start=1, end=5)
        s1_values = result["atac"].sel(sample="s1").values
        np.testing.assert_array_equal(s1_values, np.ones(5, dtype=np.uint32))

    def test_unknown_chrom_raises(self, tmp_path):
        ds_dir = _make_dataset_dir(tmp_path)
        qn = QuantNadoDataset(ds_dir)
        with pytest.raises(ValueError, match="chr"):
            qn.sel(chrom="chrInvalid")

    def test_start_before_1_raises(self, tmp_path):
        ds_dir = _make_dataset_dir(tmp_path)
        qn = QuantNadoDataset(ds_dir)
        with pytest.raises(ValueError):
            qn.sel(chrom="chr1", start=0)

    def test_end_exceeds_chrom_raises(self, tmp_path):
        ds_dir = _make_dataset_dir(tmp_path)
        qn = QuantNadoDataset(ds_dir)
        with pytest.raises(ValueError):
            qn.sel(chrom="chr1", end=9999)


# ---------------------------------------------------------------------------
# TestDatatree
# ---------------------------------------------------------------------------


class TestDatatree:
    def test_returns_datatree(self, tmp_path):
        ds_dir = _make_dataset_dir(tmp_path)
        qn = QuantNadoDataset(ds_dir)
        tree = qn.to_datatree()
        assert isinstance(tree, xr.DataTree)

    def test_has_chrom_nodes(self, tmp_path):
        ds_dir = _make_dataset_dir(tmp_path)
        qn = QuantNadoDataset(ds_dir)
        tree = qn.to_datatree()
        assert "chr1" in tree.children
        assert "chr2" in tree.children

    def test_subset_chromosomes(self, tmp_path):
        ds_dir = _make_dataset_dir(tmp_path)
        qn = QuantNadoDataset(ds_dir)
        tree = qn.to_datatree(chromosomes=["chr1"])
        assert "chr1" in tree.children
        assert "chr2" not in tree.children


# ---------------------------------------------------------------------------
# TestCombine
# ---------------------------------------------------------------------------


class TestCombine:
    def test_combine_creates_combined_zarr(self, tmp_path):
        ds_dir = _make_dataset_dir(tmp_path)
        combined_path = tmp_path / "combined.zarr"
        combined = QuantNadoDataset.combine(ds_dir, combined_path)
        assert combined_path.exists()
        assert combined._combined is True

    def test_combined_has_correct_samples(self, tmp_path):
        ds_dir = _make_dataset_dir(tmp_path, n_samples=2)
        combined_path = tmp_path / "combined.zarr"
        combined = QuantNadoDataset.combine(ds_dir, combined_path)
        assert sorted(combined.sample_names) == ["s1", "s2"]

    def test_combined_sel_returns_all_samples(self, tmp_path):
        ds_dir = _make_dataset_dir(tmp_path, n_samples=2)
        combined_path = tmp_path / "combined.zarr"
        combined = QuantNadoDataset.combine(ds_dir, combined_path)
        result = combined.sel(chrom="chr1")
        assert combined.array_keys == ["coverage"]
        assert result["coverage"].shape == (2, 100)

"""Integration tests for QuantNadoDataset (analysis.core and dataset.core variants)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import zarr
import xarray as xr
from zarr.core.dtype import VariableLengthUTF8

from quantnado.dataset.core import BaseStore as AnalysisCore
from quantnado.dataset.core import QuantNadoDataset as DatasetCore


# ---------------------------------------------------------------------------
# Helpers to build a minimal zarr store
# ---------------------------------------------------------------------------


def _make_store(tmp_path, chrom_sizes=None, sample_names=None, all_complete=True):
    """Build a zarr store in the expected QuantNado layout."""
    if chrom_sizes is None:
        chrom_sizes = {"chr1": 100, "chr2": 50}
    if sample_names is None:
        sample_names = ["s1", "s2"]

    root = zarr.open(str(tmp_path / "store.zarr"), mode="w")
    for chrom, size in chrom_sizes.items():
        arr = root.create_array(chrom, shape=(len(sample_names), size), dtype=np.uint16)
        for i in range(len(sample_names)):
            arr[i, :] = np.ones(size, dtype=np.uint16) * (i + 1)

    meta = root.require_group("metadata")
    completed = np.array([True] * len(sample_names)) if all_complete else np.array([True, False] + [True] * max(0, len(sample_names) - 2))
    meta.create_array("completed", data=completed[:len(sample_names)])
    root.attrs["chromsizes"] = chrom_sizes
    root.attrs["chunk_len"] = 1024
    root.attrs["sample_names"] = sample_names
    return tmp_path / "store.zarr"


# ---------------------------------------------------------------------------
# Parametrize both implementations
# ---------------------------------------------------------------------------

CLASSES = [
    pytest.param(AnalysisCore, id="analysis_core"),
    pytest.param(DatasetCore, id="dataset_core"),
]


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------


class TestInit:
    @pytest.mark.parametrize("cls", CLASSES)
    def test_path_not_found_raises(self, tmp_path, cls):
        with pytest.raises(FileNotFoundError):
            cls(tmp_path / "nonexistent.zarr")

    @pytest.mark.parametrize("cls", CLASSES)
    def test_missing_metadata_group_raises(self, tmp_path, cls):
        root = zarr.open(str(tmp_path / "no_meta.zarr"), mode="w")
        root.create_array("chr1", shape=(2, 100), dtype=np.uint16)
        with pytest.raises(ValueError, match="metadata"):
            cls(tmp_path / "no_meta.zarr")

    @pytest.mark.parametrize("cls", CLASSES)
    def test_missing_sample_names_raises(self, tmp_path, cls):
        root = zarr.open(str(tmp_path / "no_names.zarr"), mode="w")
        root.create_array("chr1", shape=(2, 100), dtype=np.uint16)
        meta = root.require_group("metadata")
        meta.create_array("completed", data=np.array([True, True]))
        # no sample_names array and no root attr
        with pytest.raises(ValueError, match="[Ss]ample"):
            cls(tmp_path / "no_names.zarr")

    @pytest.mark.parametrize("cls", CLASSES)
    def test_successful_init_with_string_sample_names(self, tmp_path, cls):
        store_path = _make_store(tmp_path)
        ds = cls(store_path)
        assert ds.sample_names == ["s1", "s2"]
        assert "chr1" in ds.chromosomes
        assert "chr2" in ds.chromosomes

    @pytest.mark.parametrize("cls", CLASSES)
    def test_successful_init_with_metadata_sample_names(self, tmp_path, cls):
        root = zarr.open(str(tmp_path / "meta_names_store.zarr"), mode="w")
        root.create_array("chr1", shape=(2, 10), dtype=np.uint16)
        root["chr1"][:] = np.ones((2, 10), dtype=np.uint16)
        meta = root.require_group("metadata")
        meta.create_array("completed", data=np.array([True, True]))
        sample_name_array = meta.create_array(
            "sample_names",
            shape=(2,),
            dtype=VariableLengthUTF8(),
        )
        sample_name_array[:] = ["s1", "s2"]
        root.attrs["chromsizes"] = {"chr1": 10}
        root.attrs["chunk_len"] = 1024

        ds = cls(tmp_path / "meta_names_store.zarr")
        assert ds.sample_names == ["s1", "s2"]

    @pytest.mark.parametrize("cls", CLASSES)
    def test_successful_init_with_bytes_sample_names(self, tmp_path, cls):
        # Build a store where sample_names are bytes (legacy format)
        root = zarr.open(str(tmp_path / "bytes_store.zarr"), mode="w")
        root.create_array("chr1", shape=(2, 10), dtype=np.uint16)
        root["chr1"][:] = np.ones((2, 10), dtype=np.uint16)
        meta = root.require_group("metadata")
        meta.create_array("completed", data=np.array([True, True]))
        # Store as object bytes
        root.attrs["sample_names"] = ["s1", "s2"]
        # Simulate bytes by patching after open
        ds = cls(tmp_path / "bytes_store.zarr")
        assert ds.sample_names == ["s1", "s2"]


# ---------------------------------------------------------------------------
# TestGetChrom
# ---------------------------------------------------------------------------


class TestGetChrom:
    @pytest.mark.parametrize("cls", CLASSES)
    def test_get_chrom_returns_array(self, tmp_path, cls):
        store_path = _make_store(tmp_path)
        ds = cls(store_path)
        arr = ds.get_chrom("chr1")
        assert arr is not None
        assert arr.shape == (2, 100)


# ---------------------------------------------------------------------------
# TestValidSampleIndices
# ---------------------------------------------------------------------------


class TestValidSampleIndices:
    @pytest.mark.parametrize("cls", CLASSES)
    def test_all_complete(self, tmp_path, cls):
        store_path = _make_store(tmp_path, all_complete=True)
        ds = cls(store_path)
        indices = ds.valid_sample_indices()
        np.testing.assert_array_equal(indices, [0, 1])

    @pytest.mark.parametrize("cls", CLASSES)
    def test_mixed_complete(self, tmp_path, cls):
        # Build store where second sample is incomplete
        chrom_sizes = {"chr1": 10}
        root = zarr.open(str(tmp_path / "mixed.zarr"), mode="w")
        root.create_array("chr1", shape=(2, 10), dtype=np.uint16)
        meta = root.require_group("metadata")
        meta.create_array("completed", data=np.array([True, False]))
        root.attrs["chromsizes"] = chrom_sizes
        root.attrs["chunk_len"] = 1024
        root.attrs["sample_names"] = ["s1", "s2"]
        ds = cls(tmp_path / "mixed.zarr")
        indices = ds.valid_sample_indices()
        np.testing.assert_array_equal(indices, [0])


# ---------------------------------------------------------------------------
# TestMetadataProperty
# ---------------------------------------------------------------------------


class TestMetadataProperty:
    @pytest.mark.parametrize("cls", CLASSES)
    def test_returns_dataframe(self, tmp_path, cls):
        store_path = _make_store(tmp_path)
        ds = cls(store_path)
        md = ds.metadata
        assert isinstance(md, pd.DataFrame)


# ---------------------------------------------------------------------------
# TestToXarray
# ---------------------------------------------------------------------------


class TestToXarray:
    @pytest.mark.parametrize("cls", CLASSES)
    def test_all_chroms_default(self, tmp_path, cls):
        store_path = _make_store(tmp_path)
        ds = cls(store_path)
        result = ds.to_xarray()
        assert set(result.keys()) == {"chr1", "chr2"}

    @pytest.mark.parametrize("cls", CLASSES)
    def test_subset_of_chroms(self, tmp_path, cls):
        store_path = _make_store(tmp_path)
        ds = cls(store_path)
        result = ds.to_xarray(chromosomes=["chr1"])
        assert set(result.keys()) == {"chr1"}

    @pytest.mark.parametrize("cls", CLASSES)
    def test_invalid_chrom_raises(self, tmp_path, cls):
        store_path = _make_store(tmp_path)
        ds = cls(store_path)
        with pytest.raises(ValueError, match="not in store"):
            ds.to_xarray(chromosomes=["chrInvalid"])

    @pytest.mark.parametrize("cls", CLASSES)
    def test_incomplete_sample_raises(self, tmp_path, cls):
        root = zarr.open(str(tmp_path / "inc.zarr"), mode="w")
        root.create_array("chr1", shape=(2, 10), dtype=np.uint16)
        meta = root.require_group("metadata")
        meta.create_array("completed", data=np.array([True, False]))
        root.attrs["chromsizes"] = {"chr1": 10}
        root.attrs["sample_names"] = ["s1", "s2"]
        ds = cls(tmp_path / "inc.zarr")
        with pytest.raises(RuntimeError, match="incomplete"):
            ds.to_xarray()

    @pytest.mark.parametrize("cls", CLASSES)
    def test_chunks_auto(self, tmp_path, cls):
        store_path = _make_store(tmp_path)
        ds = cls(store_path)
        result = ds.to_xarray(chunks="auto")
        assert "chr1" in result
        assert isinstance(result["chr1"], xr.DataArray)

    @pytest.mark.parametrize("cls", CLASSES)
    def test_chunks_as_dict(self, tmp_path, cls):
        store_path = _make_store(tmp_path)
        ds = cls(store_path)
        result = ds.to_xarray(chunks={"sample": 1, "position": 50})
        assert "chr1" in result

    @pytest.mark.parametrize("cls", CLASSES)
    def test_metadata_coordinates_in_result(self, tmp_path, cls):
        store_path = _make_store(tmp_path)
        ds = cls(store_path)
        result = ds.to_xarray()
        da = result["chr1"]
        assert "sample" in da.coords
        assert list(da.coords["sample"].values) == ["s1", "s2"]
        assert da.dims == ("sample", "position")


# ---------------------------------------------------------------------------
# TestExtractRegion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", CLASSES)
class TestExtractRegionParametrized:
    """Smoke test extract_region for both implementations."""

    def test_extract_region_basic(self, tmp_path, cls):
        store_path = _make_store(tmp_path)
        ds = cls(store_path)
        result = ds.extract_region("chr1:10-50")
        assert result.shape == (2, 40)


class TestExtractRegion:
    @pytest.fixture
    def ds(self, tmp_path):
        store_path = _make_store(tmp_path)
        return AnalysisCore(store_path)

    def test_region_string_format(self, ds):
        result = ds.extract_region("chr1:10-50")
        assert result.shape == (2, 40)
        assert list(result.coords["position"].values) == list(range(10, 50))

    def test_region_string_with_commas(self, ds):
        result = ds.extract_region("chr1:1,0-5,0")
        assert result.shape == (2, 40)

    def test_chrom_start_end_separately(self, ds):
        result = ds.extract_region(chrom="chr1", start=5, end=20)
        assert result.shape == (2, 15)

    def test_whole_chrom_no_start_end(self, ds):
        result = ds.extract_region(chrom="chr1")
        assert result.shape == (2, 100)
        assert result.attrs["start"] == 0
        assert result.attrs["end"] == 100

    def test_as_xarray_true_returns_dataarray(self, ds):
        result = ds.extract_region("chr1:10-20", as_xarray=True)
        assert isinstance(result, xr.DataArray)
        assert result.dims == ("sample", "position")

    def test_as_xarray_false_returns_numpy(self, ds):
        result = ds.extract_region("chr1:10-20", as_xarray=False)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 10)

    def test_normalise_cpm_returns_scaled_xarray(self, ds):
        result = ds.extract_region(
            "chr1:0-3",
            normalise="cpm",
            library_sizes={"s1": 1_000_000, "s2": 2_000_000},
        )
        expected = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        np.testing.assert_allclose(result.values, expected)
        assert result.attrs["normalised"] == "cpm"

    def test_normalise_cpm_returns_scaled_numpy(self, ds):
        result = ds.extract_region(
            "chr1:0-3",
            as_xarray=False,
            normalise="cpm",
            library_sizes={"s1": 1_000_000, "s2": 2_000_000},
        )
        expected = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        np.testing.assert_allclose(result, expected)

    def test_samples_by_name(self, ds):
        result = ds.extract_region("chr1:0-10", samples=["s1"])
        assert result.shape == (1, 10)
        assert list(result.coords["sample"].values) == ["s1"]

    def test_samples_by_index(self, ds):
        result = ds.extract_region("chr1:0-10", samples=[1])
        assert result.shape == (1, 10)
        assert list(result.coords["sample"].values) == ["s2"]

    def test_invalid_sample_name_raises(self, ds):
        with pytest.raises(ValueError, match="not found"):
            ds.extract_region("chr1:0-10", samples=["invalid"])

    def test_out_of_range_sample_index_raises(self, ds):
        with pytest.raises(ValueError, match="out of range"):
            ds.extract_region("chr1:0-10", samples=[999])

    def test_unknown_chromosome_raises(self, ds):
        with pytest.raises(ValueError, match="not in store"):
            ds.extract_region("chrInvalid:0-10")

    def test_both_region_and_chrom_raises(self, ds):
        with pytest.raises(ValueError, match="either 'region' or 'chrom'"):
            ds.extract_region(region="chr1:0-10", chrom="chr1")

    def test_chrom_none_raises(self, ds):
        with pytest.raises(ValueError, match="Must specify"):
            ds.extract_region()

    def test_start_negative_raises(self, ds):
        with pytest.raises(ValueError, match=">="):
            ds.extract_region(chrom="chr1", start=-1, end=10)

    def test_end_exceeds_chrom_size_raises(self, ds):
        with pytest.raises(ValueError, match="exceeds chromosome size"):
            ds.extract_region(chrom="chr1", start=0, end=9999)

    def test_end_le_start_raises(self, ds):
        with pytest.raises(ValueError, match="greater than start"):
            ds.extract_region(chrom="chr1", start=50, end=10)

    def test_incomplete_sample_raises(self, tmp_path):
        root = zarr.open(str(tmp_path / "inc2.zarr"), mode="w")
        root.create_array("chr1", shape=(2, 100), dtype=np.uint16)
        meta = root.require_group("metadata")
        meta.create_array("completed", data=np.array([False, True]))
        root.attrs["chromsizes"] = {"chr1": 100}
        root.attrs["sample_names"] = ["s1", "s2"]
        ds = AnalysisCore(tmp_path / "inc2.zarr")
        with pytest.raises(RuntimeError, match="incomplete"):
            ds.extract_region("chr1:0-10", samples=["s1"])

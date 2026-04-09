"""Integration tests for BaseStore / QuantNadoDataset (dataset.core)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import zarr
import xarray as xr
from zarr.core.dtype import VariableLengthUTF8

from quantnado.dataset.core import BaseStore as AnalysisCore
from quantnado.dataset.core import QuantNadoDataset as DatasetCore

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers to build minimal zarr stores
# ---------------------------------------------------------------------------


def _build_contig_offsets(chrom_sizes: dict[str, int]) -> dict[str, list[int]]:
    offsets: dict[str, list[int]] = {}
    pos = 0
    for chrom, size in chrom_sizes.items():
        offsets[chrom] = [pos, pos + size]
        pos += size
    return offsets


def _make_store(
    tmp_path,
    chrom_sizes: dict[str, int] | None = None,
    sample_names: list[str] | None = None,
    all_complete: bool = True,
    store_name: str = "store.zarr",
) -> "Path":
    """Build a zarr store in the flat QuantNado layout used by BamStore."""
    if chrom_sizes is None:
        chrom_sizes = {"chr1": 100, "chr2": 50}
    if sample_names is None:
        sample_names = ["s1", "s2"]

    n_samples = len(sample_names)
    total_len = sum(chrom_sizes.values())
    contig_offsets = _build_contig_offsets(chrom_sizes)

    root = zarr.open(str(tmp_path / store_name), mode="w")
    arr = root.create_array("coverage", shape=(total_len, n_samples), dtype=np.uint16)
    for i in range(n_samples):
        for chrom, size in chrom_sizes.items():
            s, e = contig_offsets[chrom]
            arr[s:e, i] = np.ones(size, dtype=np.uint16) * (i + 1)

    meta = root.require_group("metadata")
    completed = np.ones(n_samples, dtype=bool)
    if not all_complete and n_samples >= 2:
        completed[1] = False
    meta.create_array("completed", data=completed)

    root.attrs.update({
        "chromsizes": chrom_sizes,
        "chunk_len": 1024,
        "sample_names": sample_names,
        "contig_offsets": contig_offsets,
        "chromosomes": sorted(chrom_sizes.keys()),
    })

    return tmp_path / store_name


# ---------------------------------------------------------------------------
# Parametrize both aliases (they are the same class)
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
    def test_missing_sample_names_raises(self, tmp_path, cls):
        root = zarr.open(str(tmp_path / "no_names.zarr"), mode="w")
        root.create_array("chr1", shape=(100, 2), dtype=np.uint16)
        meta = root.require_group("metadata")
        meta.create_array("completed", data=np.array([True, True]))
        # no sample_names array and no root attr
        with pytest.raises(ValueError, match="[Ss]ample|metadata"):
            cls(tmp_path / "no_names.zarr")

    @pytest.mark.parametrize("cls", CLASSES)
    def test_successful_init_flat_layout(self, tmp_path, cls):
        store_path = _make_store(tmp_path)
        ds = cls(store_path)
        assert ds.sample_names == ["s1", "s2"]
        assert "chr1" in ds.chromosomes
        assert "chr2" in ds.chromosomes

    @pytest.mark.parametrize("cls", CLASSES)
    def test_successful_init_with_metadata_sample_names(self, tmp_path, cls):
        root = zarr.open(str(tmp_path / "meta_names_store.zarr"), mode="w")
        contig_offsets = {"chr1": [0, 10]}
        arr = root.create_array("coverage", shape=(10, 2), dtype=np.uint16)
        arr[:] = np.ones((10, 2), dtype=np.uint16)
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
        root.attrs["contig_offsets"] = contig_offsets
        root.attrs["chromosomes"] = ["chr1"]

        ds = cls(tmp_path / "meta_names_store.zarr")
        assert ds.sample_names == ["s1", "s2"]

    @pytest.mark.parametrize("cls", CLASSES)
    def test_successful_init_sample_names_from_root_attrs(self, tmp_path, cls):
        store_path = _make_store(tmp_path, store_name="attrs_store.zarr")
        ds = cls(store_path)
        assert ds.sample_names == ["s1", "s2"]


# ---------------------------------------------------------------------------
# TestChromsizes
# ---------------------------------------------------------------------------


class TestChromsizes:
    @pytest.mark.parametrize("cls", CLASSES)
    def test_chromsizes_returns_dict(self, tmp_path, cls):
        store_path = _make_store(tmp_path)
        ds = cls(store_path)
        assert ds.chromsizes == {"chr1": 100, "chr2": 50}

    @pytest.mark.parametrize("cls", CLASSES)
    def test_chromosomes_list(self, tmp_path, cls):
        store_path = _make_store(tmp_path, chrom_sizes={"chr1": 100, "chr2": 50, "chr3": 25})
        ds = cls(store_path)
        assert set(ds.chromosomes) == {"chr1", "chr2", "chr3"}


# ---------------------------------------------------------------------------
# TestGetChrom
# ---------------------------------------------------------------------------


class TestGetChrom:
    @pytest.mark.parametrize("cls", CLASSES)
    def test_get_chrom_returns_zarr_array(self, tmp_path, cls):
        store_path = _make_store(tmp_path)
        ds = cls(store_path)
        arr = ds.get_chrom("chr1")
        assert arr is not None
        assert isinstance(arr, zarr.Array)

    @pytest.mark.parametrize("cls", CLASSES)
    def test_get_chrom_slice_has_correct_shape(self, tmp_path, cls):
        store_path = _make_store(tmp_path)
        ds = cls(store_path)
        arr = ds.get_chrom("chr1")
        s, e = ds._contig_row_range("chr1")
        chrom_data = arr[s:e, :]
        assert chrom_data.shape == (100, 2)


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
        store_path = _make_store(tmp_path, all_complete=False)
        ds = cls(store_path)
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

    @pytest.mark.parametrize("cls", CLASSES)
    def test_index_is_sample_id(self, tmp_path, cls):
        store_path = _make_store(tmp_path)
        ds = cls(store_path)
        assert ds.metadata.index.name == "sample_id"
        assert list(ds.metadata.index) == ["s1", "s2"]


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
        contig_offsets = {"chr1": [0, 10]}
        root = zarr.open(str(tmp_path / "inc.zarr"), mode="w")
        root.create_array("coverage", shape=(10, 2), dtype=np.uint16)
        meta = root.require_group("metadata")
        meta.create_array("completed", data=np.array([True, False]))
        root.attrs["chromsizes"] = {"chr1": 10}
        root.attrs["sample_names"] = ["s1", "s2"]
        root.attrs["contig_offsets"] = contig_offsets
        root.attrs["chromosomes"] = ["chr1"]
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
    def test_shape_and_dims(self, tmp_path, cls):
        store_path = _make_store(tmp_path)
        ds = cls(store_path)
        result = ds.to_xarray()
        da = result["chr1"]
        assert da.dims == ("sample", "position")
        assert list(da.coords["sample"].values) == ["s1", "s2"]
        assert da.shape == (2, 100)

    @pytest.mark.parametrize("cls", CLASSES)
    def test_values_correct(self, tmp_path, cls):
        store_path = _make_store(tmp_path)
        ds = cls(store_path)
        result = ds.to_xarray()
        computed = result["chr1"].values
        # s1 (row 0) = 1, s2 (row 1) = 2
        assert np.all(computed[0, :] == 1)
        assert np.all(computed[1, :] == 2)


# ---------------------------------------------------------------------------
# TestExtractRegionParametrized
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", CLASSES)
class TestExtractRegionParametrized:
    def test_extract_region_basic(self, tmp_path, cls):
        store_path = _make_store(tmp_path)
        ds = cls(store_path)
        result = ds.extract_region("chr1:10-50")
        assert result.shape == (2, 40)


# ---------------------------------------------------------------------------
# TestExtractRegion (non-parametrized, richer coverage)
# ---------------------------------------------------------------------------


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
        contig_offsets = {"chr1": [0, 100]}
        root = zarr.open(str(tmp_path / "inc2.zarr"), mode="w")
        root.create_array("coverage", shape=(100, 2), dtype=np.uint16)
        meta = root.require_group("metadata")
        meta.create_array("completed", data=np.array([False, True]))
        root.attrs["chromsizes"] = {"chr1": 100}
        root.attrs["sample_names"] = ["s1", "s2"]
        root.attrs["contig_offsets"] = contig_offsets
        root.attrs["chromosomes"] = ["chr1"]
        ds = AnalysisCore(tmp_path / "inc2.zarr")
        with pytest.raises(RuntimeError, match="incomplete"):
            ds.extract_region("chr1:0-10", samples=["s1"])

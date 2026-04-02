"""Integration tests for BamStore: create, open, process, metadata, resume."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import bamnado

from quantnado.dataset.store_bam import BamStore, CoverageType
from quantnado.dataset.core import BaseStore as QuantNadoDataset
from quantnado.utils import estimate_chunk_len


# ---------------------------------------------------------------------------
# Write and basic read
# ---------------------------------------------------------------------------


def test_bamstore_write_and_metadata(tmp_path, chromsizes, sample_names, monkeypatch):
    def fake_chrom(self, bam_file, contig, contig_size, is_stranded, use_fragment=False, read_filter=None):
        return 0.0, np.full(contig_size, int(bam_file), dtype=np.uint16), None

    monkeypatch.setattr(BamStore, "_process_chromosome", fake_chrom)
    store = BamStore(tmp_path / "ds", chromsizes, sample_names)
    store.process_samples(["1", "2"])

    # New layout: coverage/chrom shape is (position, n_samples)
    assert np.all(store.root["coverage"]["chr1"][:, 0] == 1)   # sample 0 = bam "1"
    assert np.all(store.root["coverage"]["chr2"][:, 1] == 2)   # sample 1 = bam "2"
    stored = [s.decode() if isinstance(s, (bytes, bytearray)) else s for s in store.root.attrs["sample_names"]]
    assert stored == sample_names
    assert store.completed_mask.tolist() == [True, True]
    assert np.isfinite(store.meta["sparsity"][:]).all()


def test_bamstore_dataset_wrapper(tmp_path, chromsizes, sample_names, monkeypatch):
    def fake_chrom(self, bam_file, contig, contig_size, is_stranded, use_fragment=False, read_filter=None):
        return 0.0, np.full(contig_size, int(bam_file), dtype=np.uint16), None

    monkeypatch.setattr(BamStore, "_process_chromosome", fake_chrom)
    store = BamStore(tmp_path / "ds", chromsizes, sample_names)
    store.process_samples(["1", "2"])

    ds = QuantNadoDataset(store.store_path)
    assert ds.sample_names == sample_names
    assert ds.completed_mask.tolist() == [True, True]
    assert ds.chromsizes == chromsizes
    # get_chrom returns (position, n_samples); column 0 is sample s1 with value 1
    np.testing.assert_array_equal(ds.get_chrom("chr1")[:, 0], np.array([1, 1, 1, 1], dtype=np.uint32))


# ---------------------------------------------------------------------------
# Resume / overwrite validation
# ---------------------------------------------------------------------------


def test_resume_validates_sample_names(tmp_path, chromsizes, sample_names, monkeypatch):
    monkeypatch.setattr(BamStore, "_process_chromosome", lambda *a, **kw: (0.0, np.zeros(a[3]), None))
    BamStore(tmp_path / "ds", chromsizes, sample_names).process_samples(["0", "0"])

    BamStore(tmp_path / "ds", chromsizes, sample_names, resume=True, overwrite=False)

    with pytest.raises(ValueError, match="mismatch"):
        BamStore(tmp_path / "ds", chromsizes, ["x", "y"], resume=True, overwrite=False)


# ---------------------------------------------------------------------------
# Open read-only vs writable
# ---------------------------------------------------------------------------


def test_open_readonly_and_writable(tmp_path, chromsizes, sample_names, monkeypatch):
    monkeypatch.setattr(BamStore, "_process_chromosome", lambda *a, **kw: (0.0, np.full(a[3], int(a[1])), None))
    BamStore(tmp_path / "ds", chromsizes, sample_names).process_samples(["1", "2"])

    ro = BamStore.open(tmp_path / "ds")
    assert ro.read_only is True
    assert ro.sample_names == sample_names
    with pytest.raises(RuntimeError):
        ro.remove_metadata_columns(["sample_hash"])

    rw = BamStore.open(tmp_path / "ds", read_only=False)
    assert rw.read_only is False
    rw.remove_metadata_columns(["sample_hash"])
    assert np.all(rw.meta["sample_hashes"][:] == 0)


# ---------------------------------------------------------------------------
# Metadata CRUD
# ---------------------------------------------------------------------------


def test_bamstore_metadata_partial_updates(tmp_path, chromsizes, sample_names):
    store = BamStore(tmp_path / "ds", chromsizes, sample_names)

    meta_df = pd.DataFrame({"sample_id": sample_names, "group": ["A", "B"], "assay": ["ATAC", "RNA"]})
    store.set_metadata(meta_df)
    md = store.get_metadata()
    assert md.loc["s1", "group"] == "A"
    assert md.loc["s2", "assay"] == "RNA"

    store.set_metadata(pd.DataFrame({"sample_id": ["s1"], "group": ["C"]}), merge=True)
    md2 = store.get_metadata()
    assert md2.loc["s1", "group"] == "C"
    assert md2.loc["s2", "group"] == "B"

    store.update_metadata({"new_col": {"s2": "high"}})
    md3 = store.get_metadata()
    assert md3.loc["s2", "new_col"] == "high"
    assert pd.isna(md3.loc["s1", "new_col"])


def test_bamstore_metadata_crud_helpers(tmp_path):
    store = BamStore(tmp_path / "ds", {"chr1": 4}, ["s1"])
    store.update_metadata({"col1": ["val1"], "col2": ["val2"]})

    assert {"col1", "col2"}.issubset(set(store.list_metadata_columns()))
    store.remove_metadata_columns(["col1"])
    assert "col1" not in store.list_metadata_columns()
    assert "col2" in store.list_metadata_columns()


def test_bamstore_metadata_json_roundtrip(tmp_path):
    store = BamStore(tmp_path / "ds", {"chr1": 10}, ["s1", "s2"])
    store.update_metadata({"info": ["x", "y"]})

    json_path = tmp_path / "meta.json"
    store.metadata_to_json(json_path)
    reloaded = BamStore.metadata_from_json(json_path)
    assert reloaded.set_index("sample_id").loc["s2", "info"] == "y"


# ---------------------------------------------------------------------------
# Sample hash tracking
# ---------------------------------------------------------------------------


def test_bamstore_sample_hash_alignment(tmp_path, monkeypatch):
    f1 = tmp_path / "1.bam"
    f1.write_bytes(b"content1" * 1000)
    f2 = tmp_path / "2.bam"
    f2.write_bytes(b"content2" * 1000)

    monkeypatch.setattr(BamStore, "_process_chromosome", lambda *a, **kw: (0.0, np.zeros(a[3]), None))
    store = BamStore(tmp_path / "ds", {"chr1": 10}, ["s1", "s2"])
    store.process_samples([str(f1), str(f2)])

    hashes = store.sample_hashes
    assert len(hashes) == 2
    assert hashes[0] != hashes[1]
    assert len(hashes[0]) == 32  # MD5 hex

    md = store.get_metadata()
    assert "sample_hash" in md.columns
    assert md.loc["s1", "sample_hash"] == hashes[0]

    meta_ok = pd.DataFrame({"sample_id": ["s1", "s2"], "sample_hash": hashes})
    store.set_metadata(meta_ok)

    with pytest.raises(ValueError, match="Sample hash mismatch"):
        store.set_metadata(pd.DataFrame({"sample_id": ["s1"], "sample_hash": ["wrong"]}))


def test_bamstore_hash_validation_on_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(BamStore, "_process_chromosome", lambda *a, **kw: (0.0, np.zeros(a[3]), None))
    BamStore(tmp_path / "ds", {"chr1": 4}, ["s1", "s2"])
    BamStore(tmp_path / "ds", {"chr1": 4}, ["s1", "s2"], resume=True, overwrite=False)

    with pytest.raises(ValueError, match="mismatch"):
        BamStore(tmp_path / "ds", {"chr1": 4}, ["s2", "s1"], resume=True, overwrite=False)


# ---------------------------------------------------------------------------
# Auto chromsizes
# ---------------------------------------------------------------------------


def test_bamstore_auto_chromsizes(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "quantnado.dataset.store_bam._get_chromsizes_from_bam",
        lambda path: {"chr1": 100, "chr2": 200},
    )
    monkeypatch.setattr(BamStore, "_process_chromosome", lambda *a, **kw: (0.0, np.zeros(a[3]), None))

    bam_path = tmp_path / "test.bam"
    bam_path.write_text("dummy")

    store = BamStore.from_bam_files(
        bam_files=[str(bam_path)],
        store_path=tmp_path / "auto_ds",
        chromsizes=None,
    )
    assert store.chromsizes == {"chr1": 100, "chr2": 200}
    assert store.sample_names == ["test"]


# ---------------------------------------------------------------------------
# to_xarray
# ---------------------------------------------------------------------------


def test_to_xarray_requires_all_complete(tmp_path, chromsizes, sample_names, monkeypatch):
    monkeypatch.setattr(BamStore, "_process_chromosome", lambda *a, **kw: (0.0, np.full(a[3], int(a[1])), None))
    store = BamStore(tmp_path / "ds", chromsizes, sample_names)
    store.process_samples(["1", "2"])
    store.meta["completed"][0] = False

    with pytest.raises(RuntimeError, match="incomplete"):
        store.to_xarray()


def test_to_xarray_structure_and_metadata(tmp_path, chromsizes, sample_names, monkeypatch):
    monkeypatch.setattr(BamStore, "_process_chromosome", lambda *a, **kw: (10.0, np.full(a[3], int(a[1]), dtype=np.uint16), None))
    store = BamStore(tmp_path / "ds", chromsizes, sample_names)
    store.process_samples(["1", "2"])
    store.set_metadata(pd.DataFrame({
        "sample_id": sample_names,
        "cell_type": ["A549", "HeLa"],
        "treatment": ["control", "treated"],
    }))

    xr_dict = store.to_xarray()
    assert set(xr_dict.keys()) == set(chromsizes.keys())

    xr_subset = store.to_xarray(chromosomes=["chr1"])
    assert set(xr_subset.keys()) == {"chr1"}

    for chrom, da_xr in xr_dict.items():
        assert da_xr.dims == ("sample", "position")
        assert list(da_xr.coords["sample"].values) == sample_names
        assert da_xr.shape == (len(sample_names), chromsizes[chrom])
        assert "sample_hashes" in da_xr.attrs
        assert "cell_type" in da_xr.coords
        assert "treatment" in da_xr.coords

    with pytest.raises(ValueError, match="not in store"):
        store.to_xarray(chromosomes=["chrInvalid"])


# ---------------------------------------------------------------------------
# extract_region
# ---------------------------------------------------------------------------


def test_extract_region_string_format(tmp_path, chromsizes, sample_names, monkeypatch):
    monkeypatch.setattr(BamStore, "_process_chromosome", lambda *a, **kw: (10.0, np.full(a[3], int(a[1]), dtype=np.uint16), None))
    store = BamStore(tmp_path / "ds", chromsizes, sample_names)
    store.process_samples(["1", "2"])
    store.set_metadata(pd.DataFrame({"sample_id": sample_names, "cell_type": ["A549", "HeLa"]}))

    region = store.extract_region("chr1:1-3")
    assert region.shape == (2, 2)
    assert list(region.coords["position"].values) == [1, 2]
    assert region.attrs["chromosome"] == "chr1"

    whole = store.extract_region("chr1")
    assert whole.shape == (2, chromsizes["chr1"])
    assert whole.attrs["start"] == 0
    assert whole.attrs["end"] == chromsizes["chr1"]


def test_extract_region_separate_parameters(tmp_path, chromsizes, sample_names, monkeypatch):
    monkeypatch.setattr(BamStore, "_process_chromosome", lambda *a, **kw: (10.0, np.full(a[3], int(a[1]), dtype=np.uint16), None))
    store = BamStore(tmp_path / "ds", chromsizes, sample_names)
    store.process_samples(["1", "2"])

    region = store.extract_region(chrom="chr1", start=1, end=3)
    assert region.shape == (2, 2)

    default = store.extract_region(chrom="chr2")
    assert default.shape == (2, chromsizes["chr2"])


def test_extract_region_sample_subsetting(tmp_path, chromsizes, sample_names, monkeypatch):
    monkeypatch.setattr(BamStore, "_process_chromosome", lambda *a, **kw: (10.0, np.full(a[3], int(a[1]), dtype=np.uint16), None))
    store = BamStore(tmp_path / "ds", chromsizes, sample_names)
    store.process_samples(["1", "2"])

    s1_only = store.extract_region("chr1:0-2", samples=["s1"])
    assert s1_only.shape == (1, 2)
    assert list(s1_only.coords["sample"].values) == ["s1"]
    assert np.all(s1_only.values == 1)

    by_idx = store.extract_region("chr1:0-2", samples=[1])
    assert list(by_idx.coords["sample"].values) == ["s2"]


def test_extract_region_as_numpy(tmp_path, chromsizes, sample_names, monkeypatch):
    monkeypatch.setattr(BamStore, "_process_chromosome", lambda *a, **kw: (10.0, np.full(a[3], int(a[1]), dtype=np.uint16), None))
    store = BamStore(tmp_path / "ds", chromsizes, sample_names)
    store.process_samples(["1", "2"])

    np_result = store.extract_region("chr1:1-3", as_xarray=False)
    xr_result = store.extract_region("chr1:1-3", as_xarray=True)
    assert isinstance(np_result, np.ndarray)
    np.testing.assert_array_equal(np_result, xr_result.values)


def test_extract_region_normalise_cpm(tmp_path, chromsizes, sample_names, monkeypatch):
    monkeypatch.setattr(BamStore, "_process_chromosome", lambda *a, **kw: (10.0, np.full(a[3], int(a[1]), dtype=np.uint16), None))
    store = BamStore(tmp_path / "ds", chromsizes, sample_names)
    store.process_samples(["1", "2"])

    region = store.extract_region(
        "chr1:0-2",
        normalise="cpm",
        library_sizes={"s1": 1_000_000, "s2": 2_000_000},
    )
    np.testing.assert_allclose(region.values, np.array([[1.0, 1.0], [1.0, 1.0]]))
    assert region.attrs["normalised"] == "cpm"


def test_extract_region_metadata_coordinates(tmp_path, chromsizes, sample_names, monkeypatch):
    monkeypatch.setattr(BamStore, "_process_chromosome", lambda *a, **kw: (10.0, np.full(a[3], int(a[1]), dtype=np.uint16), None))
    store = BamStore(tmp_path / "ds", chromsizes, sample_names)
    store.process_samples(["1", "2"])
    store.set_metadata(pd.DataFrame({
        "sample_id": sample_names,
        "cell_type": ["A549", "HeLa"],
        "treatment": ["control", "treated"],
    }))

    region = store.extract_region("chr1:0-2")
    assert "cell_type" in region.coords
    assert list(region.coords["cell_type"].values) == ["A549", "HeLa"]

    subset = store.extract_region("chr1:0-2", samples=["s2"])
    assert list(subset.coords["cell_type"].values) == ["HeLa"]


def test_extract_region_error_cases(tmp_path, chromsizes, sample_names, monkeypatch):
    monkeypatch.setattr(BamStore, "_process_chromosome", lambda *a, **kw: (10.0, np.full(a[3], int(a[1]), dtype=np.uint16), None))
    store = BamStore(tmp_path / "ds", chromsizes, sample_names)
    store.process_samples(["1", "2"])

    with pytest.raises(ValueError, match="not in store"):
        store.extract_region("chrInvalid:0-10")

    with pytest.raises(ValueError, match="either 'region' or 'chrom'"):
        store.extract_region(region="chr1:0-2", chrom="chr1")

    with pytest.raises(ValueError, match="Must specify"):
        store.extract_region()

    with pytest.raises(ValueError, match="must be greater than start"):
        store.extract_region("chr1:10-5")

    with pytest.raises(ValueError, match="exceeds chromosome size"):
        store.extract_region(chrom="chr1", start=0, end=1000)

    with pytest.raises(ValueError, match="not found"):
        store.extract_region("chr1:0-2", samples=["invalid"])

    with pytest.raises(ValueError, match="out of range"):
        store.extract_region("chr1:0-2", samples=[999])

    store.meta["completed"][0] = False
    with pytest.raises(RuntimeError, match="incomplete"):
        store.extract_region("chr1:0-2")


# ---------------------------------------------------------------------------
# Chunk length auto-selection and overrides
# ---------------------------------------------------------------------------


def test_bamstore_auto_chunk_len_uses_filesystem_hint(tmp_path, monkeypatch):
    chromsizes = {"chr1": 250_000_000}

    monkeypatch.setattr("quantnado.dataset.store_bam.is_network_fs", lambda path: False)
    local_store = BamStore(tmp_path / "local_ds", chromsizes, ["s1"])
    expected_local = estimate_chunk_len(
        contig_lengths=chromsizes,
        dtype_bytes=np.dtype(np.uint32).itemsize,
        fs_is_network=False,
    )["chunk_len"]
    assert local_store.chunk_len == expected_local
    assert local_store.root.attrs["chunk_len"] == expected_local

    monkeypatch.setattr("quantnado.dataset.store_bam.is_network_fs", lambda path: True)
    network_store = BamStore(tmp_path / "network_ds", chromsizes, ["s1"])
    expected_network = estimate_chunk_len(
        contig_lengths=chromsizes,
        dtype_bytes=np.dtype(np.uint32).itemsize,
        fs_is_network=True,
    )["chunk_len"]
    assert network_store.chunk_len == expected_network
    assert network_store.root.attrs["chunk_len"] == expected_network
    assert network_store.chunk_len > local_store.chunk_len


def test_bamstore_explicit_chunk_len_overrides_auto(tmp_path, monkeypatch):
    monkeypatch.setattr("quantnado.dataset.store_bam.is_network_fs", lambda path: True)

    store = BamStore(
        tmp_path / "manual_chunk_ds",
        {"chr1": 250_000_000},
        ["s1"],
        chunk_len=131072,
    )

    assert store.chunk_len == 131072
    assert store.root.attrs["chunk_len"] == 131072


# ---------------------------------------------------------------------------
# Construction compression profiles
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("profile", "expected_compressors"),
    [("default", 1), ("fast", 1), ("none", 0)],
)
def test_bamstore_construction_compression_profiles(
    tmp_path, chromsizes, sample_names, profile, expected_compressors
):
    store = BamStore(
        tmp_path / f"{profile}_compression_ds",
        chromsizes,
        sample_names,
        construction_compression=profile,
    )

    array = store.root["coverage"]["chr1"]
    assert store.root.attrs["construction_compression"] == profile
    assert len(array.compressors) == expected_compressors


def test_bamstore_invalid_construction_compression_raises(tmp_path, chromsizes, sample_names):
    with pytest.raises(ValueError, match="construction_compression"):
        BamStore(
            tmp_path / "bad_compression_ds",
            chromsizes,
            sample_names,
            construction_compression="turbo",
        )


# ---------------------------------------------------------------------------
# Streaming write with combined workers
# ---------------------------------------------------------------------------


def test_process_samples_streaming_writes_correct_data(
    tmp_path, chromsizes, sample_names, monkeypatch
):
    """Verify that process_samples with max_workers > 1 (combined into
    effective chr_workers) produces the same results as sequential processing."""

    def fake_chrom(self, bam_file, contig, contig_size, is_stranded, use_fragment=False, read_filter=None):
        return 0.0, np.full(contig_size, int(bam_file), dtype=np.uint16), None

    monkeypatch.setattr(BamStore, "_process_chromosome", fake_chrom)

    store = BamStore(tmp_path / "streaming_ds", chromsizes, sample_names)
    store.process_samples(["1", "2"], max_workers=2)

    assert np.all(store.root["coverage"]["chr1"][:, 0] == 1)
    assert np.all(store.root["coverage"]["chr2"][:, 1] == 2)
    assert store.completed_mask.tolist() == [True, True]
    assert np.isfinite(store.meta["sparsity"][:]).all()


# ---------------------------------------------------------------------------
# Local staging: publish to final path
# ---------------------------------------------------------------------------


def test_bamstore_from_bam_files_with_local_staging_publishes_to_final_path(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "quantnado.dataset.store_bam._get_chromsizes_from_bam",
        lambda path: {"chr1": 5},
    )
    monkeypatch.setattr(
        BamStore,
        "_process_chromosome",
        lambda *args, **kwargs: (0.0, np.ones(args[3], dtype=np.uint16), None),
    )

    bam_path = tmp_path / "sample1.bam"
    bam_path.write_text("dummy")
    final_store = tmp_path / "published_ds.zarr"
    scratch_dir = tmp_path / "scratch"

    store = BamStore.from_bam_files(
        bam_files=[str(bam_path)],
        store_path=final_store,
        chromsizes=None,
        local_staging=True,
        staging_dir=scratch_dir,
    )

    assert store.store_path == final_store
    assert final_store.exists()
    assert np.all(store.root["coverage"]["chr1"][:, 0] == 1)
    assert list(scratch_dir.iterdir()) == []


def test_bamstore_staging_rejects_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "quantnado.dataset.store_bam._get_chromsizes_from_bam",
        lambda path: {"chr1": 5},
    )

    bam_path = tmp_path / "sample1.bam"
    bam_path.write_text("dummy")

    with pytest.raises(ValueError, match="resume=True is not supported"):
        BamStore.from_bam_files(
            bam_files=[str(bam_path)],
            store_path=tmp_path / "published_ds.zarr",
            chromsizes=None,
            resume=True,
            local_staging=True,
        )


# ---------------------------------------------------------------------------
# _combine_metadata_files edge cases
# ---------------------------------------------------------------------------


class TestCombineMetadataFiles:
    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="No metadata files"):
            BamStore._combine_metadata_files([])

    def test_no_valid_files_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No valid metadata files"):
            BamStore._combine_metadata_files([str(tmp_path / "nonexistent.csv")])

    def test_single_file(self, tmp_path):
        f = tmp_path / "meta.csv"
        f.write_text("sample_id,group\ns1,A\ns2,B\n")
        result = BamStore._combine_metadata_files([str(f)])
        assert len(result) == 2
        assert set(result["sample_id"]) == {"s1", "s2"}

    def test_multiple_files_merged(self, tmp_path):
        f1 = tmp_path / "a.csv"
        f2 = tmp_path / "b.csv"
        f1.write_text("sample_id,condition\ns1,ctrl\n")
        f2.write_text("sample_id,condition\ns2,treat\n")
        result = BamStore._combine_metadata_files([str(f1), str(f2)])
        assert len(result) == 2
        assert set(result["sample_id"]) == {"s1", "s2"}

    def test_skips_missing_files(self, tmp_path):
        f1 = tmp_path / "real.csv"
        f1.write_text("sample_id,group\ns1,A\n")
        result = BamStore._combine_metadata_files([str(f1), str(tmp_path / "missing.csv")])
        assert len(result) == 1

    def test_r1_r2_columns_excluded(self, tmp_path):
        f = tmp_path / "meta.csv"
        f.write_text("sample_id,r1_path,r2_path,condition\ns1,/p/r1.fq,/p/r2.fq,ctrl\n")
        result = BamStore._combine_metadata_files([str(f)])
        assert "r1_path" not in result.columns
        assert "r2_path" not in result.columns
        assert "condition" in result.columns


# ---------------------------------------------------------------------------
# set_metadata with merge=False
# ---------------------------------------------------------------------------


class TestSetMetadataMergeFalse:
    def test_merge_false_replaces_existing_columns(self, tmp_path):
        store = BamStore(tmp_path / "ds", {"chr1": 4}, ["s1", "s2"])
        store.set_metadata(pd.DataFrame({
            "sample_id": ["s1", "s2"],
            "group": ["A", "B"],
        }))
        # Now replace with merge=False - should clear existing and write new
        store.set_metadata(pd.DataFrame({
            "sample_id": ["s1", "s2"],
            "new_col": ["X", "Y"],
        }), merge=False)
        md = store.get_metadata()
        # new_col should exist
        assert "new_col" in md.columns
        # group column should be removed since merge=False clears it first
        assert "group" not in md.columns

    def test_sample_column_not_found_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Sample column"):
            BamStore(tmp_path / "ds", {"chr1": 4}, ["s1"]).set_metadata(pd.DataFrame({"wrong_col": ["s1"]}), sample_column="sample_id")


# ---------------------------------------------------------------------------
# update_metadata various paths
# ---------------------------------------------------------------------------


class TestUpdateMetadata:
    def test_update_with_list(self, tmp_path):
        store = BamStore(tmp_path / "ds", {"chr1": 4}, ["s1", "s2"])
        store.update_metadata({"condition": ["ctrl", "treat"]})
        md = store.get_metadata()
        assert md.loc["s1", "condition"] == "ctrl"
        assert md.loc["s2", "condition"] == "treat"

    def test_update_with_dict_partial(self, tmp_path):
        store = BamStore(tmp_path / "ds", {"chr1": 4}, ["s1", "s2"])
        store.update_metadata({"condition": ["ctrl", "ctrl"]})
        # Now update only s2
        store.update_metadata({"condition": {"s2": "treat"}})
        md = store.get_metadata()
        assert md.loc["s1", "condition"] == "ctrl"
        assert md.loc["s2", "condition"] == "treat"

    def test_update_with_dict_new_column(self, tmp_path):
        store = BamStore(tmp_path / "ds", {"chr1": 4}, ["s1", "s2"])
        store.update_metadata({"newcol": {"s1": "yes"}})
        md = store.get_metadata()
        assert md.loc["s1", "newcol"] == "yes"
        # s2 should have empty string
        assert md.loc["s2", "newcol"] == "" or pd.isna(md.loc["s2", "newcol"])

    def test_update_list_wrong_length_raises(self, tmp_path):
        store = BamStore(tmp_path / "ds", {"chr1": 4}, ["s1", "s2"])
        with pytest.raises(ValueError, match="items but store has"):
            store.update_metadata({"col": ["only_one"]})

    def test_update_with_invalid_type_raises(self, tmp_path):
        store = BamStore(tmp_path / "ds", {"chr1": 4}, ["s1", "s2"])
        with pytest.raises(TypeError, match="must be list or dict"):
            store.update_metadata({"col": 42})


# ---------------------------------------------------------------------------
# BamStore error paths
# ---------------------------------------------------------------------------


class TestBamStoreErrorPaths:
    def test_overwrite_in_read_only_mode_raises(self, tmp_path):
        BamStore(tmp_path / "ds", {"chr1": 4}, ["s1"])
        with pytest.raises(ValueError, match="read-only"):
            BamStore(
                tmp_path / "ds",
                {"chr1": 4},
                ["s1"],
                overwrite=True,
                read_only=True,
            )

    def test_empty_sample_names_raises(self, tmp_path):
        with pytest.raises(ValueError):
            BamStore(tmp_path / "empty", {"chr1": 4}, [])

    def test_file_exists_without_overwrite_or_resume_raises(self, tmp_path):
        BamStore(tmp_path / "ds", {"chr1": 4}, ["s1"])
        with pytest.raises(FileExistsError):
            BamStore(tmp_path / "ds", {"chr1": 4}, ["s1"], overwrite=False, resume=False)

    def test_open_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            BamStore.open(tmp_path / "nonexistent.zarr")

    def test_process_samples_wrong_length_raises(self, tmp_path):
        store = BamStore(tmp_path / "ds", {"chr1": 4}, ["s1", "s2"])
        with pytest.raises(ValueError, match="length must match"):
            store.process_samples(["only_one"])

    def test_process_samples_no_bam_files_raises(self, tmp_path):
        store = BamStore(tmp_path / "ds", {"chr1": 4}, ["s1", "s2"])
        with pytest.raises(ValueError, match="bam_files must be provided"):
            store.process_samples()


# ---------------------------------------------------------------------------
# coverage_type_map property
# ---------------------------------------------------------------------------


class TestCoverageTypeMap:
    def test_single_value_applies_to_all_samples(self, tmp_path, chromsizes, sample_names):
        store = BamStore(tmp_path / "ds", chromsizes, sample_names, coverage_type=CoverageType.STRANDED)
        assert store.coverage_type_map == {"s1": CoverageType.STRANDED, "s2": CoverageType.STRANDED}

    def test_default_is_unstranded(self, tmp_path, chromsizes, sample_names):
        store = BamStore(tmp_path / "ds", chromsizes, sample_names)
        assert store.coverage_type_map == {"s1": CoverageType.UNSTRANDED, "s2": CoverageType.UNSTRANDED}

    def test_list_maps_in_order(self, tmp_path, chromsizes, sample_names):
        store = BamStore(
            tmp_path / "ds", chromsizes, sample_names,
            coverage_type=[CoverageType.STRANDED, CoverageType.UNSTRANDED],
        )
        assert store.coverage_type_map == {"s1": CoverageType.STRANDED, "s2": CoverageType.UNSTRANDED}

    def test_dict_partial_defaults_to_unstranded(self, tmp_path, chromsizes, sample_names):
        store = BamStore(
            tmp_path / "ds", chromsizes, sample_names,
            coverage_type={"s1": CoverageType.STRANDED},
        )
        assert store.coverage_type_map["s1"] == CoverageType.STRANDED
        assert store.coverage_type_map["s2"] == CoverageType.UNSTRANDED

    def test_dict_unknown_sample_raises(self, tmp_path, chromsizes, sample_names):
        with pytest.raises(ValueError, match="Unknown sample names"):
            BamStore(
                tmp_path / "ds", chromsizes, sample_names,
                coverage_type={"s1": CoverageType.STRANDED, "bad_sample": CoverageType.UNSTRANDED},
            )

    def test_list_wrong_length_raises(self, tmp_path, chromsizes, sample_names):
        with pytest.raises(ValueError, match="same length"):
            store = BamStore(
                tmp_path / "ds", chromsizes, sample_names,
                coverage_type=[CoverageType.STRANDED],
            )
            _ = store.coverage_type_map


# ---------------------------------------------------------------------------
# bam_filter_map property
# ---------------------------------------------------------------------------


class TestBamFilterMap:
    def test_none_gives_default_filter_for_all(self, tmp_path, chromsizes, sample_names):
        store = BamStore(tmp_path / "ds", chromsizes, sample_names, bam_filter=None)
        fmap = store.bam_filter_map
        assert set(fmap.keys()) == set(sample_names)
        assert all(isinstance(f, bamnado.ReadFilter) for f in fmap.values())

    def test_single_filter_applies_to_all(self, tmp_path, chromsizes, sample_names):
        rf = bamnado.ReadFilter(min_mapq=20)
        store = BamStore(tmp_path / "ds", chromsizes, sample_names, bam_filter=rf)
        fmap = store.bam_filter_map
        assert all(f.min_mapq == 20 for f in fmap.values())

    def test_list_maps_in_order(self, tmp_path, chromsizes, sample_names):
        rf1 = bamnado.ReadFilter(min_mapq=10)
        rf2 = bamnado.ReadFilter(min_mapq=30)
        store = BamStore(tmp_path / "ds", chromsizes, sample_names, bam_filter=[rf1, rf2])
        fmap = store.bam_filter_map
        assert fmap["s1"].min_mapq == 10
        assert fmap["s2"].min_mapq == 30

    def test_dict_partial_defaults_to_empty_filter(self, tmp_path, chromsizes, sample_names):
        rf = bamnado.ReadFilter(min_mapq=25)
        store = BamStore(tmp_path / "ds", chromsizes, sample_names, bam_filter={"s1": rf})
        fmap = store.bam_filter_map
        assert fmap["s1"].min_mapq == 25
        assert fmap["s2"].min_mapq == 0

    def test_dict_unknown_sample_raises(self, tmp_path, chromsizes, sample_names):
        rf = bamnado.ReadFilter()
        store = BamStore(tmp_path / "ds", chromsizes, sample_names, bam_filter={"s_unknown": rf})
        with pytest.raises(ValueError, match="Unknown sample names"):
            _ = store.bam_filter_map


# ---------------------------------------------------------------------------
# CoverageType.STRANDED array creation and open() roundtrip
# ---------------------------------------------------------------------------


def test_stranded_store_creates_fwd_rev_arrays(tmp_path, chromsizes, sample_names, monkeypatch):
    monkeypatch.setattr(
        BamStore, "_process_chromosome",
        lambda *a, **kw: (0.0, np.ones(a[3], dtype=np.uint32), np.full(a[3], 2, dtype=np.uint32))
        if a[4] else (0.0, np.zeros(a[3]), None),
    )
    store = BamStore(tmp_path / "ds", chromsizes, sample_names, coverage_type=CoverageType.STRANDED)
    store.process_samples(["0", "0"])

    assert "coverage_fwd" in store.root, "coverage_fwd group not found"
    assert "coverage_rev" in store.root, "coverage_rev group not found"
    for chrom in chromsizes:
        assert chrom in store.root["coverage_fwd"], f"coverage_fwd/{chrom} not found"
        assert chrom in store.root["coverage_rev"], f"coverage_rev/{chrom} not found"
    assert np.all(store.root["coverage_fwd"]["chr1"][:, 0] == 1)
    assert np.all(store.root["coverage_rev"]["chr1"][:, 0] == 2)


def test_open_roundtrips_stranded_coverage_type(tmp_path, chromsizes, sample_names, monkeypatch):
    monkeypatch.setattr(
        BamStore, "_process_chromosome",
        lambda *a, **kw: (0.0, np.zeros(a[3]), np.zeros(a[3])) if a[4] else (0.0, np.zeros(a[3]), None),
    )
    BamStore(tmp_path / "ds", chromsizes, sample_names, coverage_type=CoverageType.STRANDED).process_samples(["0", "0"])

    reopened = BamStore.open(tmp_path / "ds")
    assert all(bt == CoverageType.STRANDED for bt in reopened.coverage_type_map.values())


def test_open_roundtrips_unstranded_coverage_type(tmp_path, chromsizes, sample_names):
    BamStore(tmp_path / "ds", chromsizes, sample_names)

    reopened = BamStore.open(tmp_path / "ds")
    assert all(bt == CoverageType.UNSTRANDED for bt in reopened.coverage_type_map.values())


# ---------------------------------------------------------------------------
# MCC: from_bam_files expansion and sample_bam_map / viewpoint_map
# ---------------------------------------------------------------------------


class TestMCCUnit:
    def _mock_env(self, monkeypatch, viewpoints, chromsizes=None):
        """Apply the three monkeypatches needed for a unit-level MCC test."""
        monkeypatch.setattr(
            "quantnado.dataset.store_bam._get_viewpoints_from_mcc_bam",
            lambda path: viewpoints,
        )
        monkeypatch.setattr(
            "quantnado.dataset.store_bam._get_chromsizes_from_bam",
            lambda path: chromsizes or {"chr1": 10},
        )
        monkeypatch.setattr(
            "quantnado.dataset.store_bam._process_chromosome_mcc",
            lambda bam_file, contig, contig_size, viewpoints, viewpoint_tag, base_filter, use_fragment=False: {
                vp: np.zeros(contig_size, dtype=np.uint32) for vp in viewpoints
            },
        )

    def test_sample_names_expanded_per_viewpoint(self, tmp_path, monkeypatch):
        self._mock_env(monkeypatch, ["VP_A", "VP_B"])
        bam = tmp_path / "sample1.bam"
        bam.write_text("dummy")

        store = BamStore.from_bam_files(
            bam_files=[str(bam)],
            store_path=tmp_path / "store",
            coverage_type=CoverageType.MICRO_CAPTURE_C,
        )

        assert store.sample_names == ["sample1_VP_A", "sample1_VP_B"]

    def test_viewpoint_map_populated(self, tmp_path, monkeypatch):
        self._mock_env(monkeypatch, ["VP_A", "VP_B"])
        bam = tmp_path / "sample1.bam"
        bam.write_text("dummy")

        store = BamStore.from_bam_files(
            bam_files=[str(bam)],
            store_path=tmp_path / "store",
            coverage_type=CoverageType.MICRO_CAPTURE_C,
        )

        assert store.viewpoint_map == {"sample1_VP_A": "VP_A", "sample1_VP_B": "VP_B"}

    def test_sample_bam_map_all_point_to_source(self, tmp_path, monkeypatch):
        self._mock_env(monkeypatch, ["VP_A", "VP_B"])
        bam = tmp_path / "sample1.bam"
        bam.write_text("dummy")

        store = BamStore.from_bam_files(
            bam_files=[str(bam)],
            store_path=tmp_path / "store",
            coverage_type=CoverageType.MICRO_CAPTURE_C,
        )

        assert set(store.sample_bam_map.values()) == {str(bam)}

    def test_all_virtual_samples_completed(self, tmp_path, monkeypatch):
        self._mock_env(monkeypatch, ["VP_A", "VP_B", "VP_C"])
        bam = tmp_path / "s.bam"
        bam.write_text("dummy")

        store = BamStore.from_bam_files(
            bam_files=[str(bam)],
            store_path=tmp_path / "store",
            coverage_type=CoverageType.MICRO_CAPTURE_C,
        )

        assert store.completed_mask.all()
        assert store.n_samples == 3

    def test_mixed_mcc_and_regular_bam(self, tmp_path, monkeypatch):
        """Non-MCC BAM files are not expanded; only MCC ones get viewpoint suffixes."""
        monkeypatch.setattr(
            "quantnado.dataset.store_bam._get_viewpoints_from_mcc_bam",
            lambda path: ["VP_A", "VP_B"],
        )
        monkeypatch.setattr(BamStore, "_process_chromosome", lambda *a, **kw: (0.0, np.zeros(a[3]), None))
        monkeypatch.setattr(
            "quantnado.dataset.store_bam._process_chromosome_mcc",
            lambda bam_file, contig, contig_size, viewpoints, viewpoint_tag, base_filter, use_fragment=False: {
                vp: np.zeros(contig_size, dtype=np.uint32) for vp in viewpoints
            },
        )

        bam_mcc = tmp_path / "mcc.bam"
        bam_reg = tmp_path / "reg.bam"
        bam_mcc.write_text("dummy")
        bam_reg.write_text("dummy")

        store = BamStore.from_bam_files(
            bam_files=[str(bam_mcc), str(bam_reg)],
            store_path=tmp_path / "store",
            coverage_type=[CoverageType.MICRO_CAPTURE_C, CoverageType.UNSTRANDED],
            chromsizes={"chr1": 10},
        )

        assert "mcc_VP_A" in store.sample_names
        assert "mcc_VP_B" in store.sample_names
        assert "reg" in store.sample_names
        assert store.n_samples == 3

    def test_process_samples_uses_sample_bam_map(self, tmp_path, monkeypatch):
        """process_samples() with no bam_files arg routes via sample_bam_map."""
        calls = []

        def fake_chrom_mcc(bam_file, contig, contig_size, viewpoints, viewpoint_tag, base_filter, use_fragment=False):
            calls.append((bam_file, contig, set(viewpoints)))
            return {vp: np.zeros(contig_size, dtype=np.uint32) for vp in viewpoints}

        monkeypatch.setattr("quantnado.dataset.store_bam._process_chromosome_mcc", fake_chrom_mcc)

        store = BamStore(
            tmp_path / "ds",
            {"chr1": 5},
            ["s_VP_A", "s_VP_B"],
            coverage_type=CoverageType.MICRO_CAPTURE_C,
            sample_bam_map={"s_VP_A": "source.bam", "s_VP_B": "source.bam"},
            viewpoint_map={"s_VP_A": "VP_A", "s_VP_B": "VP_B"},
        )
        store.process_samples()

        assert store.completed_mask.all()
        # One call per chromosome per BAM file, with all viewpoints together
        assert len(calls) == 1
        assert calls[0] == ("source.bam", "chr1", {"VP_A", "VP_B"})


# ---------------------------------------------------------------------------
# VP tag filter injection for MCC samples
# ---------------------------------------------------------------------------


class TestVPTagFilterInjection:
    def test_vp_filter_set_on_mcc_sample(self, tmp_path, monkeypatch):
        """_process_chromosome receives filter_tag/filter_tag_value for MCC samples."""
        captured: dict[str, bamnado.ReadFilter] = {}

        def fake_chrom(self, bam_file, contig, contig_size, is_stranded, use_fragment=False, read_filter=None):
            if read_filter and read_filter.filter_tag_value:
                captured[read_filter.filter_tag_value] = read_filter
            return 0.0, np.zeros(contig_size), None

        monkeypatch.setattr(BamStore, "_process_chromosome", fake_chrom)

        store = BamStore(
            tmp_path / "ds",
            {"chr1": 5},
            ["s_GENE1", "s_GENE2"],
            coverage_type=CoverageType.MICRO_CAPTURE_C,
            sample_bam_map={"s_GENE1": "x.bam", "s_GENE2": "x.bam"},
            viewpoint_map={"s_GENE1": "GENE1", "s_GENE2": "GENE2"},
        )
        store.process_samples()

        assert "GENE1" in captured, "filter_tag_value=GENE1 not set on read filter"
        assert "GENE2" in captured, "filter_tag_value=GENE2 not set on read filter"
        assert captured["GENE1"].filter_tag == "VP"
        assert captured["GENE2"].filter_tag == "VP"

    def test_non_mcc_sample_has_no_vp_filter(self, tmp_path, chromsizes, sample_names, monkeypatch):
        """Unstranded samples must not have filter_tag set."""
        captured_tags = []

        def fake_chrom(self, bam_file, contig, contig_size, is_stranded, use_fragment=False, read_filter=None):
            captured_tags.append(getattr(read_filter, "filter_tag", None))
            return 0.0, np.zeros(contig_size), None

        monkeypatch.setattr(BamStore, "_process_chromosome", fake_chrom)
        store = BamStore(tmp_path / "ds", chromsizes, sample_names)
        store.process_samples(["0", "0"])

        assert all(t is None for t in captured_tags), f"Unexpected VP filter on unstranded sample: {captured_tags}"

    def test_custom_viewpoint_tag_respected(self, tmp_path, monkeypatch):
        """viewpoint_tag parameter controls which SAM tag is used for filtering."""
        captured: dict[str, bamnado.ReadFilter] = {}

        def fake_chrom(self, bam_file, contig, contig_size, is_stranded, use_fragment=False, read_filter=None):
            if read_filter and read_filter.filter_tag_value:
                captured[read_filter.filter_tag_value] = read_filter
            return 0.0, np.zeros(contig_size), None

        monkeypatch.setattr(BamStore, "_process_chromosome", fake_chrom)

        store = BamStore(
            tmp_path / "ds",
            {"chr1": 5},
            ["s_LOCUS1"],
            coverage_type=CoverageType.MICRO_CAPTURE_C,
            sample_bam_map={"s_LOCUS1": "x.bam"},
            viewpoint_map={"s_LOCUS1": "LOCUS1"},
            viewpoint_tag="VT",
        )
        store.process_samples()

        assert captured["LOCUS1"].filter_tag == "VT"

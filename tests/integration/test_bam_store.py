"""Integration tests for BamStore: create, open, process, metadata, resume."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantnado.dataset.bam import BamStore
from quantnado.dataset.core import QuantNadoDataset


# ---------------------------------------------------------------------------
# Write and basic read
# ---------------------------------------------------------------------------


def test_bamstore_write_and_metadata(tmp_path, chromsizes, sample_names, monkeypatch):
    def fake_chrom(self, bam_file, contig, size):
        return contig, np.full(size, int(bam_file), dtype=np.uint16), 0.0

    monkeypatch.setattr(BamStore, "_process_chromosome", fake_chrom)
    store = BamStore(tmp_path / "ds", chromsizes, sample_names)
    store.process_samples(["1", "2"])

    assert np.all(store.root["chr1"][0, :] == 1)
    assert np.all(store.root["chr2"][1, :] == 2)
    stored = [s.decode() if isinstance(s, (bytes, bytearray)) else s for s in store.root.attrs["sample_names"]]
    assert stored == sample_names
    assert store.completed_mask.tolist() == [True, True]
    assert np.isfinite(store.meta["sparsity"][:]).all()


def test_bamstore_dataset_wrapper(tmp_path, chromsizes, sample_names, monkeypatch):
    def fake_chrom(self, bam_file, contig, size):
        return contig, np.full(size, int(bam_file), dtype=np.uint16), 0.0

    monkeypatch.setattr(BamStore, "_process_chromosome", fake_chrom)
    store = BamStore(tmp_path / "ds", chromsizes, sample_names)
    store.process_samples(["1", "2"])

    ds = QuantNadoDataset(store.store_path)
    assert ds.sample_names == sample_names
    assert ds.completed_mask.tolist() == [True, True]
    assert ds.chromsizes == chromsizes
    np.testing.assert_array_equal(ds.get_chrom("chr1")[0, :], np.array([1, 1, 1, 1], dtype=np.uint32))


# ---------------------------------------------------------------------------
# Resume / overwrite validation
# ---------------------------------------------------------------------------


def test_resume_validates_sample_names(tmp_path, chromsizes, sample_names, monkeypatch):
    monkeypatch.setattr(BamStore, "_process_chromosome", lambda *a, **kw: (a[2], np.zeros(a[3]), 0.0))
    BamStore(tmp_path / "ds", chromsizes, sample_names).process_samples(["0", "0"])

    BamStore(tmp_path / "ds", chromsizes, sample_names, resume=True, overwrite=False)

    with pytest.raises(ValueError, match="names do not match"):
        BamStore(tmp_path / "ds", chromsizes, ["x", "y"], resume=True, overwrite=False)


# ---------------------------------------------------------------------------
# Open read-only vs writable
# ---------------------------------------------------------------------------


def test_open_readonly_and_writable(tmp_path, chromsizes, sample_names, monkeypatch):
    monkeypatch.setattr(BamStore, "_process_chromosome", lambda *a, **kw: (a[2], np.full(a[3], int(a[1])), 0.0))
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

    assert set(store.list_metadata_columns()) == {"col1", "col2"}
    store.remove_metadata_columns(["col1"])
    assert store.list_metadata_columns() == ["col2"]


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

    monkeypatch.setattr(BamStore, "_process_chromosome", lambda *a, **kw: (a[2], np.zeros(a[3]), 0.0))
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
    monkeypatch.setattr(BamStore, "_process_chromosome", lambda *a, **kw: (a[2], np.zeros(a[3]), 0.0))
    BamStore(tmp_path / "ds", {"chr1": 4}, ["s1", "s2"])
    BamStore(tmp_path / "ds", {"chr1": 4}, ["s1", "s2"], resume=True, overwrite=False)

    with pytest.raises(ValueError, match="names do not match"):
        BamStore(tmp_path / "ds", {"chr1": 4}, ["s2", "s1"], resume=True, overwrite=False)


# ---------------------------------------------------------------------------
# Auto chromsizes
# ---------------------------------------------------------------------------


def test_bamstore_auto_chromsizes(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "quantnado.dataset.bam._get_chromsizes_from_bam",
        lambda path: {"chr1": 100, "chr2": 200},
    )
    monkeypatch.setattr(BamStore, "_process_chromosome", lambda *a, **kw: (a[2], np.zeros(a[3]), 0.0))

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
    monkeypatch.setattr(BamStore, "_process_chromosome", lambda *a, **kw: (a[2], np.full(a[3], int(a[1])), 0.0))
    store = BamStore(tmp_path / "ds", chromsizes, sample_names)
    store.process_samples(["1", "2"])
    store.meta["completed"][0] = False

    with pytest.raises(RuntimeError, match="incomplete"):
        store.to_xarray()


def test_to_xarray_structure_and_metadata(tmp_path, chromsizes, sample_names, monkeypatch):
    monkeypatch.setattr(BamStore, "_process_chromosome", lambda *a, **kw: (a[2], np.full(a[3], int(a[1]), dtype=np.uint16), 10.0))
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
    monkeypatch.setattr(BamStore, "_process_chromosome", lambda *a, **kw: (a[2], np.full(a[3], int(a[1]), dtype=np.uint16), 10.0))
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
    monkeypatch.setattr(BamStore, "_process_chromosome", lambda *a, **kw: (a[2], np.full(a[3], int(a[1]), dtype=np.uint16), 10.0))
    store = BamStore(tmp_path / "ds", chromsizes, sample_names)
    store.process_samples(["1", "2"])

    region = store.extract_region(chrom="chr1", start=1, end=3)
    assert region.shape == (2, 2)

    default = store.extract_region(chrom="chr2")
    assert default.shape == (2, chromsizes["chr2"])


def test_extract_region_sample_subsetting(tmp_path, chromsizes, sample_names, monkeypatch):
    monkeypatch.setattr(BamStore, "_process_chromosome", lambda *a, **kw: (a[2], np.full(a[3], int(a[1]), dtype=np.uint16), 10.0))
    store = BamStore(tmp_path / "ds", chromsizes, sample_names)
    store.process_samples(["1", "2"])

    s1_only = store.extract_region("chr1:0-2", samples=["s1"])
    assert s1_only.shape == (1, 2)
    assert list(s1_only.coords["sample"].values) == ["s1"]
    assert np.all(s1_only.values == 1)

    by_idx = store.extract_region("chr1:0-2", samples=[1])
    assert list(by_idx.coords["sample"].values) == ["s2"]


def test_extract_region_as_numpy(tmp_path, chromsizes, sample_names, monkeypatch):
    monkeypatch.setattr(BamStore, "_process_chromosome", lambda *a, **kw: (a[2], np.full(a[3], int(a[1]), dtype=np.uint16), 10.0))
    store = BamStore(tmp_path / "ds", chromsizes, sample_names)
    store.process_samples(["1", "2"])

    np_result = store.extract_region("chr1:1-3", as_xarray=False)
    xr_result = store.extract_region("chr1:1-3", as_xarray=True)
    assert isinstance(np_result, np.ndarray)
    np.testing.assert_array_equal(np_result, xr_result.values)


def test_extract_region_metadata_coordinates(tmp_path, chromsizes, sample_names, monkeypatch):
    monkeypatch.setattr(BamStore, "_process_chromosome", lambda *a, **kw: (a[2], np.full(a[3], int(a[1]), dtype=np.uint16), 10.0))
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
    monkeypatch.setattr(BamStore, "_process_chromosome", lambda *a, **kw: (a[2], np.full(a[3], int(a[1]), dtype=np.uint16), 10.0))
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

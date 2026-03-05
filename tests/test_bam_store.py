def test_open_readonly_and_writable(tmp_path, chromsizes, sample_names, monkeypatch):
    # Fake per-chromosome processing to return constant arrays per sample.
    def fake_chrom(self, bam_file, contig, size):
        val = int(bam_file)
        arr = np.full(size, val, dtype=np.uint16)
        return contig, arr, 0.0

    monkeypatch.setattr(BamStore, "_process_chromosome", fake_chrom)

    bam_files = ["1", "2"]
    store = BamStore(tmp_path / "ds", chromsizes, sample_names)
    store.process_samples(bam_files)

    # Open in read-only mode (default)
    ro_store = BamStore.open(tmp_path / "ds")
    assert ro_store.read_only is True
    assert ro_store.sample_names == sample_names
    # Attempting to remove metadata should fail
    with pytest.raises(RuntimeError):
        ro_store.remove_metadata_columns(["sample_hash"])

    # Open in writable mode
    rw_store = BamStore.open(tmp_path / "ds", read_only=False)
    assert rw_store.read_only is False
    # Should be able to remove metadata without error
    rw_store.remove_metadata_columns(["sample_hash"])
    # Confirm sample_hashes are zeroed
    assert np.all(rw_store.meta["sample_hashes"][:] == 0)
import numpy as np
import pandas as pd
import pytest

from quantnado.dataset.bam import BamStore
from quantnado.dataset.core import QuantNadoDataset
from quantnado.dataset.reduce import reduce_byranges_signal


@pytest.fixture
def chromsizes():
    return {"chr1": 4, "chr2": 3}


@pytest.fixture
def sample_names():
    return ["s1", "s2"]


def test_bamstore_write_and_metadata(tmp_path, chromsizes, sample_names, monkeypatch):
    # Fake per-chromosome processing to return constant arrays per sample.
    def fake_chrom(self, bam_file, contig, size):
        val = int(bam_file)
        arr = np.full(size, val, dtype=np.uint16)
        return contig, arr, 0.0

    monkeypatch.setattr(BamStore, "_process_chromosome", fake_chrom)

    bam_files = ["1", "2"]
    store = BamStore(tmp_path / "ds", chromsizes, sample_names)
    store.process_samples(bam_files)

    # Verify data written per chromosome and sample.
    assert np.all(store.root["chr1"][0, :] == 1)
    assert np.all(store.root["chr2"][1, :] == 2)

    stored_names = [s.decode() if isinstance(s, (bytes, bytearray)) else s for s in store.root.attrs["sample_names"]]
    assert stored_names == sample_names
    assert store.completed_mask.tolist() == [True, True]
    assert np.isfinite(store.meta["sparsity"][:]).all()


def test_resume_validates_sample_names(tmp_path, chromsizes, sample_names, monkeypatch):
    def fake_chrom(self, bam_file, contig, size):
        arr = np.zeros(size, dtype=np.uint16)
        return contig, arr, 100.0

    monkeypatch.setattr(BamStore, "_process_chromosome", fake_chrom)

    bam_files = ["0", "0"]
    store = BamStore(tmp_path / "ds", chromsizes, sample_names)
    store.process_samples(bam_files)

    # Resume with same names should succeed.
    BamStore(tmp_path / "ds", chromsizes, sample_names, resume=True, overwrite=False)

    # Resume with mismatched names should fail.
    with pytest.raises(ValueError):
        BamStore(tmp_path / "ds", chromsizes, ["x", "y"], resume=True, overwrite=False)


def test_reduce_filters_incomplete_samples(tmp_path, chromsizes, sample_names, monkeypatch):
    def fake_chrom(self, bam_file, contig, size):
        val = int(bam_file)
        arr = np.full(size, val, dtype=np.uint16)
        return contig, arr, 0.0

    monkeypatch.setattr(BamStore, "_process_chromosome", fake_chrom)

    bam_files = ["1", "2"]
    store = BamStore(tmp_path / "ds", chromsizes, sample_names)
    store.process_samples(bam_files)

    # Mark second sample as incomplete to test filtering.
    store.meta["completed"][1] = False

    qd = QuantNadoDataset(store.store_path)
    ranges = pd.DataFrame({"contig": ["chr1"], "start": [0], "end": [2]})

    reduced = reduce_byranges_signal(qd, ranges_df=ranges)
    assert reduced["sum"].shape == (1, 1)
    assert reduced["sum"].values[0, 0] == 2  # only sample s1 (value 1 over length 2)

    reduced_all = reduce_byranges_signal(qd, ranges_df=ranges, include_incomplete=True)
    assert reduced_all["sum"].shape == (1, 2)
    assert reduced_all["sum"].values[0].tolist() == [2, 4]


def test_to_xarray_requires_complete(tmp_path, chromsizes, sample_names, monkeypatch):
    """Test that to_xarray() raises error if any sample is incomplete."""
    def fake_chrom(self, bam_file, contig, size):
        val = int(bam_file)
        arr = np.full(size, val, dtype=np.uint16)
        return contig, arr, 0.0

    monkeypatch.setattr(BamStore, "_process_chromosome", fake_chrom)

    bam_files = ["1", "2"]
    store = BamStore(tmp_path / "ds", chromsizes, sample_names)
    store.process_samples(bam_files)

    # Mark one sample as incomplete
    store.meta["completed"][0] = False

    # to_xarray() should raise RuntimeError
    with pytest.raises(RuntimeError, match="incomplete"):
        store.to_xarray()


def test_to_xarray_structure_and_metadata(tmp_path, chromsizes, sample_names, monkeypatch):
    """Test that to_xarray() creates correct DataArray structure and metadata."""
    def fake_chrom(self, bam_file, contig, size):
        val = int(bam_file)
        arr = np.full(size, val, dtype=np.uint16)
        return contig, arr, 10.0

    monkeypatch.setattr(BamStore, "_process_chromosome", fake_chrom)

    bam_files = ["1", "2"]
    store = BamStore(tmp_path / "ds", chromsizes, sample_names)
    store.process_samples(bam_files)
    
    # Add metadata columns
    store.set_metadata(pd.DataFrame({
        "sample_id": sample_names,
        "cell_type": ["A549", "HeLa"],
        "treatment": ["control", "treated"],
    }))

    # Extract all chromosomes
    xr_dict = store.to_xarray()
    assert set(xr_dict.keys()) == set(chromsizes.keys())

    # Extract specific chromosomes
    xr_dict_subset = store.to_xarray(chromosomes=["chr1"])
    assert set(xr_dict_subset.keys()) == {"chr1"}

    # Check structure
    for chrom, da_xr in xr_dict.items():
        assert da_xr.dims == ("sample", "position")
        assert list(da_xr.coords["sample"].values) == sample_names
        assert da_xr.shape[0] == len(sample_names)
        assert da_xr.shape[1] == chromsizes[chrom]

        # Check only sample_hashes in attributes
        assert "sample_hashes" in da_xr.attrs
        assert "completed" not in da_xr.attrs
        assert "sparsity" not in da_xr.attrs
        assert len(da_xr.attrs["sample_hashes"]) == len(sample_names)
        assert all(isinstance(h, str) for h in da_xr.attrs["sample_hashes"])
        
        # Check metadata columns as coordinates
        assert "cell_type" in da_xr.coords
        assert "treatment" in da_xr.coords
        assert list(da_xr.coords["cell_type"].values) == ["A549", "HeLa"]
        assert list(da_xr.coords["treatment"].values) == ["control", "treated"]
    
    # Test error on invalid chromosome
    with pytest.raises(ValueError, match="not in store"):
        store.to_xarray(chromosomes=["chrInvalid"])


def test_extract_region_string_format(tmp_path, chromsizes, sample_names, monkeypatch):
    """Test extract_region() with region string format."""
    def fake_chrom(self, bam_file, contig, size):
        val = int(bam_file)
        arr = np.full(size, val, dtype=np.uint16)
        return contig, arr, 10.0

    monkeypatch.setattr(BamStore, "_process_chromosome", fake_chrom)

    bam_files = ["1", "2"]
    store = BamStore(tmp_path / "ds", chromsizes, sample_names)
    store.process_samples(bam_files)
    
    # Add metadata
    store.set_metadata(pd.DataFrame({
        "sample_id": sample_names,
        "cell_type": ["A549", "HeLa"],
    }))

    # Test region string with commas
    region_xr = store.extract_region("chr1:1-3")
    assert region_xr.shape == (2, 2)  # 2 samples, positions 1-3 (exclusive)
    assert list(region_xr.coords["position"].values) == [1, 2]
    assert region_xr.attrs["chromosome"] == "chr1"
    assert region_xr.attrs["start"] == 1
    assert region_xr.attrs["end"] == 3
    
    # Test with commas in coordinates
    region_xr_comma = store.extract_region("chr2:0-2")
    assert region_xr_comma.shape == (2, 2)
    assert list(region_xr_comma.coords["position"].values) == [0, 1]
    
    # Test whole chromosome (no coordinates)
    whole_chr = store.extract_region("chr1")
    assert whole_chr.shape == (2, chromsizes["chr1"])
    assert whole_chr.attrs["start"] == 0
    assert whole_chr.attrs["end"] == chromsizes["chr1"]


def test_extract_region_separate_parameters(tmp_path, chromsizes, sample_names, monkeypatch):
    """Test extract_region() with separate chrom/start/end parameters."""
    def fake_chrom(self, bam_file, contig, size):
        val = int(bam_file)
        arr = np.full(size, val, dtype=np.uint16)
        return contig, arr, 10.0

    monkeypatch.setattr(BamStore, "_process_chromosome", fake_chrom)

    bam_files = ["1", "2"]
    store = BamStore(tmp_path / "ds", chromsizes, sample_names)
    store.process_samples(bam_files)

    # Test separate parameters
    region_xr = store.extract_region(chrom="chr1", start=1, end=3)
    assert region_xr.shape == (2, 2)
    assert list(region_xr.coords["position"].values) == [1, 2]
    
    # Test defaults: start=0, end=chrom_size
    region_default = store.extract_region(chrom="chr2")
    assert region_default.shape == (2, chromsizes["chr2"])
    assert region_default.attrs["start"] == 0
    assert region_default.attrs["end"] == chromsizes["chr2"]
    
    # Test partial defaults
    region_start_only = store.extract_region(chrom="chr1", start=2)
    assert region_start_only.shape == (2, chromsizes["chr1"] - 2)
    assert region_start_only.attrs["start"] == 2
    
    region_end_only = store.extract_region(chrom="chr1", end=2)
    assert region_end_only.shape == (2, 2)
    assert region_end_only.attrs["end"] == 2


def test_extract_region_sample_subsetting(tmp_path, chromsizes, sample_names, monkeypatch):
    """Test extract_region() with sample subsetting."""
    def fake_chrom(self, bam_file, contig, size):
        val = int(bam_file)
        arr = np.full(size, val, dtype=np.uint16)
        return contig, arr, 10.0

    monkeypatch.setattr(BamStore, "_process_chromosome", fake_chrom)

    bam_files = ["1", "2"]
    store = BamStore(tmp_path / "ds", chromsizes, sample_names)
    store.process_samples(bam_files)

    # Test sample names
    region_subset = store.extract_region("chr1:0-2", samples=["s1"])
    assert region_subset.shape == (1, 2)
    assert list(region_subset.coords["sample"].values) == ["s1"]
    assert np.all(region_subset.values == 1)  # bam_file "1" -> value 1
    
    # Test sample indices
    region_idx = store.extract_region("chr1:0-2", samples=[1])
    assert region_idx.shape == (1, 2)
    assert list(region_idx.coords["sample"].values) == ["s2"]
    assert np.all(region_idx.values == 2)  # bam_file "2" -> value 2
    
    # Test multiple samples
    region_multi = store.extract_region("chr1", samples=["s1", "s2"])
    assert region_multi.shape == (2, chromsizes["chr1"])


def test_extract_region_as_numpy(tmp_path, chromsizes, sample_names, monkeypatch):
    """Test extract_region() with as_xarray=False."""
    def fake_chrom(self, bam_file, contig, size):
        val = int(bam_file)
        arr = np.full(size, val, dtype=np.uint16)
        return contig, arr, 10.0

    monkeypatch.setattr(BamStore, "_process_chromosome", fake_chrom)

    bam_files = ["1", "2"]
    store = BamStore(tmp_path / "ds", chromsizes, sample_names)
    store.process_samples(bam_files)

    # Test numpy return
    region_np = store.extract_region("chr1:1-3", as_xarray=False)
    assert isinstance(region_np, np.ndarray)
    assert region_np.shape == (2, 2)
    
    # Test values match xarray version
    region_xr = store.extract_region("chr1:1-3", as_xarray=True)
    np.testing.assert_array_equal(region_np, region_xr.values)


def test_extract_region_metadata_coordinates(tmp_path, chromsizes, sample_names, monkeypatch):
    """Test that extract_region() includes metadata as coordinates."""
    def fake_chrom(self, bam_file, contig, size):
        val = int(bam_file)
        arr = np.full(size, val, dtype=np.uint16)
        return contig, arr, 10.0

    monkeypatch.setattr(BamStore, "_process_chromosome", fake_chrom)

    bam_files = ["1", "2"]
    store = BamStore(tmp_path / "ds", chromsizes, sample_names)
    store.process_samples(bam_files)
    
    # Add metadata
    store.set_metadata(pd.DataFrame({
        "sample_id": sample_names,
        "cell_type": ["A549", "HeLa"],
        "treatment": ["control", "treated"],
    }))

    region_xr = store.extract_region("chr1:0-2")
    
    # Check metadata coordinates
    assert "cell_type" in region_xr.coords
    assert "treatment" in region_xr.coords
    assert list(region_xr.coords["cell_type"].values) == ["A549", "HeLa"]
    assert list(region_xr.coords["treatment"].values) == ["control", "treated"]
    
    # Test with sample subsetting
    region_subset = store.extract_region("chr1:0-2", samples=["s2"])
    assert list(region_subset.coords["cell_type"].values) == ["HeLa"]
    assert list(region_subset.coords["treatment"].values) == ["treated"]


def test_extract_region_errors(tmp_path, chromsizes, sample_names, monkeypatch):
    """Test extract_region() error handling."""
    def fake_chrom(self, bam_file, contig, size):
        val = int(bam_file)
        arr = np.full(size, val, dtype=np.uint16)
        return contig, arr, 10.0

    monkeypatch.setattr(BamStore, "_process_chromosome", fake_chrom)

    bam_files = ["1", "2"]
    store = BamStore(tmp_path / "ds", chromsizes, sample_names)
    store.process_samples(bam_files)

    # Test invalid chromosome
    with pytest.raises(ValueError, match="not in store"):
        store.extract_region("chrInvalid:0-10")
    
    # Test both region and chrom specified
    with pytest.raises(ValueError, match="either 'region' or 'chrom'"):
        store.extract_region(region="chr1:0-10", chrom="chr1")
    
    # Test neither specified
    with pytest.raises(ValueError, match="Must specify"):
        store.extract_region()
    
    # Test invalid coordinates
    with pytest.raises(ValueError, match="must be greater than start"):
        store.extract_region("chr1:10-5")
    
    # Test coordinates out of bounds
    with pytest.raises(ValueError, match="exceeds chromosome size"):
        store.extract_region(chrom="chr1", start=0, end=1000)
    
    # Test negative coordinates
    with pytest.raises(ValueError, match="non-negative"):
        store.extract_region("chr1:-5-10")
    
    # Test invalid sample
    with pytest.raises(ValueError, match="not found"):
        store.extract_region("chr1:0-2", samples=["invalid_sample"])
    
    # Test sample index out of range
    with pytest.raises(ValueError, match="out of range"):
        store.extract_region("chr1:0-2", samples=[999])
    
    # Test incomplete sample
    store.meta["completed"][0] = False
    with pytest.raises(RuntimeError, match="incomplete"):
        store.extract_region("chr1:0-2")

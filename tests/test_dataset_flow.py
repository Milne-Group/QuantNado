import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantnado.analysis.core import QuantNadoDataset
from quantnado.analysis.counts import count_features
from quantnado.analysis.pca import run_pca
from quantnado.analysis.reduce import reduce_byranges_signal
from quantnado.dataset.store_bam import BamStore


@pytest.fixture
def simple_store(tmp_path, monkeypatch):
    """Create a small BamStore with deterministic values per sample."""

    chromsizes = {"chr1": 4}
    sample_names = ["s1", "s2"]

    def fake_chrom(self, bam_file, contig, size):
        val = int(bam_file)
        arr = np.full(size, val, dtype=np.uint16)
        sparsity = 0.0
        return contig, arr, sparsity

    monkeypatch.setattr(BamStore, "_process_chromosome", fake_chrom)

    store = BamStore(tmp_path / "ds", chromsizes, sample_names)
    store.process_samples(["1", "2"])
    return store


def test_quantnado_dataset_wrapper(simple_store):
    ds = QuantNadoDataset(simple_store.store_path)

    assert ds.sample_names == ["s1", "s2"]
    assert ds.completed_mask.tolist() == [True, True]
    assert ds.chromsizes == {"chr1": 4}
    np.testing.assert_array_equal(
        ds.get_chrom("chr1")[0, :], np.array([1, 1, 1, 1], dtype=np.uint32)
    )


def test_feature_counts_basic(simple_store):
    ranges = pd.DataFrame(
        {
            "contig": ["chr1", "chr1"],
            "start": [0, 2],
            "end": [2, 4],
            "gene_id": ["g1", "g2"],
        }
    )

    counts_df, feature_metadata = count_features(
        simple_store,
        ranges_df=ranges,
        contig_col="contig",
        feature_id_col="gene_id",
        integerize=True,
    )

    assert list(counts_df.columns) == ["s1", "s2"]
    assert counts_df.loc["g1", "s1"] == 2
    assert counts_df.loc["g1", "s2"] == 4
    assert counts_df.loc["g2", "s1"] == 2
    assert counts_df.loc["g2", "s2"] == 4

    assert feature_metadata["range_length"].tolist() == [2, 2]
    assert feature_metadata["contig"].tolist() == ["chr1", "chr1"]


def test_run_pca_nan_imputation_and_standardize():
    data = np.array(
        [
            [1.0, np.nan, 3.0],
            [1.0, 2.0, np.nan],
            [2.0, 2.0, 2.0],
        ]
    )
    darr = xr.DataArray(da.from_array(data, chunks=(2, 3)), dims=("sample", "feature"))

    pca_obj, transformed = run_pca(
        darr,
        n_components=2,
        nan_handling_strategy="mean_value_imputation",
        standardize=True,
        random_state=0,
    )

    transformed_arr = transformed.compute()
    assert transformed_arr.shape == (3, 2)
    assert np.all(np.isfinite(transformed_arr))
    assert np.allclose(transformed_arr.mean(axis=0), 0.0, atol=1e-6)


def test_run_pca_drops_nan_columns():
    data = np.array([[1.0, np.nan], [2.0, 3.0]])
    darr = xr.DataArray(da.from_array(data, chunks=(2, 2)), dims=("sample", "feature"))

    pca_obj, transformed = run_pca(
        darr,
        n_components=1,
        nan_handling_strategy="drop",
        random_state=1,
    )

    transformed_arr = transformed.compute()
    assert transformed_arr.shape == (2, 1)
    assert pca_obj.n_components == 1


def test_bamstore_hash_validation_on_resume(tmp_path, monkeypatch):
    chromsizes = {"chr1": 4}
    names = ["s1", "s2"]

    def fake_chrom(self, bam_file, contig, size):
        return contig, np.zeros(size), 0.0

    monkeypatch.setattr(BamStore, "_process_chromosome", fake_chrom)

    # Initial create
    BamStore(tmp_path / "ds", chromsizes, names)

    # Resume with same names should work
    BamStore(tmp_path / "ds", chromsizes, names, resume=True, overwrite=False)

    # Resume with different order should fail hash check (and list match)
    with pytest.raises(ValueError, match="names do not match"):
        BamStore(tmp_path / "ds", chromsizes, ["s2", "s1"], resume=True, overwrite=False)


def test_bamstore_metadata_partial_updates(tmp_path, monkeypatch):
    chromsizes = {"chr1": 4}
    names = ["s1", "s2"]

    store = BamStore(tmp_path / "ds", chromsizes, names)

    # 1. Full dataframe update
    meta_df = pd.DataFrame(
        {"sample_id": ["s1", "s2"], "group": ["A", "B"], "assay": ["ATAC", "RNA"]}
    )
    store.set_metadata(meta_df)

    md = store.get_metadata()
    assert md.loc["s1", "group"] == "A"
    assert md.loc["s2", "assay"] == "RNA"

    # 2. Partial sample update (subset)
    partial_df = pd.DataFrame({"sample_id": ["s1"], "group": ["C"]})
    store.set_metadata(partial_df, merge=True)

    md2 = store.get_metadata()
    assert md2.loc["s1", "group"] == "C"
    assert md2.loc["s2", "group"] == "B"  # Preserved

    # 3. Dictionary update (sparse)
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


def test_bamstore_sample_hash_alignment(tmp_path, monkeypatch):
    # Create two dummy files with different content
    f1 = tmp_path / "1.bam"
    f1.write_bytes(b"content1" * 1000)
    f2 = tmp_path / "2.bam"
    f2.write_bytes(b"content2" * 1000)

    chromsizes = {"chr1": 10}
    names = ["s1", "s2"]

    # Mock _process_chromosome to avoid bamnado
    monkeypatch.setattr(
        BamStore, "_process_chromosome", lambda *args, **kwargs: (args[2], np.zeros(args[3]), 0.0)
    )

    store = BamStore(tmp_path / "ds", chromsizes, names)
    store.process_samples([str(f1), str(f2)])

    hashes = store.sample_hashes
    assert len(hashes) == 2
    assert hashes[0] != hashes[1]
    assert len(hashes[0]) == 32  # MD5 hex

    # Verify get_metadata includes it
    md = store.get_metadata()
    assert "sample_hash" in md.columns
    assert md.loc["s1", "sample_hash"] == hashes[0]

    # Test set_metadata with CORRECT hashes (should pass)
    meta_ok = pd.DataFrame({"sample_id": ["s1", "s2"], "sample_hash": hashes})
    store.set_metadata(meta_ok)

    # Test set_metadata with WRONG hashes (should fail)
    meta_bad = pd.DataFrame({"sample_id": ["s1"], "sample_hash": ["wrong_hash"]})
    with pytest.raises(ValueError, match="Sample hash mismatch"):
        store.set_metadata(meta_bad)


def test_bamstore_auto_chromsizes(tmp_path, monkeypatch):
    # Mock _get_chromsizes_from_bam
    monkeypatch.setattr(
        "quantnado.dataset.store_bam._get_chromsizes_from_bam",
        lambda path: {"chr1": 100, "chr2": 200},
    )
    # Mock _process_chromosome to avoid bamnado
    monkeypatch.setattr(
        BamStore, "_process_chromosome", lambda *args, **kwargs: (args[2], np.zeros(args[3]), 0.0)
    )

    # Create dummy BAM path
    bam_path = tmp_path / "test.bam"
    bam_path.write_text("dummy")

    store = BamStore.from_bam_files(
        bam_files=[str(bam_path)],
        store_path=tmp_path / "auto_ds",
        chromsizes=None,  # Triggers auto-extraction
    )

    assert store.chromsizes == {"chr1": 100, "chr2": 200}
    assert store.sample_names == ["test"]


def test_bamstore_metadata_json_roundtrip(tmp_path):
    store = BamStore(tmp_path / "ds", {"chr1": 10}, ["s1", "s2"])
    store.update_metadata({"info": ["x", "y"]})

    json_path = tmp_path / "meta.json"
    store.metadata_to_json(json_path)

    # Verify reload
    reloaded = BamStore.metadata_from_json(json_path)
    assert reloaded.set_index("sample_id").loc["s2", "info"] == "y"


def test_reduce_with_bed_file_max(simple_store, tmp_path):
    bed_path = tmp_path / "ranges.bed"
    pd.DataFrame({"contig": ["chr1"], "start": [0], "end": [4]}).to_csv(
        bed_path, sep="\t", header=False, index=False
    )

    reduced = reduce_byranges_signal(
        simple_store,
        intervals_path=str(bed_path),
        reduction="max",
        include_incomplete=True,
    )

    assert list(reduced.sample.values) == ["s1", "s2"]
    assert reduced["max"].shape == (1, 2)
    assert reduced["max"].values.tolist() == [[1, 2]]


def test_reduce_respects_completed_mask(simple_store):
    simple_store.meta["completed"][:] = False
    with pytest.raises(ValueError):
        reduce_byranges_signal(
            simple_store,
            ranges_df=pd.DataFrame({"contig": ["chr1"], "start": [0], "end": [2]}),
            include_incomplete=False,
        )


def test_reduce_invalid_contig_raises(simple_store):
    with pytest.raises(ValueError):
        reduce_byranges_signal(
            simple_store,
            ranges_df=pd.DataFrame({"contig": ["chrX"], "start": [0], "end": [2]}),
        )

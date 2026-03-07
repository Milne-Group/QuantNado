"""Integration tests for the QuantNado high-level facade."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantnado.dataset.store_bam import BamStore
from quantnado import QuantNado


@pytest.fixture
def qn(simple_store):
    """QuantNado facade wrapping the shared simple_store fixture."""
    return QuantNado(simple_store)


# ---------------------------------------------------------------------------
# Factory methods
# ---------------------------------------------------------------------------


def test_open_wraps_bamstore(tmp_path, chromsizes, sample_names, monkeypatch):
    monkeypatch.setattr(BamStore, "_process_chromosome", lambda *a, **kw: (a[2], np.full(a[3], int(a[1])), 0.0))
    BamStore(tmp_path / "ds", chromsizes, sample_names).process_samples(["1", "2"])

    qn = QuantNado.open(tmp_path / "ds")
    assert isinstance(qn, QuantNado)
    assert qn.samples == sample_names


def test_from_bam_files(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "quantnado.dataset.store_bam._get_chromsizes_from_bam",
        lambda path: {"chr1": 10},
    )
    monkeypatch.setattr(BamStore, "_process_chromosome", lambda *a, **kw: (a[2], np.zeros(a[3]), 0.0))

    bam = tmp_path / "s1.bam"
    bam.write_text("dummy")
    qn = QuantNado.from_bam_files(bam_files=[str(bam)], store_path=tmp_path / "ds", chromsizes=None)
    assert isinstance(qn, QuantNado)
    assert qn.samples == ["s1"]


def test_from_bam_files_with_local_staging(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "quantnado.dataset.store_bam._get_chromsizes_from_bam",
        lambda path: {"chr1": 10},
    )
    monkeypatch.setattr(BamStore, "_process_chromosome", lambda *a, **kw: (a[2], np.zeros(a[3]), 0.0))

    bam = tmp_path / "s1.bam"
    bam.write_text("dummy")
    staging_dir = tmp_path / "scratch"
    final_store = tmp_path / "staged_api_ds"

    qn = QuantNado.from_bam_files(
        bam_files=[str(bam)],
        store_path=final_store,
        chromsizes=None,
        local_staging=True,
        staging_dir=staging_dir,
    )

    assert isinstance(qn, QuantNado)
    assert qn.store_path == final_store.with_suffix(".zarr")
    assert qn.store_path.exists()
    assert list(staging_dir.iterdir()) == []


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_samples_property(qn, sample_names):
    assert qn.samples == sample_names


def test_chromosomes_property(qn, chromsizes):
    assert set(qn.chromosomes) == set(chromsizes.keys())


def test_chromsizes_property(qn, chromsizes):
    assert qn.chromsizes == chromsizes


def test_metadata_property(qn, sample_names):
    md = qn.metadata
    assert isinstance(md, pd.DataFrame)
    assert list(md.index) == sample_names


def test_store_path_property(qn, tmp_path):
    from pathlib import Path
    assert isinstance(qn.store_path, Path)


# ---------------------------------------------------------------------------
# to_xarray
# ---------------------------------------------------------------------------


def test_to_xarray_returns_dict(qn, chromsizes):
    xr_dict = qn.to_xarray()
    assert set(xr_dict.keys()) == set(chromsizes.keys())
    for chrom, da in xr_dict.items():
        assert da.dims == ("sample", "position")
        assert da.shape[1] == chromsizes[chrom]


def test_to_xarray_chromosome_subset(qn):
    result = qn.to_xarray(chromosomes=["chr1"])
    assert set(result.keys()) == {"chr1"}


# ---------------------------------------------------------------------------
# extract_region
# ---------------------------------------------------------------------------


def test_extract_region_string(qn, chromsizes, sample_names):
    region = qn.extract_region("chr1:0-2")
    assert region.shape == (len(sample_names), 2)


def test_extract_region_separate_params(qn, chromsizes, sample_names):
    region = qn.extract_region(chrom="chr1", start=0, end=2)
    assert region.shape == (len(sample_names), 2)


def test_extract_region_sample_filter(qn):
    result = qn.extract_region("chr1:0-2", samples=["s1"])
    assert result.shape[0] == 1
    assert list(result.coords["sample"].values) == ["s1"]


# ---------------------------------------------------------------------------
# reduce
# ---------------------------------------------------------------------------


def test_reduce_returns_dataset(qn):
    ranges = pd.DataFrame({"contig": ["chr1"], "start": [0], "end": [4]})
    result = qn.reduce(ranges_df=ranges)
    assert "mean" in result
    assert "sum" in result
    assert result["sum"].shape == (1, 2)


def test_reduce_from_bed(qn, tmp_path):
    bed = tmp_path / "regions.bed"
    pd.DataFrame({"c": ["chr1"], "s": [0], "e": [4]}).to_csv(bed, sep="\t", header=False, index=False)
    result = qn.reduce(intervals_path=str(bed))
    assert result["mean"].shape == (1, 2)


def test_reduce_correct_values(qn):
    ranges = pd.DataFrame({"contig": ["chr1"], "start": [0], "end": [4]})
    result = qn.reduce(ranges_df=ranges, reduction="sum")
    # s1=1, s2=2 at every position; chr1 has 4 positions
    np.testing.assert_array_equal(result["sum"].values[0], [4, 8])


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------


def test_extract_returns_dataarray(qn):
    import xarray as xr
    ranges = pd.DataFrame({"contig": ["chr1"], "start": [0], "end": [4]})
    result = qn.extract(ranges_df=ranges)
    assert isinstance(result, xr.DataArray)
    assert result.dims == ("interval", "relative_position", "sample")


def test_extract_fixed_width(qn):
    ranges = pd.DataFrame({"contig": ["chr1"], "start": [0], "end": [4]})
    result = qn.extract(ranges_df=ranges, fixed_width=4)
    assert result.shape[1] == 4


# ---------------------------------------------------------------------------
# count_features
# ---------------------------------------------------------------------------


def test_feature_counts_from_ranges(qn):
    ranges = pd.DataFrame({
        "contig": ["chr1", "chr1"],
        "start": [0, 2],
        "end": [2, 4],
        "gene_id": ["g1", "g2"],
    })
    counts_df, meta = qn.count_features(ranges=ranges)
    assert "s1" in counts_df.columns
    assert "s2" in counts_df.columns
    assert counts_df.shape[0] == 2


# ---------------------------------------------------------------------------
# pca
# ---------------------------------------------------------------------------


def test_pca_returns_object_and_transformed(qn):
    import xarray as xr
    import dask.array as da

    ranges = pd.DataFrame({"contig": ["chr1", "chr2"], "start": [0, 0], "end": [4, 3]})
    reduced = qn.reduce(ranges_df=ranges)
    pca_obj, transformed = qn.pca(reduced["mean"].transpose(), n_components=1)
    assert transformed.compute().shape[0] == 2  # 2 samples
    assert transformed.compute().shape[1] == 1  # 1 component

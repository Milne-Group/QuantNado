from __future__ import annotations

import numpy as np
import pytest

import bamnado
import xarray as xr

from quantnado.cli import _parse_paired_value, _parse_stranded_value
from quantnado.analysis.core import (
    _PlotnadoCoverageAdapter,
    _orient_rna_strands,
)
from quantnado.dataset.store_bam import BamStore


@pytest.mark.parametrize("library_strandedness", ["R", "F"])
def test_bamstore_stranded_rna_uses_genomic_forward_reverse_filters(
    tmp_path, monkeypatch, library_strandedness
):
    calls: list[str] = []

    def fake_get_signal_for_chromosome(
        *,
        bam_path,
        chromosome_name,
        bin_size,
        scale_factor,
        use_fragment,
        ignore_scaffold_chromosomes,
        read_filter,
    ):
        calls.append(read_filter.strand)
        fill = 1 if read_filter.strand == "forward" else 2
        return np.full(5, fill, dtype=np.uint32)

    monkeypatch.setattr(bamnado, "get_signal_for_chromosome", fake_get_signal_for_chromosome)

    store = BamStore(
        tmp_path / f"rna_{library_strandedness}.zarr",
        assay="RNA",
        sample="rna-1",
        chromsizes={"chr1": 5},
        stranded=library_strandedness,
        chunk_len=2,
        compressors=[],
    )
    fwd, rev = store._process_chromosome(
        bam_file="dummy.bam",
        chrom="chr1",
        chrom_size=5,
        read_filter=bamnado.ReadFilter(),
        use_fragment=False,
    )

    assert calls == ["forward", "reverse"]
    np.testing.assert_array_equal(fwd, np.full(5, 1, dtype=np.uint32))
    np.testing.assert_array_equal(rev, np.full(5, 2, dtype=np.uint32))


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("R", "R"),
        ("reverse", "R"),
        ("2", "R"),
        ("F", "F"),
        ("forward", "F"),
        ("1", "F"),
        ("U", None),
        ("unstranded", None),
        ("0", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_stranded_value_normalizes_common_metadata_values(raw, expected):
    assert _parse_stranded_value(raw) == expected


def test_parse_stranded_value_rejects_unknown_values():
    with pytest.raises(ValueError, match="Invalid stranded value"):
        _parse_stranded_value("weird")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("paired", True),
        ("paired-end", True),
        ("paired end", True),
        ("PE", True),
        ("true", True),
        ("1", True),
        ("single", False),
        ("single-end", False),
        ("single end", False),
        ("SE", False),
        ("false", False),
        ("0", False),
        ("", True),
        (None, True),
    ],
)
def test_parse_paired_value_normalizes_common_metadata_values(raw, expected):
    assert _parse_paired_value(raw) == expected


def test_parse_paired_value_rejects_unknown_values():
    with pytest.raises(ValueError, match="Invalid paired value"):
        _parse_paired_value("weird")


class _FakeDataset:
    sample_names = ["rna-1"]

    def __init__(self, stranded: str):
        self._stranded = stranded

    def _get_stranded_per_sample(self):
        return [self._stranded]

    def sel(self, chrom, start, end, samples=None):
        coords = {"sample": ["rna-1"], "position": [start, start + 1]}
        ds = xr.Dataset(
            {
                "rna_fwd": (("sample", "position"), np.array([[1.0, 2.0]])),
                "rna_rev": (("sample", "position"), np.array([[10.0, 20.0]])),
            },
            coords=coords,
        )
        return _orient_rna_strands(ds, {"rna-1": self._stranded})


def test_plotnado_adapter_swaps_rna_strands_for_reverse_libraries():
    adapter = _PlotnadoCoverageAdapter(_FakeDataset("R"))
    plus = adapter.extract_region("chr1:10-12", samples=["rna-1"], strand="+")
    minus = adapter.extract_region("chr1:10-12", samples=["rna-1"], strand="-")
    np.testing.assert_array_equal(plus.values, np.array([[10.0, 20.0]]))
    np.testing.assert_array_equal(minus.values, np.array([[1.0, 2.0]]))


def test_plotnado_adapter_keeps_forward_libraries_in_place():
    adapter = _PlotnadoCoverageAdapter(_FakeDataset("F"))
    plus = adapter.extract_region("chr1:10-12", samples=["rna-1"], strand="+")
    minus = adapter.extract_region("chr1:10-12", samples=["rna-1"], strand="-")
    np.testing.assert_array_equal(plus.values, np.array([[1.0, 2.0]]))
    np.testing.assert_array_equal(minus.values, np.array([[10.0, 20.0]]))


def test_rna_from_bam_files_defaults_to_fragment_counting(tmp_path, monkeypatch):
    seen = {}

    monkeypatch.setattr(
        "quantnado.dataset.store_bam._get_chromsizes_from_bam",
        lambda bam_path: {"chr1": 5},
    )

    def fake_write_coverage(self, bam_file, read_filter, use_fragment):
        seen["use_fragment"] = use_fragment
        return 0.0

    monkeypatch.setattr(BamStore, "_write_coverage", fake_write_coverage)
    monkeypatch.setattr(BamStore, "_finalise", lambda self, bam_file, sparsity: None)

    BamStore.from_bam_files(
        bam_path="dummy.bam",
        store_path=tmp_path / "rna.zarr",
        assay="RNA",
        sample="rna-1",
        stranded="R",
        chromsizes={"chr1": 5},
    )

    assert seen["use_fragment"] is True


def test_rna_from_bam_files_single_end_uses_read_counting(tmp_path, monkeypatch):
    seen = {}

    monkeypatch.setattr(
        "quantnado.dataset.store_bam._get_chromsizes_from_bam",
        lambda bam_path: {"chr1": 5},
    )

    def fake_write_coverage(self, bam_file, read_filter, use_fragment):
        seen["use_fragment"] = use_fragment
        seen["proper_pair"] = read_filter.proper_pair
        return 0.0

    monkeypatch.setattr(BamStore, "_write_coverage", fake_write_coverage)
    monkeypatch.setattr(BamStore, "_finalise", lambda self, bam_file, sparsity: None)

    BamStore.from_bam_files(
        bam_path="dummy.bam",
        store_path=tmp_path / "rna.zarr",
        assay="RNA",
        sample="rna-1",
        stranded="R",
        chromsizes={"chr1": 5},
        paired=False,
    )

    assert seen["use_fragment"] is False
    assert seen["proper_pair"] is False


def test_orient_rna_strands_only_swaps_reverse_libraries():
    coords = {"sample": ["atac", "rna-r", "rna-f"], "position": [10, 11]}
    ds = xr.Dataset(
        {
            "rna_fwd": (("sample", "position"), np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])),
            "rna_rev": (("sample", "position"), np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]])),
        },
        coords=coords,
    )
    out = _orient_rna_strands(ds, {"rna-r": "R", "rna-f": "F"})

    np.testing.assert_array_equal(out["rna_fwd"].sel(sample="atac").values, np.array([1.0, 1.0]))
    np.testing.assert_array_equal(out["rna_rev"].sel(sample="atac").values, np.array([10.0, 10.0]))
    np.testing.assert_array_equal(out["rna_fwd"].sel(sample="rna-r").values, np.array([20.0, 20.0]))
    np.testing.assert_array_equal(out["rna_rev"].sel(sample="rna-r").values, np.array([2.0, 2.0]))
    np.testing.assert_array_equal(out["rna_fwd"].sel(sample="rna-f").values, np.array([3.0, 3.0]))
    np.testing.assert_array_equal(out["rna_rev"].sel(sample="rna-f").values, np.array([30.0, 30.0]))

from __future__ import annotations

import textwrap

import numpy as np
import xarray as xr
import pytest

from quantnado.analysis import counts as counts_module


pytestmark = pytest.mark.unit


class _DatasetStub:
    sample_names = ["S1", "S2"]


@pytest.fixture
def exon_gtf_file(tmp_path):
    gtf_content = textwrap.dedent(
        """\
        chr1\tENSEMBL\texon\t1001\t1100\t.\t+\t.\tgene_id "G1"; gene_name "GENE1"; eid "E1";
        chr1\tENSEMBL\texon\t1201\t1300\t.\t+\t.\tgene_id "G1"; gene_name "GENE1"; eid "E2";
        chr1\tENSEMBL\texon\t2001\t2100\t.\t-\t.\tgene_id "G2"; gene_name "GENE2"; eid "E3";
        """
    )
    path = tmp_path / "exons.gtf"
    path.write_text(gtf_content)
    return str(path)


def _fake_reduced():
    return xr.Dataset(
        data_vars={
            "sum": (("ranges", "sample"), np.array([[10.0, 1.0], [20.0, 2.0], [30.0, 3.0]])),
            "start": ("ranges", np.array([1000, 1200, 2000])),
            "end": ("ranges", np.array([1100, 1300, 2100])),
            "range_length": ("ranges", np.array([100, 100, 100])),
        },
        coords={
            "ranges": np.array([0, 1, 2]),
            "sample": np.array(["S1", "S2"]),
            "range_index": ("ranges", np.array([0, 1, 2])),
            "contig": ("ranges", np.array(["chr1", "chr1", "chr1"])),
        },
    )


def test_count_features_exon_aggregates_by_feature_id_attr(monkeypatch, exon_gtf_file):
    monkeypatch.setattr(
        counts_module,
        "reduce_byranges_signal",
        lambda *args, **kwargs: _fake_reduced(),
    )

    counts_df, feature_metadata = counts_module.count_features(
        _DatasetStub(),
        gtf_file=exon_gtf_file,
        feature_type="exon",
        feature_id_attr="gene_name",
        integerize=False,
    )

    assert list(counts_df.index) == ["GENE1", "GENE2"]
    assert counts_df.loc["GENE1", "S1"] == 30.0
    assert counts_df.loc["GENE1", "S2"] == 3.0
    assert counts_df.loc["GENE2", "S1"] == 30.0
    assert feature_metadata.loc["GENE1", "start"] == 1000
    assert feature_metadata.loc["GENE1", "end"] == 1300
    assert feature_metadata.loc["GENE1", "range_length"] == 200


def test_count_features_loads_custom_gtf_attribute_ids(monkeypatch, exon_gtf_file):
    monkeypatch.setattr(
        counts_module,
        "reduce_byranges_signal",
        lambda *args, **kwargs: _fake_reduced(),
    )

    counts_df, feature_metadata = counts_module.count_features(
        _DatasetStub(),
        gtf_file=exon_gtf_file,
        feature_type="exon",
        feature_id_attr="eid",
        integerize=False,
    )

    assert list(counts_df.index) == ["E1", "E2", "E3"]
    assert counts_df.loc["E2", "S1"] == 20.0
    assert counts_df.loc["E3", "S2"] == 3.0
    assert feature_metadata.loc["E1", "eid"] == "E1"
    assert feature_metadata.loc["E3", "eid"] == "E3"

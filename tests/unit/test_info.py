from __future__ import annotations

import pandas as pd
import xarray as xr

from quantnado.analysis.core import QuantNadoDataset


class _FakeSubset:
    def __init__(self, sample_names, array_keys, ips=None):
        self.sample_names = sample_names
        self.array_keys = array_keys
        self._ips = ips or []

    def _get_ip_per_sample(self):
        return self._ips


class _FakeInfoQN:
    info = QuantNadoDataset.info
    group_by = QuantNadoDataset.group_by

    def __init__(self):
        self._combined = True
        self.path = "fake_dataset.zarr"
        self._subset_samples = None
        self._group_sets = {}
        self._last_group_name = None
        self._assays = ["ATAC", "CHIP"]
        self._chromosomes = ["chr13", "chr21"]
        self._chromsizes = {"chr13": 114364328, "chr21": 46709983}
        self._subsets = {
            "ATAC": _FakeSubset(["ATAC-1", "ATAC-2"], ["coverage"], ["", ""]),
            "CHIP": _FakeSubset(["ChIP-1"], ["coverage"], ["MLLN"]),
        }

    @property
    def assays(self):
        return self._assays

    @property
    def chromosomes(self):
        return self._chromosomes

    @property
    def chromsizes(self):
        return self._chromsizes

    @property
    def sample_names(self):
        return ["ATAC-1", "ATAC-2", "ChIP-1"]

    def _get_assay_per_sample(self):
        return ["ATAC", "ATAC", "CHIP"]

    def _get_ip_per_sample(self):
        return ["", "", "MLLN"]

    def _get_stranded_per_sample(self):
        return ["", "", ""]

    def subset(self, assay=None, samples=None):
        return self._subsets[assay]


def test_info_prints_and_returns_summary(capsys):
    qn = _FakeInfoQN()

    summary = qn.info
    out = capsys.readouterr().out

    assert out == ""

    assert summary["assays"] == ["ATAC", "CHIP"]
    assert summary["chromosomes"] == ["chr13", "chr21"]
    assert summary["chromsizes"] == {"chr13": 114364328, "chr21": 46709983}
    assert summary["extras"]["layout"] == "combined"
    assert summary["extras"]["path"] == "fake_dataset.zarr"
    assert summary["extras"]["subset"] is False
    assert summary["extras"]["groups"] == {}
    assert summary["per_assay"]["ATAC"]["n_samples"] == 2
    assert summary["per_assay"]["ATAC"]["sample_names"] == ["ATAC-1", "ATAC-2"]
    assert summary["per_assay"]["CHIP"]["array_keys"] == ["coverage"]
    assert summary["per_assay"]["CHIP"]["ips"] == ["MLLN"]
    rendered = repr(summary)
    assert "DatasetInfo" in rendered
    assert "assays      : ATAC, CHIP" in rendered
    assert "chromosomes : chr13, chr21" in rendered
    assert "chromsizes  : chr13=114,364,328, chr21=46,709,983" in rendered
    assert "layout      : combined" in rendered
    assert "ATAC" in rendered
    assert "n        : 2" in rendered
    assert "samples   : ATAC-1, ATAC-2" in rendered
    assert "keys      : coverage" in rendered
    assert "ips       : MLLN" in rendered


def test_normalised_dataset_info_reports_method():
    qn = _FakeInfoQN()
    from quantnado.analysis.core import NormalisedQuantNadoDataset

    norm = NormalisedQuantNadoDataset(qn, method="cpm", library_sizes={"ATAC-1": 1, "ATAC-2": 1, "ChIP-1": 1})
    summary = norm.info

    assert summary["extras"]["normalised"] is True
    assert summary["extras"]["normalise_method"] == "cpm"
    rendered = repr(summary)
    assert "normalised  : True" in rendered
    assert "normalise_method : cpm" in rendered


def test_info_of_summarises_xarray_and_pandas_objects():
    qn = _FakeInfoQN()

    da = xr.DataArray(
        [[1.0, 2.0]],
        dims=("sample", "position"),
        coords={"sample": ["s1"], "position": [1, 2]},
        name="rna_fwd_cpm",
    )
    ds = xr.Dataset({"coverage": da})
    df = pd.DataFrame({"a": [1, 2]})

    da_info = QuantNadoDataset.info_of(qn, da)
    ds_info = QuantNadoDataset.info_of(qn, ds)
    df_info = QuantNadoDataset.info_of(qn, df)

    assert da_info["type"] == "DataArray"
    assert da_info["name"] == "rna_fwd_cpm"
    assert da_info["dims"] == ["sample", "position"]
    assert ds_info["type"] == "Dataset"
    assert ds_info["data_vars"] == ["coverage"]
    assert df_info["type"] == "DataFrame"
    assert df_info["shape"] == [2, 1]


def test_groups_returns_assay_to_samples_mapping():
    qn = _FakeInfoQN()
    groups = QuantNadoDataset.groups.__get__(qn, _FakeInfoQN)

    assert groups["ATAC"] == ["ATAC-1", "ATAC-2"]
    assert groups["CHIP"] == ["ChIP-1"]
    rendered = repr(groups)
    assert "GroupInfo" in rendered
    assert "ATAC" in rendered
    assert "ATAC-1, ATAC-2" in rendered


def test_group_by_ip_and_custom_groups():
    qn = _FakeInfoQN()

    ip_groups = QuantNadoDataset.group_by(qn, "ip")
    assert ip_groups["MLLN"] == ["ChIP-1"]

    custom = QuantNadoDataset.group_by(
        qn,
        groups={
            "control": ["ATAC-1", "ATAC-2"],
            "treated": ["ChIP-1"],
        },
    )
    assert custom["control"] == ["ATAC-1", "ATAC-2"]
    assert custom["treated"] == ["ChIP-1"]


def test_group_by_custom_contains_patterns():
    class FakeNames(_FakeInfoQN):
        @property
        def sample_names(self):
            return [
                "rna-spikein-control-rep1",
                "rna-spikein-control-rep2",
                "rna-spikein-treated-rep1",
                "rna-spikein-treated-rep2",
            ]

        def _get_assay_per_sample(self):
            return ["RNA", "RNA", "RNA", "RNA"]

        def _get_ip_per_sample(self):
            return ["", "", "", ""]

    qn = FakeNames()
    groups = QuantNadoDataset.group_by(
        qn,
        groups={
            "control": "control",
            "treated": "treated",
        },
        match="contains",
    )

    assert groups["control"] == [
        "rna-spikein-control-rep1",
        "rna-spikein-control-rep2",
    ]
    assert groups["treated"] == [
        "rna-spikein-treated-rep1",
        "rna-spikein-treated-rep2",
    ]
    assert qn._group_sets["group"]["control"] == groups["control"]


def test_group_by_multiple_named_group_sets_and_info_summary():
    class FakeNames(_FakeInfoQN):
        @property
        def sample_names(self):
            return [
                "chip-rx_MLL",
                "rna-control-rep1-WT-1hr",
                "rna-treated-rep2-KO-2hr",
            ]

        def _get_assay_per_sample(self):
            return ["CHIP", "RNA", "RNA"]

        def _get_ip_per_sample(self):
            return ["MLL", "", ""]

    qn = FakeNames()
    grouped = QuantNadoDataset.group_by(
        qn,
        ip="ip",
        treatment={"control": ["control"], "treated": ["treated"]},
        replicate={"rep1": ["rep1"], "rep2": ["rep2"]},
        genotype={"WT": ["WT"], "KO": ["KO"]},
        match="contains",
    )

    assert grouped["ip"]["MLL"] == ["chip-rx_MLL"]
    assert grouped["treatment"]["control"] == ["rna-control-rep1-WT-1hr"]
    assert grouped["replicate"]["rep2"] == ["rna-treated-rep2-KO-2hr"]
    assert "NamedGroupInfo" in repr(grouped)
    summary = qn.info
    assert summary["extras"]["groups"]["ip"] == {"labels": ["MLL"], "n": 1}
    assert summary["extras"]["groups"]["treatment"] == {"labels": ["control", "treated"], "n": 2}
    rendered = repr(summary)
    assert "groups" in rendered
    assert "ip       : MLL (1)" in rendered
    assert "treatment: control, treated (2)" in rendered


def test_group_by_supports_named_metadata_shorthand_ip():
    qn = _FakeInfoQN()
    grouped = QuantNadoDataset.group_by(
        qn,
        ip="ip",
        treatment={"treated": ["ChIP"]},
        match="contains",
    )

    assert grouped["ip"]["MLLN"] == ["ChIP-1"]
    assert grouped["treatment"]["treated"] == ["ChIP-1"]


def test_group_by_named_sets_accumulate_cached_namespaces():
    qn = _FakeInfoQN()
    QuantNadoDataset.group_by(
        qn,
        condition={"old": ["ATAC"]},
        match="contains",
    )
    QuantNadoDataset.group_by(
        qn,
        ip="ip",
        replicate={"rep1": ["ATAC"], "rep2": ["ChIP"]},
        match="contains",
    )

    assert sorted(qn._group_sets) == ["condition", "ip", "replicate"]


def test_group_by_contains_supports_multiple_patterns_per_label():
    class FakeNames(_FakeInfoQN):
        @property
        def sample_names(self):
            return [
                "chip-rx_MLL",
                "rna-spikein-control-rep1",
                "rna-treated-rep1",
            ]

        def _get_assay_per_sample(self):
            return ["CHIP", "RNA", "RNA"]

        def _get_ip_per_sample(self):
            return ["MLL", "", ""]

    qn = FakeNames()
    grouped = QuantNadoDataset.group_by(
        qn,
        spikein={
            "spikein": ["spikein", "rx"],
        },
        match="contains",
    )

    assert grouped["spikein"]["spikein"] == [
        "chip-rx_MLL",
        "rna-spikein-control-rep1",
    ]


def test_infer_ip_from_sample_name_fallback():
    assert QuantNadoDataset._infer_ip_from_sample_name("chip-rx_MLL", "CHIP") == "MLL"
    assert QuantNadoDataset._infer_ip_from_sample_name("CAT-SEM_H3K27ac", "CUT&TAG") == "H3K27ac"
    assert QuantNadoDataset._infer_ip_from_sample_name("atac", "ATAC") == ""


def test_group_by_ip_falls_back_when_metadata_column_missing():
    class FakeCombinedNoIp(_FakeInfoQN):
        def _get_ip_per_sample(self):
            return QuantNadoDataset._get_ip_per_sample(self)

        def _get_assay_per_sample(self):
            return ["CHIP", "CUT&TAG", "RNA"]

        @property
        def sample_names(self):
            return ["chip-rx_MLL", "CAT-SEM_H3K27ac", "rna-spikein-control-rep1"]

        _combined = True
        _subset_samples = None

        class _Root:
            attrs = {"sample_names": ["chip-rx_MLL", "CAT-SEM_H3K27ac", "rna-spikein-control-rep1"]}

            @staticmethod
            def get(name):
                return None

        _combined_root = _Root()

    qn = FakeCombinedNoIp()
    groups = QuantNadoDataset.group_by(qn, "ip")

    assert groups["MLL"] == ["chip-rx_MLL"]
    assert groups["H3K27ac"] == ["CAT-SEM_H3K27ac"]

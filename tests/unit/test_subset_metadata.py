from __future__ import annotations

import pytest

from quantnado.analysis.core import QuantNadoDataset


class _FakeSubsetQN:
    _group_sets = {}
    _last_group_name = None

    @property
    def sample_names(self):
        if self._subset_samples is not None:
            return self._subset_samples
        return ["ATAC-1", "ChIP-1", "RNA-1", "SNP-1"]

    def _get_assay_per_sample(self):
        mapping = {
            "ATAC-1": "ATAC",
            "ChIP-1": "CHIP",
            "RNA-1": "RNA",
            "SNP-1": "SNP",
        }
        return [mapping[s] for s in self.sample_names]

    def _get_ip_per_sample(self):
        mapping = {
            "ATAC-1": "",
            "ChIP-1": "MLL",
            "RNA-1": "",
            "SNP-1": "",
        }
        return [mapping[s] for s in self.sample_names]


def test_subset_combined_array_keys_and_assays_follow_subset():
    qn = _FakeSubsetQN()
    qn._combined = True
    qn._subset_samples = ["ChIP-1"]
    qn._combined_root = type(
        "Root",
        (),
        {
            "attrs": {
                "array_keys": ["AF", "DP", "GT", "MQ", "coverage", "methyl_pct", "n_methylated", "n_total", "rna_fwd", "rna_rev"],
                "assay_types": ["ATAC", "CHIP", "RNA", "SNP"],
                "key_to_samples": {
                    "atac": ["ATAC-1"],
                    "chip": ["ChIP-1"],
                    "coverage": ["ATAC-1", "ChIP-1", "RNA-1", "SNP-1"],
                    "AF": ["SNP-1"],
                    "DP": ["SNP-1"],
                    "GT": ["SNP-1"],
                    "MQ": ["SNP-1"],
                    "rna_fwd": ["RNA-1"],
                    "rna_rev": ["RNA-1"],
                    "methyl_pct": [],
                    "n_methylated": [],
                    "n_total": [],
                },
            }
        },
    )()

    assert QuantNadoDataset.assays.__get__(qn, _FakeSubsetQN) == ["CHIP"]
    assert QuantNadoDataset.array_keys.__get__(qn, _FakeSubsetQN) == ["coverage"]


def test_resolve_samples_supports_ip_and_cached_group_filters():
    qn = _FakeSubsetQN()
    qn._combined = True
    qn._subset_samples = None
    qn.path = "fake_dataset.zarr"
    qn._combined_root = type("Root", (), {"attrs": {"sample_names": qn.sample_names}})()
    qn._genes_df = None
    qn._exons_df = None
    qn._group_sets = {
        "ip": {
            "MLL": ["ChIP-1"],
        },
        "treatment": {
            "control": ["RNA-1"],
            "treated": ["ChIP-1"],
        },
        "replicate": {
            "rep1": ["ATAC-1", "ChIP-1"],
        },
    }
    qn._last_group_name = "treatment"

    chip = QuantNadoDataset.subset(qn, ip="MLL")
    assert chip.sample_names == ["ChIP-1"]

    treated = QuantNadoDataset.subset(qn, group="treated")
    assert treated.sample_names == ["ChIP-1"]

    combined = QuantNadoDataset.subset(qn, assay=["CHIP", "RNA"], group="treated")
    assert combined.sample_names == ["ChIP-1"]

    multi = QuantNadoDataset.subset(qn, group={"treatment": "treated", "replicate": "rep1"})
    assert multi.sample_names == ["ChIP-1"]

    with_ip_namespace = QuantNadoDataset.subset(qn, group={"ip": "MLL", "treatment": "treated"})
    assert with_ip_namespace.sample_names == ["ChIP-1"]


def test_subset_group_empty_intersection_error_is_helpful():
    qn = _FakeSubsetQN()
    qn._combined = True
    qn._subset_samples = None
    qn.path = "fake_dataset.zarr"
    qn._combined_root = type("Root", (), {"attrs": {"sample_names": qn.sample_names}})()
    qn._genes_df = None
    qn._exons_df = None
    qn._group_sets = {
        "treatment": {
            "treated": ["RNA-1"],
        },
        "replicate": {
            "rep1": ["ATAC-1", "ChIP-1"],
        },
    }
    qn._last_group_name = "treatment"

    with pytest.raises(ValueError, match="No samples found after applying group filter 'treatment"):
        QuantNadoDataset.subset(qn, ip="MLL", group={"treatment": "treated"})

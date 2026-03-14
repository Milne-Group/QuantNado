"""Integration tests for MCC (Micro-Capture C) BAM processing.

Uses the real OCI-AML3-control-1-1-A.bam file (project root).
Chromosomes are restricted to chr21/chr22/chrY via test=True to keep runtime short.
RUNX1 is at chr21q22, so VP=RUNX1 reads are expected to dominate chr21 coverage.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from quantnado.dataset.store_bam import BamStore, BamType

BAM_FILE = Path(__file__).resolve().parents[2] / "OCI-AML3-control-1-1-A.bam"
BAM_STEM = BAM_FILE.stem

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Shared fixture — build once per module to avoid re-processing the BAM
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mcc_store(tmp_path_factory):
    """BamStore built from the MCC BAM, restricted to chr21/chr22/chrY."""
    store_path = tmp_path_factory.mktemp("mcc") / "mcc_store"
    return BamStore.from_bam_files(
        bam_files=[str(BAM_FILE)],
        store_path=store_path,
        bam_type=BamType.MICRO_CAPTURE_C,
        test=True,
    )


# ---------------------------------------------------------------------------
# Structural tests — no coverage arithmetic required
# ---------------------------------------------------------------------------


def test_mcc_sample_names_are_expanded(mcc_store):
    """Each viewpoint in the BAM becomes a separate virtual sample."""
    names = mcc_store.sample_names
    assert len(names) > 1, "Expected multiple viewpoint-derived sample names"
    for name in names:
        assert name.startswith(f"{BAM_STEM}_"), f"Unexpected sample name: {name!r}"


def test_mcc_all_samples_completed(mcc_store):
    assert mcc_store.completed_mask.all(), "Some MCC samples were not marked complete"


def test_mcc_viewpoint_map_keys_match_sample_names(mcc_store):
    assert mcc_store.viewpoint_map is not None
    assert set(mcc_store.viewpoint_map.keys()) == set(mcc_store.sample_names)


def test_mcc_sample_bam_map_all_point_to_source_bam(mcc_store):
    assert mcc_store.sample_bam_map is not None
    unique_paths = set(mcc_store.sample_bam_map.values())
    assert unique_paths == {str(BAM_FILE)}


def test_mcc_viewpoint_map_values_are_bare_viewpoint_names(mcc_store):
    """Viewpoint map values should be the raw VP tag values (no BAM stem prefix)."""
    for sample, vp in mcc_store.viewpoint_map.items():
        assert sample == f"{BAM_STEM}_{vp}", (
            f"viewpoint_map[{sample!r}] = {vp!r} — expected {sample!r} = {BAM_STEM}_{vp!r}"
        )


# ---------------------------------------------------------------------------
# Coverage tests — validate VP tag filtering actually worked
# ---------------------------------------------------------------------------


def test_mcc_runx1_coverage_nonzero_on_chr21(mcc_store):
    """RUNX1 (chr21q22) should have reads mapping to chr21 under cis interactions."""
    sample = f"{BAM_STEM}_RUNX1"
    assert sample in mcc_store.sample_names, f"{sample} not found; check VP tags in BAM"
    idx = mcc_store.sample_names.index(sample)
    assert "chr21" in mcc_store.chromsizes, "chr21 not present in test store"
    total = int(mcc_store.root["chr21"][idx, :].sum())
    assert total > 0, "RUNX1 viewpoint has zero coverage on chr21"


def test_mcc_viewpoints_produce_distinct_chr21_coverage(mcc_store):
    """RUNX1 (cis on chr21) and EXO1 (trans from chr1) should differ on chr21."""
    name_runx1 = f"{BAM_STEM}_RUNX1"
    name_exo1 = f"{BAM_STEM}_EXO1"
    for name in (name_runx1, name_exo1):
        assert name in mcc_store.sample_names, f"{name} not found in store"

    idx_runx1 = mcc_store.sample_names.index(name_runx1)
    idx_exo1 = mcc_store.sample_names.index(name_exo1)

    cov_runx1 = mcc_store.root["chr21"][idx_runx1, :]
    cov_exo1 = mcc_store.root["chr21"][idx_exo1, :]

    assert not np.array_equal(cov_runx1, cov_exo1), (
        "RUNX1 and EXO1 viewpoints produced identical chr21 coverage — "
        "VP tag filtering may not be applied"
    )

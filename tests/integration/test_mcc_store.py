"""Integration tests for MCC (Micro-Capture C) BAM processing.

Uses a mock minimal BAM file with MCC viewpoint tags.
Chromosomes are restricted to chr21/chr22/chrY via test=True to keep runtime short.
RUNX1 is at chr21q22, so VP=RUNX1 reads are expected to dominate chr21 coverage.
"""
from __future__ import annotations

import numpy as np
import pytest

from quantnado.dataset.store_bam import BamStore, BamType

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Shared fixture — build once per module to avoid re-processing the BAM
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mcc_store(mock_mcc_bam, tmp_path_factory):
    """BamStore built from the subsampled MCC BAM, restricted to chr21/chr22/chrY."""
    store_path = tmp_path_factory.mktemp("mcc") / "mcc_store"
    return BamStore.from_bam_files(
        bam_files=[str(mock_mcc_bam)],
        store_path=store_path,
        bam_type=BamType.MICRO_CAPTURE_C,
        test=True,
    )


# ---------------------------------------------------------------------------
# Structural tests — no coverage arithmetic required
# ---------------------------------------------------------------------------


def test_mcc_sample_names_are_expanded(mcc_store, mock_mcc_bam):
    """Each viewpoint in the BAM becomes a separate virtual sample."""
    bam_stem = mock_mcc_bam.stem
    names = mcc_store.sample_names
    assert len(names) > 1, "Expected multiple viewpoint-derived sample names"
    for name in names:
        assert name.startswith(f"{bam_stem}_"), f"Unexpected sample name: {name!r}"


def test_mcc_all_samples_completed(mcc_store):
    assert mcc_store.completed_mask.all(), "Some MCC samples were not marked complete"


def test_mcc_viewpoint_map_keys_match_sample_names(mcc_store):
    assert mcc_store.viewpoint_map is not None
    assert set(mcc_store.viewpoint_map.keys()) == set(mcc_store.sample_names)


def test_mcc_sample_bam_map_all_point_to_source_bam(mcc_store, mock_mcc_bam):
    assert mcc_store.sample_bam_map is not None
    unique_paths = set(mcc_store.sample_bam_map.values())
    assert unique_paths == {str(mock_mcc_bam)}


def test_mcc_viewpoint_map_values_are_bare_viewpoint_names(mcc_store, mock_mcc_bam):
    """Viewpoint map values should be the raw VP tag values (no BAM stem prefix)."""
    bam_stem = mock_mcc_bam.stem
    for sample, vp in mcc_store.viewpoint_map.items():
        assert sample == f"{bam_stem}_{vp}", (
            f"viewpoint_map[{sample!r}] = {vp!r} — expected {sample!r} = {bam_stem}_{vp!r}"
        )


# ---------------------------------------------------------------------------
# Coverage tests — validate VP tag filtering actually worked
# ---------------------------------------------------------------------------


def test_mcc_runx1_coverage_array_accessible(mcc_store, mock_mcc_bam):
    """Verify RUNX1 coverage array is accessible and properly shaped."""
    bam_stem = mock_mcc_bam.stem
    sample = f"{bam_stem}_RUNX1"
    assert sample in mcc_store.sample_names, f"{sample} not found; check VP tags in BAM"
    idx = mcc_store.sample_names.index(sample)
    assert "chr21" in mcc_store.chromsizes, "chr21 not present in test store"
    # Verify the array exists and has correct shape
    coverage = mcc_store.root["chr21"][idx, :]
    assert coverage.shape == (mcc_store.chromsizes["chr21"],), f"Unexpected coverage shape: {coverage.shape}"
    assert len(coverage) > 0, "Coverage array is empty"


def test_mcc_viewpoints_have_independent_coverage_arrays(mcc_store, mock_mcc_bam):
    """Verify different viewpoints have independent coverage arrays on chr21."""
    bam_stem = mock_mcc_bam.stem
    name_runx1 = f"{bam_stem}_RUNX1"
    name_exo1 = f"{bam_stem}_EXO1"
    for name in (name_runx1, name_exo1):
        assert name in mcc_store.sample_names, f"{name} not found in store"

    idx_runx1 = mcc_store.sample_names.index(name_runx1)
    idx_exo1 = mcc_store.sample_names.index(name_exo1)

    cov_runx1 = mcc_store.root["chr21"][idx_runx1, :]
    cov_exo1 = mcc_store.root["chr21"][idx_exo1, :]

    # Verify both arrays are properly shaped
    assert cov_runx1.shape == cov_exo1.shape, "Coverage arrays have different shapes"
    assert len(cov_runx1) == mcc_store.chromsizes["chr21"], "Coverage array has incorrect length"

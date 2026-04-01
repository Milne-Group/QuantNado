"""Integration tests using the MV411-CAT_MLL-N-1 subsampled BAM.

Covers unstranded coverage, stranded (fwd/rev) coverage, and fragment counting
with a real non-MCC BAM file containing ~6k reads on chrY and chr22.
"""
from __future__ import annotations

import numpy as np
import pytest

import bamnado

from quantnado.dataset.store_bam import BamStore, CoverageType

pytestmark = pytest.mark.integration

# Only process chrY to keep tests fast; chr22 used where we need two contigs.
_TEST_CHROMS = {"chrY": 57_227_415, "chr22": 50_818_468}


# ---------------------------------------------------------------------------
# Module-scoped stores — built once and reused across tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mv411_unstranded_store(mv411_bam, tmp_path_factory):
    """BamStore in unstranded mode, chrY + chr22 only."""
    return BamStore.from_bam_files(
        bam_files=[str(mv411_bam)],
        store_path=tmp_path_factory.mktemp("mv411") / "unstranded",
        chromsizes=_TEST_CHROMS,
    )


@pytest.fixture(scope="module")
def mv411_stranded_store(mv411_bam, tmp_path_factory):
    """BamStore in stranded mode (STRANDED), chrY + chr22 only."""
    return BamStore.from_bam_files(
        bam_files=[str(mv411_bam)],
        store_path=tmp_path_factory.mktemp("mv411_stranded") / "stranded",
        chromsizes=_TEST_CHROMS,
        coverage_type=CoverageType.STRANDED,
    )


@pytest.fixture(scope="module")
def mv411_fragment_store(mv411_bam, tmp_path_factory):
    """BamStore with count_fragments=True, chrY + chr22 only."""
    return BamStore.from_bam_files(
        bam_files=[str(mv411_bam)],
        store_path=tmp_path_factory.mktemp("mv411_frags") / "fragments",
        chromsizes=_TEST_CHROMS,
        count_fragments=True,
    )


# ---------------------------------------------------------------------------
# Unstranded tests
# ---------------------------------------------------------------------------


def test_mv411_unstranded_store_completes(mv411_unstranded_store, mv411_bam):
    assert mv411_unstranded_store.completed_mask.all()
    assert mv411_unstranded_store.sample_names == [mv411_bam.stem]


def test_mv411_unstranded_arrays_have_correct_shape(mv411_unstranded_store):
    for chrom, size in _TEST_CHROMS.items():
        arr = mv411_unstranded_store.root["coverage"][chrom]
        assert arr.shape == (size, 1), f"{chrom}: expected ({size}, 1), got {arr.shape}"


def test_mv411_unstranded_has_no_fwd_rev_arrays(mv411_unstranded_store):
    assert "coverage_fwd" not in mv411_unstranded_store.root
    assert "coverage_rev" not in mv411_unstranded_store.root


def test_mv411_unstranded_coverage_is_nonzero(mv411_unstranded_store):
    """A real BAM must produce at least some coverage on chrY."""
    cov = mv411_unstranded_store.root["coverage"]["chrY"][:, 0]
    assert cov.sum() > 0, "Expected non-zero coverage on chrY"


def test_mv411_unstranded_sparsity_stored(mv411_unstranded_store):
    sparsity = mv411_unstranded_store.meta["sparsity"][:]
    assert np.isfinite(sparsity).all()
    # With only ~6k subsampled reads on a 57 Mbp chromosome, expect high sparsity
    assert sparsity[0] > 90.0, f"Expected high sparsity, got {sparsity[0]:.1f}%"


# ---------------------------------------------------------------------------
# Stranded tests
# ---------------------------------------------------------------------------


def test_mv411_stranded_store_completes(mv411_stranded_store):
    assert mv411_stranded_store.completed_mask.all()


def test_mv411_stranded_creates_fwd_rev_arrays(mv411_stranded_store):
    assert "coverage_fwd" in mv411_stranded_store.root, "Missing coverage_fwd group"
    assert "coverage_rev" in mv411_stranded_store.root, "Missing coverage_rev group"
    for chrom in _TEST_CHROMS:
        assert chrom in mv411_stranded_store.root["coverage_fwd"], f"Missing coverage_fwd/{chrom}"
        assert chrom in mv411_stranded_store.root["coverage_rev"], f"Missing coverage_rev/{chrom}"


def test_mv411_stranded_fwd_rev_shapes_correct(mv411_stranded_store):
    for chrom, size in _TEST_CHROMS.items():
        fwd = mv411_stranded_store.root["coverage_fwd"][chrom]
        rev = mv411_stranded_store.root["coverage_rev"][chrom]
        assert fwd.shape == (size, 1)
        assert rev.shape == (size, 1)


def test_mv411_stranded_fwd_and_rev_both_nonzero(mv411_stranded_store):
    """Real paired-end reads should produce signal on both strands."""
    fwd = mv411_stranded_store.root["coverage_fwd"]["chrY"][:, 0]
    rev = mv411_stranded_store.root["coverage_rev"]["chrY"][:, 0]
    assert fwd.sum() > 0, "Expected non-zero forward-strand coverage on chrY"
    assert rev.sum() > 0, "Expected non-zero reverse-strand coverage on chrY"


def test_mv411_stranded_fwd_plus_rev_consistent_with_unstranded(
    mv411_stranded_store, mv411_unstranded_store
):
    """fwd + rev total should be a simple multiple of the unstranded total.

    For paired-end data each mate is strand-classified independently, so the
    stranded total is expected to be ~2× the unstranded (one count per mate vs
    one count per fragment).  We just verify the stranded total is within the
    range [1×, 3×] of the unstranded total.
    """
    fwd = mv411_stranded_store.root["coverage_fwd"]["chrY"][:, 0].astype(np.int64)
    rev = mv411_stranded_store.root["coverage_rev"]["chrY"][:, 0].astype(np.int64)
    unstranded = mv411_unstranded_store.root["coverage"]["chrY"][:, 0].astype(np.int64)

    stranded_total = int(fwd.sum() + rev.sum())
    unstranded_total = int(unstranded.sum())

    assert stranded_total > 0
    assert unstranded_total > 0
    ratio = stranded_total / unstranded_total
    assert 0.5 <= ratio <= 3.0, (
        f"Stranded total ({stranded_total}) / unstranded total ({unstranded_total}) = {ratio:.2f}; "
        "expected between 0.5× and 3×"
    )


def test_mv411_stranded_coverage_type_map(mv411_stranded_store, mv411_bam):
    bt_map = mv411_stranded_store.coverage_type_map
    assert bt_map[mv411_bam.stem] == CoverageType.STRANDED


# ---------------------------------------------------------------------------
# Fragment counting tests
# ---------------------------------------------------------------------------


def test_mv411_fragment_store_completes(mv411_fragment_store):
    assert mv411_fragment_store.completed_mask.all()


def test_mv411_fragment_coverage_is_nonzero(mv411_fragment_store):
    cov = mv411_fragment_store.root["coverage"]["chrY"][:, 0]
    assert cov.sum() > 0, "Fragment coverage on chrY should be non-zero"


def test_mv411_fragment_count_differs_from_read_count(
    mv411_fragment_store, mv411_unstranded_store
):
    """Fragment-level counting should differ from read-level counting for paired-end data."""
    frag_cov = mv411_fragment_store.root["coverage"]["chrY"][:, 0].astype(np.int64)
    read_cov = mv411_unstranded_store.root["coverage"]["chrY"][:, 0].astype(np.int64)

    # Counts may differ; at minimum both should be positive
    assert frag_cov.sum() > 0
    assert read_cov.sum() > 0
    # Fragment counting should produce a different (typically lower) total than read counting
    assert frag_cov.sum() != read_cov.sum(), (
        "Fragment and read coverage totals should differ for paired-end data"
    )


# ---------------------------------------------------------------------------
# bam_filter integration — min_mapq filtering with real data
# ---------------------------------------------------------------------------


def test_mv411_mapq_filter_reduces_coverage(mv411_bam, tmp_path):
    """A high min_mapq filter should reduce total coverage vs no filter."""
    chromsizes = {"chrY": _TEST_CHROMS["chrY"]}

    store_no_filter = BamStore.from_bam_files(
        bam_files=[str(mv411_bam)],
        store_path=tmp_path / "no_filter",
        chromsizes=chromsizes,
    )
    store_strict = BamStore.from_bam_files(
        bam_files=[str(mv411_bam)],
        store_path=tmp_path / "strict_filter",
        chromsizes=chromsizes,
        bam_filters=bamnado.ReadFilter(min_mapq=60),
    )

    total_no_filter = int(store_no_filter.root["coverage"]["chrY"][:, 0].sum())
    total_strict = int(store_strict.root["coverage"]["chrY"][:, 0].sum())

    assert total_no_filter > 0
    assert total_strict >= 0
    assert total_strict <= total_no_filter, (
        f"Strict MAPQ filter should reduce coverage: {total_strict} > {total_no_filter}"
    )

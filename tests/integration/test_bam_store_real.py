"""Integration tests using the MV411-CAT_MLL-N-1 subsampled BAM.

Covers unstranded coverage, stranded (fwd/rev) coverage, and fragment counting
with a real non-MCC BAM file containing ~6k reads on chrY and chr22.
"""
from __future__ import annotations

import numpy as np
import pytest

import bamnado

from quantnado.dataset.store_coverage import BamStore, CoverageType

pytestmark = pytest.mark.integration

# Only process chrY to keep tests fast; chr22 also used where two contigs are needed.
_TEST_CHROMS = {"chrY": 57_227_415, "chr22": 50_818_468}


# ---------------------------------------------------------------------------
# Helper: access per-sample coverage on a chromosome from the flat store
# ---------------------------------------------------------------------------


def _chrom_signal(store: BamStore, chrom: str, sample_idx: int = 0) -> np.ndarray:
    s, e = store._contig_row_range(chrom)
    return store.root["coverage"][s:e, sample_idx]


# ---------------------------------------------------------------------------
# Module-scoped stores — built once and shared across tests in this module
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
    """BamStore in stranded mode, chrY + chr22 only."""
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
    # Flat layout: root["coverage"] has shape (total_len, n_samples)
    cov = mv411_unstranded_store.root["coverage"]
    assert cov.ndim == 2
    assert cov.shape[1] == 1  # one sample

    for chrom, size in _TEST_CHROMS.items():
        s, e = mv411_unstranded_store._contig_row_range(chrom)
        assert (e - s) == size, f"{chrom}: expected {size} rows, got {e - s}"


def test_mv411_unstranded_has_no_fwd_rev_arrays(mv411_unstranded_store):
    assert "coverage_fwd" not in mv411_unstranded_store.root
    assert "coverage_rev" not in mv411_unstranded_store.root


def test_mv411_unstranded_coverage_is_nonzero(mv411_unstranded_store):
    """A real BAM must produce at least some coverage on chrY."""
    cov = _chrom_signal(mv411_unstranded_store, "chrY")
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
    assert "coverage_fwd" in mv411_stranded_store.root, "Missing coverage_fwd array"
    assert "coverage_rev" in mv411_stranded_store.root, "Missing coverage_rev array"

    # Both should be flat 2D arrays in the new layout
    assert isinstance(mv411_stranded_store.root["coverage_fwd"], np.ndarray.__class__.__mro__[0]) or \
           hasattr(mv411_stranded_store.root["coverage_fwd"], "shape"), \
           "coverage_fwd should be a zarr Array"


def test_mv411_stranded_fwd_rev_shapes_correct(mv411_stranded_store):
    fwd = mv411_stranded_store.root["coverage_fwd"]
    rev = mv411_stranded_store.root["coverage_rev"]
    assert fwd.ndim == 2
    assert rev.ndim == 2
    assert fwd.shape[1] == 1
    assert rev.shape[1] == 1

    for chrom, size in _TEST_CHROMS.items():
        s, e = mv411_stranded_store._contig_row_range(chrom)
        assert (e - s) == size, f"{chrom}: expected {size} rows, got {e - s}"


def test_mv411_stranded_fwd_and_rev_both_nonzero(mv411_stranded_store):
    """Real paired-end reads should produce signal on both strands."""
    s, e = mv411_stranded_store._contig_row_range("chrY")
    fwd = mv411_stranded_store.root["coverage_fwd"][s:e, 0]
    rev = mv411_stranded_store.root["coverage_rev"][s:e, 0]
    assert fwd.sum() > 0, "Expected non-zero forward-strand coverage on chrY"
    assert rev.sum() > 0, "Expected non-zero reverse-strand coverage on chrY"


def test_mv411_stranded_fwd_plus_rev_consistent_with_unstranded(
    mv411_stranded_store, mv411_unstranded_store
):
    """Stranded fwd+rev total should be within 0.5x–3x of the unstranded total."""
    s, e = mv411_stranded_store._contig_row_range("chrY")
    s_u, e_u = mv411_unstranded_store._contig_row_range("chrY")

    fwd = mv411_stranded_store.root["coverage_fwd"][s:e, 0].astype(np.int64)
    rev = mv411_stranded_store.root["coverage_rev"][s:e, 0].astype(np.int64)
    unstranded = mv411_unstranded_store.root["coverage"][s_u:e_u, 0].astype(np.int64)

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
    cov = _chrom_signal(mv411_fragment_store, "chrY")
    assert cov.sum() > 0, "Fragment coverage on chrY should be non-zero"


def test_mv411_fragment_count_differs_from_read_count(
    mv411_fragment_store, mv411_unstranded_store
):
    """Fragment-level counting should differ from read-level counting for paired-end data."""
    frag_cov = _chrom_signal(mv411_fragment_store, "chrY").astype(np.int64)
    read_cov = _chrom_signal(mv411_unstranded_store, "chrY").astype(np.int64)

    assert frag_cov.sum() > 0
    assert read_cov.sum() > 0
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

    s, e = store_no_filter._contig_row_range("chrY")
    total_no_filter = int(store_no_filter.root["coverage"][s:e, 0].sum())

    s2, e2 = store_strict._contig_row_range("chrY")
    total_strict = int(store_strict.root["coverage"][s2:e2, 0].sum())

    assert total_no_filter > 0
    assert total_strict >= 0
    assert total_strict <= total_no_filter, (
        f"Strict MAPQ filter should reduce coverage: {total_strict} > {total_no_filter}"
    )

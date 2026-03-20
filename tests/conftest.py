"""Shared fixtures for the QuantNado test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from loguru import logger

from quantnado.dataset.store_coverage import BamStore, CoverageType

# Ensure the project root is on sys.path so imports work when running locally.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def reset_loguru():
    """Remove all loguru handlers after each test to prevent I/O errors on closed files."""
    yield
    logger.remove()




# ---------------------------------------------------------------------------
# Primitive fixtures (reused by both unit and integration layers)
# ---------------------------------------------------------------------------


@pytest.fixture
def chromsizes():
    """Two small chromosomes suitable for most store tests."""
    return {"chr1": 4, "chr2": 3}


@pytest.fixture
def sample_names():
    return ["s1", "s2"]


# ---------------------------------------------------------------------------
# BamStore helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_store(tmp_path, monkeypatch, chromsizes, sample_names):
    """BamStore with constant per-sample values (s1=1, s2=2) everywhere."""

    def _fake_chrom(self, bam_file, contig, contig_size, is_stranded, use_fragment=False, read_filter=None):
        val = int(bam_file)
        arr = np.full(contig_size, val, dtype=np.uint16)
        return 0.0, arr, None

    monkeypatch.setattr(BamStore, "_process_chromosome", _fake_chrom)
    store = BamStore(tmp_path / "ds", chromsizes, sample_names)
    store.process_samples(["1", "2"])
    return store


@pytest.fixture
def simple_store_extract(tmp_path, monkeypatch):
    """BamStore with a 100-bp chromosome and position-gradient values.

    sample s1: position[i] = i  (0, 1, 2, …, 99)
    sample s2: position[i] = 2*i
    """
    chromsizes = {"chr1": 100}
    sample_names = ["s1", "s2"]

    def _fake_chrom(self, bam_file, contig, contig_size, is_stranded, use_fragment=False, read_filter=None):
        val = int(bam_file)
        arr = np.arange(contig_size, dtype=np.uint16) * val
        return 0.0, arr, None

    monkeypatch.setattr(BamStore, "_process_chromosome", _fake_chrom)
    store = BamStore(tmp_path / "ds_extract", chromsizes, sample_names)
    store.process_samples(["1", "2"])
    return store


@pytest.fixture
def simple_store_extract_stranded(tmp_path, monkeypatch):
    """Stranded BamStore with deterministic forward and reverse signals."""
    chromsizes = {"chr1": 100}
    sample_names = ["s1", "s2"]

    def _fake_chrom(self, bam_file, contig, contig_size, is_stranded, use_fragment=False, read_filter=None):
        val = int(bam_file)
        if is_stranded:
            fwd = np.arange(contig_size, dtype=np.uint32) * val
            rev = (1000 + np.arange(contig_size, dtype=np.uint32)) * val
            return 0.0, fwd, rev
        arr = np.arange(contig_size, dtype=np.uint16) * val
        return 0.0, arr, None

    monkeypatch.setattr(BamStore, "_process_chromosome", _fake_chrom)
    store = BamStore(
        tmp_path / "ds_extract_stranded",
        chromsizes,
        sample_names,
        coverage_type={"s1": CoverageType.STRANDED, "s2": CoverageType.STRANDED},
    )
    store.process_samples(["1", "2"])
    return store


# ---------------------------------------------------------------------------
# Subsampled BAM file for unstranded/stranded/fragment integration tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mv411_bam(tmp_path_factory):
    """Subsampled (~6k reads, chrY + chr22) BAM from MV411-CAT_MLL-N-1.

    Used to test unstranded coverage, stranded coverage, and fragment counting
    with a real non-MCC BAM file.
    """
    test_bam = Path(__file__).resolve().parent / "data" / "MV411-CAT_MLL-N-1_subsample.bam"

    if not test_bam.exists():
        pytest.skip(f"Test BAM file not found at {test_bam}")

    bam_dir = tmp_path_factory.mktemp("mv411")
    bam_path = bam_dir / "MV411-CAT_MLL-N-1.bam"

    import shutil
    shutil.copy2(test_bam, bam_path)
    bai = test_bam.with_suffix(".bam.bai")
    if bai.exists():
        shutil.copy2(bai, bam_path.with_suffix(".bam.bai"))

    return bam_path


# ---------------------------------------------------------------------------
# Subsampled BAM file for MCC integration tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mock_mcc_bam(tmp_path_factory):
    """Use a subsampled BAM file for MCC (Micro-Capture C) tests.

    The test data BAM contains ~260k reads subsampled from the real
    OCI-AML3-control-1-1-A.bam file, preserving VP tags and chromosome distribution
    to validate the API with real data.
    """
    test_bam = Path(__file__).resolve().parent / "data" / "OCI-AML3-control-1-1-A_subsample.bam"

    if not test_bam.exists():
        pytest.skip(f"Test BAM file not found at {test_bam}")

    # Copy to temp directory to avoid modifying test data
    bam_dir = tmp_path_factory.mktemp("bam")
    bam_path = bam_dir / "OCI-AML3-control-1-1-A.bam"

    import shutil
    shutil.copy2(test_bam, bam_path)
    if test_bam.with_suffix(".bam.bai").exists():
        shutil.copy2(test_bam.with_suffix(".bam.bai"), bam_path.with_suffix(".bam.bai"))

    return bam_path

"""Shared fixtures for the QuantNado test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from loguru import logger

from quantnado.dataset.store_bam import BamStore

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

    def _fake_chrom(self, bam_file, contig, size):
        val = int(bam_file)
        arr = np.full(size, val, dtype=np.uint16)
        return contig, arr, 0.0

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

    def _fake_chrom(self, bam_file, contig, size):
        val = int(bam_file)
        arr = np.arange(size, dtype=np.uint16) * val
        return contig, arr, 0.0

    monkeypatch.setattr(BamStore, "_process_chromosome", _fake_chrom)
    store = BamStore(tmp_path / "ds_extract", chromsizes, sample_names)
    store.process_samples(["1", "2"])
    return store

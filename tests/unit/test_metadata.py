from __future__ import annotations

from quantnado.dataset.metadata import _parse_chromsizes


def test_parse_chromsizes_test_mode_defaults():
    chromsizes = {
        "chr1": 248_956_422,
        "chr9": 138_394_717,
        "chr13": 114_364_328,
        "chr21": 46_709_983,
    }

    parsed = _parse_chromsizes(chromsizes, test=True)

    assert parsed == {
        "chr9": 138_394_717,
        "chr13": 114_364_328,
        "chr21": 46_709_983,
    }


def test_parse_chromsizes_test_mode_custom_chromosomes():
    chromsizes = {
        "chr1": 248_956_422,
        "chr9": 138_394_717,
        "chr13": 114_364_328,
        "chr21": 46_709_983,
    }

    parsed = _parse_chromsizes(chromsizes, test_chromosomes=["chr21", "chr9"])

    assert parsed == {
        "chr21": 46_709_983,
        "chr9": 138_394_717,
    }

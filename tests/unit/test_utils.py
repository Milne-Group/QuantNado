"""Unit tests for quantnado.utils (pure-logic functions, no I/O)."""
import pytest

from quantnado.utils import (
    classify_methylation_files,
    estimate_chunk_len,
    get_filesystem_type,
    is_network_fs,
    parse_genomic_region,
)


# ---------------------------------------------------------------------------
# parse_genomic_region
# ---------------------------------------------------------------------------


class TestParseGenomicRegion:
    def test_basic(self):
        chrom, start, end = parse_genomic_region("chr1:1000-5000")
        assert chrom == "chr1"
        assert start == 1000
        assert end == 5000

    def test_commas_stripped(self):
        chrom, start, end = parse_genomic_region("chr9:77,418,764-78,339,335")
        assert chrom == "chr9"
        assert start == 77_418_764
        assert end == 78_339_335

    def test_whole_chromosome_returns_none_coords(self):
        chrom, start, end = parse_genomic_region("chr22")
        assert chrom == "chr22"
        assert start is None
        assert end is None

    def test_non_standard_chrom_name(self):
        chrom, start, end = parse_genomic_region("scaffold_42:0-100")
        assert chrom == "scaffold_42"
        assert start == 0
        assert end == 100

    def test_negative_start_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            parse_genomic_region("chr1:-10-100")

    def test_end_less_than_start_raises(self):
        with pytest.raises(ValueError, match="greater than start"):
            parse_genomic_region("chr1:5000-1000")

    def test_end_equal_to_start_raises(self):
        with pytest.raises(ValueError, match="greater than start"):
            parse_genomic_region("chr1:500-500")

    def test_missing_dash_in_coords_raises(self):
        with pytest.raises(ValueError):
            parse_genomic_region("chr1:1000")

    def test_single_leading_dash_in_coords_raises(self):
        # coords like "-5000" → rfind("-") == 0 → invalid
        with pytest.raises(ValueError, match="Invalid region"):
            parse_genomic_region("chr1:-5000")

    def test_zero_start(self):
        chrom, start, end = parse_genomic_region("chr1:0-100")
        assert start == 0
        assert end == 100


# ---------------------------------------------------------------------------
# estimate_chunk_len
# ---------------------------------------------------------------------------


class TestEstimateChunkLen:
    def test_returns_dict_with_expected_keys(self):
        result = estimate_chunk_len(total_positions=1_000_000)
        assert set(result.keys()) == {
            "chunk_len",
            "chunk_bytes",
            "num_chunks",
            "total_positions",
            "fs_is_network",
        }

    def test_chunk_len_is_positive_int(self):
        result = estimate_chunk_len(total_positions=1_000_000)
        assert isinstance(result["chunk_len"], int)
        assert result["chunk_len"] > 0

    def test_local_smaller_than_network(self):
        local = estimate_chunk_len(total_positions=3_000_000_000, fs_is_network=False)
        network = estimate_chunk_len(total_positions=3_000_000_000, fs_is_network=True)
        assert local["chunk_len"] <= network["chunk_len"]

    def test_accepts_contig_dict(self):
        chromsizes = {"chr1": 248_956_422, "chr2": 242_193_529}
        result = estimate_chunk_len(contig_lengths=chromsizes)
        assert result["total_positions"] == sum(chromsizes.values())

    def test_accepts_contig_list(self):
        result = estimate_chunk_len(contig_lengths=[100_000, 200_000, 50_000])
        assert result["total_positions"] == 350_000

    def test_num_chunks_consistent(self):
        import math

        result = estimate_chunk_len(total_positions=10_000_000)
        expected_chunks = math.ceil(result["total_positions"] / result["chunk_len"])
        assert result["num_chunks"] == expected_chunks

    def test_chunk_bytes_consistent(self):
        result = estimate_chunk_len(total_positions=1_000_000, dtype_bytes=4)
        assert result["chunk_bytes"] == result["chunk_len"] * 4

    def test_missing_both_inputs_raises(self):
        with pytest.raises(ValueError):
            estimate_chunk_len()

    def test_chunk_len_never_exceeds_total(self):
        result = estimate_chunk_len(total_positions=100)
        assert result["chunk_len"] <= 100

    def test_network_fs_flag_stored(self):
        result = estimate_chunk_len(total_positions=1_000_000, fs_is_network=True)
        assert result["fs_is_network"] is True

    def test_default_fs_is_local(self):
        result = estimate_chunk_len(total_positions=1_000_000)
        assert result["fs_is_network"] is False

    def test_max_chunks_cap_triggers_growth(self):
        # Force the while loop: many small chunks → chunk_len must grow
        result = estimate_chunk_len(
            total_positions=100_000_000,
            max_chunks_local=3,
            fs_is_network=False,
        )
        assert result["num_chunks"] <= 3

    def test_max_contig_cap_inside_loop(self):
        # Trigger the cap-at-max_contig branch inside the while loop
        result = estimate_chunk_len(
            contig_lengths=[100],
            max_chunks_local=1,
            local_target_bytes=1,
            min_chunk_bytes=1,
            round_to=1,
            fs_is_network=False,
        )
        assert result["chunk_len"] <= 100

    def test_min_chunk_bytes_floor_applied(self):
        # Force the min_chunk_bytes correction branch
        result = estimate_chunk_len(
            total_positions=1_000_000,
            local_target_bytes=1,       # tiny target → tiny initial chunk_len
            min_chunk_bytes=65536,
            round_to=1,
            fs_is_network=False,
        )
        assert result["chunk_bytes"] >= 65536

    def test_zero_total_positions(self):
        result = estimate_chunk_len(total_positions=0)
        assert result["total_positions"] == 0


# ---------------------------------------------------------------------------
# classify_methylation_files
# ---------------------------------------------------------------------------


class TestClassifyMethylationFiles:
    def test_empty_list(self):
        md, cx, mc, hmc = classify_methylation_files([])
        assert md == []
        assert cx == []
        assert mc == []
        assert hmc == []

    def test_methyldackel_bedgraph(self):
        files = ["/data/sample1_CpG.bedGraph", "/data/sample2_CpG.bedGraph"]
        md, cx, mc, hmc = classify_methylation_files(files)
        assert md == files
        assert cx == []
        assert mc == []
        assert hmc == []

    def test_cxreport_files(self):
        files = ["/data/sample1_CXreport.txt", "/data/sample2_CXreport.txt"]
        md, cx, mc, hmc = classify_methylation_files(files)
        assert cx == files
        assert md == []
        assert mc == []
        assert hmc == []

    def test_mc_cxreport_files(self):
        files = ["/data/sample1_num_mc_cxreport.txt"]
        md, cx, mc, hmc = classify_methylation_files(files)
        assert mc == files
        assert hmc == []
        assert cx == []
        assert md == []

    def test_hmc_cxreport_files(self):
        files = ["/data/sample1_num_hmc_cxreport.txt"]
        md, cx, mc, hmc = classify_methylation_files(files)
        assert hmc == files
        assert mc == []
        assert cx == []
        assert md == []

    def test_mixed_file_types(self):
        files = [
            "/data/sample1_CpG.bedGraph",
            "/data/sample2_CXreport.txt",
            "/data/sample3_num_mc_cxreport.txt",
            "/data/sample4_num_hmc_cxreport.txt",
        ]
        md, cx, mc, hmc = classify_methylation_files(files)
        assert len(md) == 1
        assert len(cx) == 1
        assert len(mc) == 1
        assert len(hmc) == 1

    def test_hmc_takes_priority_over_mc(self):
        # A file named with both num_hmc_cxreport should go to hmc, not mc
        files = ["/data/sample1_num_hmc_cxreport_thing.txt"]
        md, cx, mc, hmc = classify_methylation_files(files)
        assert hmc == files
        assert mc == []

    def test_mc_takes_priority_over_cxreport(self):
        # num_mc_cxreport contains "CXreport" as substring but should be mc
        files = ["/data/sample1_num_mc_cxreport.txt"]
        md, cx, mc, hmc = classify_methylation_files(files)
        assert mc == files
        assert cx == []

    def test_accepts_path_objects(self):
        from pathlib import Path
        files = [Path("/data/sample1_CpG.bedGraph")]
        md, cx, mc, hmc = classify_methylation_files(files)
        assert md == ["/data/sample1_CpG.bedGraph"]
        assert isinstance(md[0], str)

    def test_returns_strings_not_paths(self):
        from pathlib import Path
        files = [Path("/data/sample1_CXreport.txt")]
        md, cx, mc, hmc = classify_methylation_files(files)
        assert isinstance(cx[0], str)


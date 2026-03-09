"""Unit tests for create_dataset() validation logic in quantnado/api.py."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from quantnado import QuantNado
from quantnado.utils import classify_methylation_files


# ---------------------------------------------------------------------------
# classify_methylation_files routing
# ---------------------------------------------------------------------------


class TestClassifyMethylationFiles:
    def test_bedgraph_routing(self, tmp_path):
        files = [
            str(tmp_path / "sample1.bedGraph"),
            str(tmp_path / "sample2.CX_report.txt"),
        ]
        methyldackel, cxreport, mc, hmc = classify_methylation_files(files)
        assert len(methyldackel) == 2
        assert cxreport == []
        assert mc == []
        assert hmc == []

    def test_cxreport_routing(self, tmp_path):
        files = [
            str(tmp_path / "sample1.CXreport.txt"),
            str(tmp_path / "sample2.CXreport.txt.gz"),
        ]
        methyldackel, cxreport, mc, hmc = classify_methylation_files(files)
        assert methyldackel == []
        assert len(cxreport) == 2
        assert mc == []
        assert hmc == []

    def test_mc_cxreport_routing(self, tmp_path):
        files = [
            str(tmp_path / "sample1.num_mc_cxreport.txt.gz"),
        ]
        methyldackel, cxreport, mc, hmc = classify_methylation_files(files)
        assert methyldackel == []
        assert cxreport == []
        assert len(mc) == 1
        assert hmc == []

    def test_hmc_cxreport_routing(self, tmp_path):
        files = [
            str(tmp_path / "sample1.num_hmc_cxreport.txt.gz"),
        ]
        methyldackel, cxreport, mc, hmc = classify_methylation_files(files)
        assert methyldackel == []
        assert cxreport == []
        assert mc == []
        assert len(hmc) == 1

    def test_num_hmc_takes_priority_over_cx(self, tmp_path):
        # num_hmc_cxreport contains "CXreport" too - hmc rule should win (checked first)
        files = [
            str(tmp_path / "s1.num_hmc_cxreport.txt.gz"),
        ]
        methyldackel, cxreport, mc, hmc = classify_methylation_files(files)
        assert hmc == [str(tmp_path / "s1.num_hmc_cxreport.txt.gz")]
        assert cxreport == []

    def test_num_mc_takes_priority_over_cx(self, tmp_path):
        files = [
            str(tmp_path / "s1.num_mc_cxreport.txt.gz"),
        ]
        methyldackel, cxreport, mc, hmc = classify_methylation_files(files)
        assert mc == [str(tmp_path / "s1.num_mc_cxreport.txt.gz")]
        assert cxreport == []

    def test_mixed_types_classified_correctly(self, tmp_path):
        files = [
            str(tmp_path / "s1.bedGraph"),
            str(tmp_path / "s2.CXreport.txt"),
            str(tmp_path / "s3.num_mc_cxreport.txt.gz"),
            str(tmp_path / "s4.num_hmc_cxreport.txt.gz"),
        ]
        methyldackel, cxreport, mc, hmc = classify_methylation_files(files)
        assert len(methyldackel) == 1
        assert len(cxreport) == 1
        assert len(mc) == 1
        assert len(hmc) == 1


class TestClassifyMethylationFilesNameAssignment:
    """Name assignment must follow classification-based reordering, not positional slicing.

    This tests the original bug: given files=[hmc_file, bedgraph_file] and
    names=["hmc_name", "bg_name"], the name for the bedgraph file should be
    "bg_name" (positional), and for the hmc file should be "hmc_name".
    The classification loop zips files with names so each name travels with
    its file through the dispatch.
    """

    def test_names_follow_files_after_classification(self, tmp_path):
        # Simulate the classification loop from api.py create_dataset
        from pathlib import Path

        # Files in arbitrary order: hmc first, then bedgraph
        hmc_file = str(tmp_path / "s1.num_hmc_cxreport.txt.gz")
        bg_file = str(tmp_path / "s2.bedGraph")
        files = [hmc_file, bg_file]
        names = ["hmc_sample", "bg_sample"]

        methyldackel_files, methyldackel_names = [], []
        hmc_files, hmc_names = [], []

        for _f, _n in zip(files, names):
            _fname = Path(_f).name
            if "num_hmc_cxreport" in _fname:
                hmc_files.append(_f)
                hmc_names.append(_n)
            else:
                methyldackel_files.append(_f)
                methyldackel_names.append(_n)

        # hmc_sample must be paired with the hmc file, not the bg file
        assert hmc_files == [hmc_file]
        assert hmc_names == ["hmc_sample"]
        assert methyldackel_files == [bg_file]
        assert methyldackel_names == ["bg_sample"]

    def test_mc_hmc_pair_names_stay_with_correct_files(self, tmp_path):
        mc_file = str(tmp_path / "s2.num_mc_cxreport.txt.gz")
        hmc_file = str(tmp_path / "s2.num_hmc_cxreport.txt.gz")
        bg_file = str(tmp_path / "s1.bedGraph")
        # Provide all three files in shuffled order
        files = [mc_file, bg_file, hmc_file]
        names = ["mc_name", "bg_name", "hmc_name"]

        from pathlib import Path

        methyldackel, methyldackel_n = [], []
        mc, mc_n = [], []
        hmc, hmc_n = [], []

        for _f, _n in zip(files, names):
            fname = Path(_f).name
            if "num_hmc_cxreport" in fname:
                hmc.append(_f)
                hmc_n.append(_n)
            elif "num_mc_cxreport" in fname:
                mc.append(_f)
                mc_n.append(_n)
            else:
                methyldackel.append(_f)
                methyldackel_n.append(_n)

        assert methyldackel == [bg_file]
        assert methyldackel_n == ["bg_name"]
        assert mc == [mc_file]
        assert mc_n == ["mc_name"]
        assert hmc == [hmc_file]
        assert hmc_n == ["hmc_name"]


# ---------------------------------------------------------------------------
# Helpers for stub file creation
# ---------------------------------------------------------------------------


def _make_bam(path: Path) -> Path:
    """Create a tiny stub .bam file with a .bai index."""
    path.write_bytes(b"BAM_STUB")
    path.with_suffix(".bam.bai").write_bytes(b"BAI_STUB")
    return path


def _make_vcf(path: Path) -> Path:
    """Create a tiny stub .vcf.gz file with a .tbi index."""
    path.write_bytes(b"VCF_STUB")
    Path(str(path) + ".tbi").write_bytes(b"TBI_STUB")
    return path


# ---------------------------------------------------------------------------
# create_dataset validation: BAM file errors
# ---------------------------------------------------------------------------


class TestCreateDatasetBamValidation:
    def _call(self, tmp_path, **kwargs):
        """Call create_dataset with MultiomicsStore.from_files monkeypatched to a no-op."""
        with patch(
            "quantnado.api.MultiomicsStore.from_files",
            side_effect=RuntimeError("should not be reached"),
        ):
            return QuantNado.create_dataset(store_dir=tmp_path / "store", **kwargs)

    def test_missing_bam_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="BAM"):
            self._call(tmp_path, bam_files=[str(tmp_path / "missing.bam")])

    def test_empty_bam_file_raises(self, tmp_path):
        empty = tmp_path / "empty.bam"
        empty.write_bytes(b"")
        (tmp_path / "empty.bam.bai").write_bytes(b"x")
        with pytest.raises(ValueError, match="empty"):
            self._call(tmp_path, bam_files=[str(empty)])

    def test_bam_missing_bai_index_raises(self, tmp_path):
        bam = tmp_path / "sample.bam"
        bam.write_bytes(b"BAM_STUB")
        # No .bai / .csi file created
        with pytest.raises(FileNotFoundError, match="missing index"):
            self._call(tmp_path, bam_files=[str(bam)])

    def test_wrong_bam_extension_raises(self, tmp_path):
        wrong = tmp_path / "sample.txt"
        wrong.write_bytes(b"data")
        with pytest.raises(ValueError, match=r"\.bam"):
            self._call(tmp_path, bam_files=[str(wrong)])

    def test_valid_bam_passes_validation(self, tmp_path):
        bam = _make_bam(tmp_path / "sample.bam")
        # Should proceed past validation; MultiomicsStore.from_files is
        # monkeypatched to raise so we catch its sentinel error instead.
        with patch(
            "quantnado.api.MultiomicsStore.from_files",
            side_effect=RuntimeError("sentinel"),
        ):
            with pytest.raises(RuntimeError, match="sentinel"):
                QuantNado.create_dataset(
                    store_dir=tmp_path / "store",
                    bam_files=[str(bam)],
                )


# ---------------------------------------------------------------------------
# create_dataset validation: VCF file errors
# ---------------------------------------------------------------------------


class TestCreateDatasetVcfValidation:
    def test_vcf_missing_tbi_raises(self, tmp_path):
        vcf = tmp_path / "sample.vcf.gz"
        vcf.write_bytes(b"VCF_STUB")
        # No .tbi index
        with patch(
            "quantnado.api.MultiomicsStore.from_files",
            side_effect=RuntimeError("sentinel"),
        ):
            with pytest.raises(FileNotFoundError, match="missing tabix"):
                QuantNado.create_dataset(
                    store_dir=tmp_path / "store",
                    vcf_files=[str(vcf)],
                )

    def test_wrong_vcf_extension_raises(self, tmp_path):
        wrong = tmp_path / "sample.vcf"
        wrong.write_bytes(b"VCF_STUB")
        with patch(
            "quantnado.api.MultiomicsStore.from_files",
            side_effect=RuntimeError("sentinel"),
        ):
            with pytest.raises(ValueError, match=r"vcf\.gz"):
                QuantNado.create_dataset(
                    store_dir=tmp_path / "store",
                    vcf_files=[str(wrong)],
                )

    def test_valid_vcf_passes_validation(self, tmp_path):
        vcf = _make_vcf(tmp_path / "sample.vcf.gz")
        with patch(
            "quantnado.api.MultiomicsStore.from_files",
            side_effect=RuntimeError("sentinel"),
        ):
            with pytest.raises(RuntimeError, match="sentinel"):
                QuantNado.create_dataset(
                    store_dir=tmp_path / "store",
                    vcf_files=[str(vcf)],
                )


# ---------------------------------------------------------------------------
# create_dataset validation: methylation_sample_names length
# ---------------------------------------------------------------------------


class TestCreateDatasetMethylationNameLength:
    def test_wrong_length_raises(self, tmp_path):
        bg = tmp_path / "s1.bedGraph"
        bg.write_bytes(b"data")
        with patch(
            "quantnado.api.MultiomicsStore.from_files",
            side_effect=RuntimeError("sentinel"),
        ):
            with pytest.raises(ValueError, match="methylation_sample_names length"):
                QuantNado.create_dataset(
                    store_dir=tmp_path / "store",
                    methylation_files=[str(bg)],
                    methylation_sample_names=["name1", "name2"],  # 2 names but 1 file
                )

    def test_correct_length_passes(self, tmp_path):
        bg = tmp_path / "s1.bedGraph"
        bg.write_bytes(b"data")
        with patch(
            "quantnado.api.MultiomicsStore.from_files",
            side_effect=RuntimeError("sentinel"),
        ):
            with pytest.raises(RuntimeError, match="sentinel"):
                QuantNado.create_dataset(
                    store_dir=tmp_path / "store",
                    methylation_files=[str(bg)],
                    methylation_sample_names=["name1"],
                )


# ---------------------------------------------------------------------------
# create_dataset validation: duplicate names within a category
# ---------------------------------------------------------------------------


class TestCreateDatasetDuplicateNamesWithinCategory:
    def test_duplicate_bam_names_raises(self, tmp_path):
        bam1 = _make_bam(tmp_path / "a.bam")
        bam2 = _make_bam(tmp_path / "b.bam")
        with patch(
            "quantnado.api.MultiomicsStore.from_files",
            side_effect=RuntimeError("sentinel"),
        ):
            with pytest.raises(ValueError, match="Duplicate"):
                QuantNado.create_dataset(
                    store_dir=tmp_path / "store",
                    bam_files=[str(bam1), str(bam2)],
                    bam_sample_names=["same", "same"],
                )

    def test_duplicate_methyldackel_names_raises(self, tmp_path):
        bg1 = tmp_path / "s1.bedGraph"
        bg2 = tmp_path / "s2.bedGraph"
        bg1.write_bytes(b"data")
        bg2.write_bytes(b"data")
        with patch(
            "quantnado.api.MultiomicsStore.from_files",
            side_effect=RuntimeError("sentinel"),
        ):
            with pytest.raises(ValueError, match="Duplicate"):
                QuantNado.create_dataset(
                    store_dir=tmp_path / "store",
                    methylation_files=[str(bg1), str(bg2)],
                    methylation_sample_names=["dup", "dup"],
                )


# ---------------------------------------------------------------------------
# create_dataset validation: duplicate names across methylation categories
# ---------------------------------------------------------------------------


class TestCreateDatasetDuplicateNamesAcrossCategories:
    def test_bedgraph_and_cxreport_same_name_raises(self, tmp_path):
        bg = tmp_path / "s1.bedGraph"
        bg.write_bytes(b"data")
        cx = tmp_path / "s1.CXreport.txt"
        cx.write_bytes(b"data")
        with patch(
            "quantnado.api.MultiomicsStore.from_files",
            side_effect=RuntimeError("sentinel"),
        ):
            with pytest.raises(ValueError, match="Duplicate methylation sample names across"):
                QuantNado.create_dataset(
                    store_dir=tmp_path / "store",
                    methylation_files=[str(bg), str(cx)],
                    # both derive the same prefix "s1" from filename
                )

    def test_bedgraph_and_mc_same_name_raises(self, tmp_path):
        bg = tmp_path / "s1.bedGraph"
        bg.write_bytes(b"data")
        mc = tmp_path / "s1.num_mc_cxreport.txt.gz"
        mc.write_bytes(b"data")
        with patch(
            "quantnado.api.MultiomicsStore.from_files",
            side_effect=RuntimeError("sentinel"),
        ):
            with pytest.raises(ValueError, match="Duplicate methylation sample names across"):
                QuantNado.create_dataset(
                    store_dir=tmp_path / "store",
                    methylation_files=[str(bg), str(mc)],
                    methylation_sample_names=["shared_name", "shared_name"],
                )

    def test_no_cross_duplicate_passes(self, tmp_path):
        bg = tmp_path / "s1.bedGraph"
        bg.write_bytes(b"data")
        mc = tmp_path / "s2.num_mc_cxreport.txt.gz"
        mc.write_bytes(b"data")
        with patch(
            "quantnado.api.MultiomicsStore.from_files",
            side_effect=RuntimeError("sentinel"),
        ):
            with pytest.raises(RuntimeError, match="sentinel"):
                QuantNado.create_dataset(
                    store_dir=tmp_path / "store",
                    methylation_files=[str(bg), str(mc)],
                    methylation_sample_names=["s1", "s2"],
                )

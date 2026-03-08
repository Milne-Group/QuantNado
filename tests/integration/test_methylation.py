"""Integration tests for MethylStore using monkeypatched _read_bedgraph."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantnado.dataset.store_methyl import MethylStore, _read_bedgraph, _read_split_cxreport


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _fake_bedgraph_data(chrom: str, positions: list[int], pct: float = 75.0):
    """Build a fake per-chromosome DataFrame matching _read_bedgraph output."""
    n = len(positions)
    return pd.DataFrame(
        {
            "chrom": [chrom] * n,
            "start": positions,
            "end": [p + 2 for p in positions],
            "methylation_pct": np.full(n, pct, dtype=np.float32),
            "n_unmethylated": np.ones(n, dtype=np.uint16),
            "n_methylated": np.full(n, 3, dtype=np.uint16),
        }
    )


def _make_fake_reader(data: dict[str, dict[str, pd.DataFrame]]):
    """Return a callable that maps path stem → dict of chrom → DataFrame."""
    def _fake(path, filter_chromosomes=True):
        stem = str(path)
        for key, val in data.items():
            if key in stem:
                return val
        return {}
    return _fake


@pytest.fixture
def two_sample_store(tmp_path, monkeypatch):
    """MethylStore with 2 samples and 2 chromosomes, all monkeypatched."""
    positions_s1 = {
        "chr1": [100, 200, 300],
        "chr2": [50, 150],
    }
    positions_s2 = {
        "chr1": [200, 300, 400],  # 200 and 300 shared with s1; 400 is new
        "chr2": [50],
    }

    fake_data = {
        "s1": {c: _fake_bedgraph_data(c, ps) for c, ps in positions_s1.items()},
        "s2": {c: _fake_bedgraph_data(c, ps, pct=50.0) for c, ps in positions_s2.items()},
    }

    monkeypatch.setattr("quantnado.dataset.store_methyl._read_bedgraph", _make_fake_reader(fake_data))

    store = MethylStore.from_bedgraph_files(
        methyldackel_files=[tmp_path / "s1.bedGraph", tmp_path / "s2.bedGraph"],
        store_path=tmp_path / "methyl",
        sample_names=["s1", "s2"],
    )
    return store


# ---------------------------------------------------------------------------
# _read_bedgraph (unit-level, uses real parsing)
# ---------------------------------------------------------------------------

class TestReadBedgraph:
    def test_basic_parsing(self, tmp_path):
        bg = tmp_path / "sample.bedGraph"
        bg.write_text(
            "chr1\t100\t102\t75.0\t1\t3\n"
            "chr1\t200\t202\t50.0\t2\t2\n"
            "chr2\t50\t52\t100.0\t0\t4\n"
        )
        result = _read_bedgraph(bg, filter_chromosomes=False)
        assert set(result.keys()) == {"chr1", "chr2"}
        assert len(result["chr1"]) == 2
        assert result["chr1"]["methylation_pct"].dtype == np.float32

    def test_track_header_skipped(self, tmp_path):
        bg = tmp_path / "sample.bedGraph"
        bg.write_text(
            "track type=bedGraph\n"
            "chr1\t100\t102\t75.0\t1\t3\n"
        )
        result = _read_bedgraph(bg, filter_chromosomes=False)
        assert len(result["chr1"]) == 1

    def test_filter_chromosomes_removes_scaffolds(self, tmp_path):
        bg = tmp_path / "sample.bedGraph"
        bg.write_text(
            "chr1\t100\t102\t75.0\t1\t3\n"
            "chr1_alt\t100\t102\t50.0\t1\t1\n"
            "scaffold1\t0\t2\t0.0\t0\t0\n"
        )
        result = _read_bedgraph(bg, filter_chromosomes=True)
        assert "chr1" in result
        assert "chr1_alt" not in result
        assert "scaffold1" not in result


# ---------------------------------------------------------------------------
# MethylStore construction
# ---------------------------------------------------------------------------

class TestMethylStoreConstruction:
    def test_empty_sample_names_raises(self, tmp_path):
        with pytest.raises(ValueError, match="empty"):
            MethylStore(tmp_path / "m", sample_names=[])

    def test_overwrite_false_raises_on_existing(self, tmp_path):
        MethylStore(tmp_path / "m", sample_names=["s1"])
        with pytest.raises(FileExistsError):
            MethylStore(tmp_path / "m", sample_names=["s1"], overwrite=False)

    def test_resume_stores_root(self, tmp_path):
        MethylStore(tmp_path / "m", sample_names=["s1"])
        store = MethylStore(tmp_path / "m", sample_names=["s1"], overwrite=False, resume=True)
        assert store.root is not None

    def test_normalize_path_adds_zarr_suffix(self, tmp_path):
        store = MethylStore(tmp_path / "mystore", sample_names=["s1"])
        assert str(store.store_path).endswith(".zarr")

    def test_sample_names_mismatch_raises_on_resume(self, tmp_path):
        MethylStore(tmp_path / "m", sample_names=["s1"])
        with pytest.raises(ValueError, match="mismatch"):
            MethylStore(tmp_path / "m", sample_names=["s2"], overwrite=False, resume=True)

    def test_read_only_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            MethylStore(tmp_path / "nonexistent", sample_names=["s1"], read_only=True)

    def test_read_only_overwrite_raises(self, tmp_path):
        MethylStore(tmp_path / "m", sample_names=["s1"])
        with pytest.raises(ValueError, match="read-only"):
            MethylStore(tmp_path / "m", sample_names=["s1"], overwrite=True, read_only=True)

    def test_overwrite_existing_store(self, tmp_path, monkeypatch):
        # Covers lines 101-106: delete dir + reinit when overwrite=True on existing store
        monkeypatch.setattr(
            "quantnado.dataset.store_methyl._read_bedgraph",
            lambda *a, **k: {"chr1": _fake_bedgraph_data("chr1", [100])},
        )
        MethylStore.from_bedgraph_files(
            methyldackel_files=[tmp_path / "s1.bedGraph"],
            store_path=tmp_path / "m",
            sample_names=["s1"],
        )
        # Overwrite the same path - triggers rmtree + re-init
        store2 = MethylStore.from_bedgraph_files(
            methyldackel_files=[tmp_path / "s1.bedGraph"],
            store_path=tmp_path / "m",
            sample_names=["s1"],
            overwrite=True,
        )
        assert len(store2.chromosomes) == 1

    def test_validate_sample_names_missing_attr_raises(self, tmp_path):
        # Covers line 183: missing sample_names in root attrs during resume
        import zarr
        from zarr.storage import LocalStore
        store_path = tmp_path / "bad.zarr"
        store = LocalStore(str(store_path))
        root = zarr.group(store=store, overwrite=True, zarr_format=3)
        meta = root.create_group("metadata")
        meta.create_array("completed", shape=(1,), dtype=bool, fill_value=False)
        # Don't write sample_names to attrs
        with pytest.raises(ValueError, match="missing sample_names"):
            MethylStore(store_path, sample_names=["s1"], overwrite=False, resume=True)


# ---------------------------------------------------------------------------
# MethylStore.open
# ---------------------------------------------------------------------------

class TestMethylStoreOpen:
    def test_open_existing(self, two_sample_store, tmp_path):
        store2 = MethylStore.open(two_sample_store.store_path)
        assert store2.sample_names == ["s1", "s2"]

    def test_open_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            MethylStore.open(tmp_path / "missing.zarr")

    def test_open_missing_sample_names_raises(self, tmp_path):
        # Covers lines 130-131 in open(): KeyError when sample_names attr absent
        import zarr
        from zarr.storage import LocalStore
        store_path = tmp_path / "noattr.zarr"
        root = zarr.group(store=LocalStore(str(store_path)), overwrite=True, zarr_format=3)
        # No sample_names attribute
        with pytest.raises(ValueError, match="Missing required attribute"):
            MethylStore.open(store_path)

    def test_from_bedgraph_with_metadata_df(self, tmp_path, monkeypatch):
        # Covers lines 317-321: passing metadata DataFrame to from_bedgraph_files
        monkeypatch.setattr(
            "quantnado.dataset.store_methyl._read_bedgraph",
            lambda *a, **k: {"chr1": _fake_bedgraph_data("chr1", [100])},
        )
        md = pd.DataFrame({"sample_id": ["s1"], "condition": ["ctrl"]})
        store = MethylStore.from_bedgraph_files(
            methyldackel_files=[tmp_path / "s1.bedGraph"],
            store_path=tmp_path / "m",
            sample_names=["s1"],
            metadata=md,
        )
        result = store.get_metadata()
        assert "condition" in result.columns


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------

class TestMethylStoreDataAccess:
    def test_chromosomes_property(self, two_sample_store):
        chroms = two_sample_store.chromosomes
        assert set(chroms) == {"chr1", "chr2"}

    def test_get_positions_chr1(self, two_sample_store):
        positions = two_sample_store.get_positions("chr1")
        # Union of s1=[100,200,300] and s2=[200,300,400] = [100,200,300,400]
        assert list(positions) == [100, 200, 300, 400]

    def test_get_positions_chr2(self, two_sample_store):
        positions = two_sample_store.get_positions("chr2")
        assert list(positions) == [50, 150]

    def test_to_xarray_returns_dict(self, two_sample_store):
        result = two_sample_store.to_xarray(variable="methylation_pct")
        assert isinstance(result, dict)
        assert "chr1" in result

    def test_to_xarray_dims(self, two_sample_store):
        result = two_sample_store.to_xarray(variable="methylation_pct")
        da = result["chr1"]
        assert da.dims == ("sample", "position")
        assert da.shape == (2, 4)

    def test_to_xarray_invalid_variable_raises(self, two_sample_store):
        with pytest.raises(ValueError, match="variable"):
            two_sample_store.to_xarray(variable="bad_var")

    def test_to_xarray_invalid_chrom_raises(self, two_sample_store):
        with pytest.raises(ValueError, match="not in store"):
            two_sample_store.to_xarray(chromosomes=["chrX"])

    def test_to_xarray_subset_chromosomes(self, two_sample_store):
        result = two_sample_store.to_xarray(chromosomes=["chr1"])
        assert list(result.keys()) == ["chr1"]

    def test_completed_mask_all_true(self, two_sample_store):
        assert two_sample_store.completed_mask.all()


# ---------------------------------------------------------------------------
# extract_region
# ---------------------------------------------------------------------------

class TestExtractRegion:
    def test_region_string(self, two_sample_store):
        result = two_sample_store.extract_region("chr1:150-350")
        assert isinstance(result, xr.DataArray)
        # positions in [150,350) → 200, 300
        assert result.shape[1] == 2

    def test_chrom_start_end(self, two_sample_store):
        result = two_sample_store.extract_region(chrom="chr1", start=200, end=400)
        assert isinstance(result, xr.DataArray)
        assert result.shape == (2, 2)  # positions 200, 300

    def test_as_numpy(self, two_sample_store):
        result = two_sample_store.extract_region("chr1", as_xarray=False)
        assert isinstance(result, np.ndarray)

    def test_subset_samples_by_name(self, two_sample_store):
        result = two_sample_store.extract_region("chr1", samples=["s1"])
        assert result.shape[0] == 1

    def test_subset_samples_by_int(self, two_sample_store):
        # Covers the `else: idx = int(s)` branch (line 467)
        result = two_sample_store.extract_region("chr1", samples=[0])
        assert result.shape[0] == 1

    def test_unknown_sample_raises(self, two_sample_store):
        with pytest.raises(ValueError, match="not found"):
            two_sample_store.extract_region("chr1", samples=["ghost"])

    def test_both_region_and_chrom_raises(self, two_sample_store):
        with pytest.raises(ValueError, match="not both"):
            two_sample_store.extract_region(region="chr1:0-100", chrom="chr1")

    def test_missing_chrom_raises(self, two_sample_store):
        with pytest.raises(ValueError):
            two_sample_store.extract_region(chrom=None)

    def test_unknown_chrom_raises(self, two_sample_store):
        with pytest.raises(ValueError, match="not in store"):
            two_sample_store.extract_region(chrom="chrZ")


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

class TestMethylStoreMetadata:
    def test_set_and_get_metadata(self, two_sample_store):
        md = pd.DataFrame({"sample_id": ["s1", "s2"], "condition": ["ctrl", "treat"]})
        two_sample_store.set_metadata(md)
        result = two_sample_store.get_metadata()
        assert "condition" in result.columns

    def test_missing_sample_column_raises(self, two_sample_store):
        md = pd.DataFrame({"name": ["s1", "s2"]})
        with pytest.raises(ValueError, match="sample_id"):
            two_sample_store.set_metadata(md)

    def test_read_only_set_metadata_raises(self, two_sample_store, tmp_path):
        store_ro = MethylStore.open(two_sample_store.store_path, read_only=True)
        md = pd.DataFrame({"sample_id": ["s1", "s2"], "x": [1, 2]})
        with pytest.raises(RuntimeError, match="read-only"):
            store_ro.set_metadata(md)


# ---------------------------------------------------------------------------
# Helpers for split CXreport tests
# ---------------------------------------------------------------------------

def _write_split_cxreport(path, rows):
    """Write a 7-column split CXreport file.

    Columns: chrom, pos(0-based), strand, n_mod, n_not_mod, context, trinucleotide
    """
    content = "\n".join(
        "\t".join(str(v) for v in row) for row in rows
    ) + "\n"
    path.write_text(content)
    return path


# ---------------------------------------------------------------------------
# TestReadSplitCXreport
# ---------------------------------------------------------------------------


class TestReadSplitCXreport:
    """Unit-level tests for the _read_split_cxreport helper."""

    def _make_mc_rows(self):
        # chrom, pos, strand, n_mod, n_not_mod, context, trinuc
        return [
            ("chr1", 100, "+", 5, 5, "CG", "CGT"),
            ("chr1", 101, "-", 3, 7, "CG", "CGC"),
            ("chr2", 50,  "+", 2, 8, "CG", "CGG"),
        ]

    def _make_hmc_rows(self):
        return [
            ("chr1", 100, "+", 1, 9, "CG", "CGT"),
            ("chr1", 101, "-", 2, 8, "CG", "CGC"),
            ("chr2", 50,  "+", 0, 10, "CG", "CGG"),
        ]

    def test_both_mc_and_hmc_paths(self, tmp_path):
        mc_path = _write_split_cxreport(tmp_path / "mc.txt", self._make_mc_rows())
        hmc_path = _write_split_cxreport(tmp_path / "hmc.txt", self._make_hmc_rows())

        result = _read_split_cxreport(mc_path, hmc_path, filter_chromosomes=False)

        assert "chr1" in result
        assert "chr2" in result
        df1 = result["chr1"]
        assert "n_mc" in df1.columns
        assert "n_hmc" in df1.columns
        assert "n_c" in df1.columns
        assert "methylation_pct" in df1.columns
        # Both strands for chr1 pos 100 and 101 are merged into one row per canonical pos
        assert len(df1) == 1  # canonical_pos for both rows is 100

    def test_only_mc_path(self, tmp_path):
        mc_path = _write_split_cxreport(tmp_path / "mc.txt", self._make_mc_rows())

        result = _read_split_cxreport(mc_path, None, filter_chromosomes=False)

        assert "chr1" in result
        df1 = result["chr1"]
        # n_hmc should be all zeros
        assert (df1["n_hmc"] == 0).all()
        # n_mc should reflect n_mod from mc file
        assert df1["n_mc"].sum() > 0

    def test_only_hmc_path(self, tmp_path):
        hmc_path = _write_split_cxreport(tmp_path / "hmc.txt", self._make_hmc_rows())

        result = _read_split_cxreport(None, hmc_path, filter_chromosomes=False)

        assert "chr1" in result
        df1 = result["chr1"]
        # n_mc should be all zeros
        assert (df1["n_mc"] == 0).all()
        # n_hmc should reflect n_mod from hmc file
        assert df1["n_hmc"].sum() > 0

    def test_neither_path_raises(self):
        with pytest.raises(ValueError):
            _read_split_cxreport(None, None)

    def test_filter_chromosomes_removes_scaffolds(self, tmp_path):
        rows = [
            ("chr1", 100, "+", 5, 5, "CG", "CGT"),
            ("chr1_alt", 100, "+", 2, 8, "CG", "CGT"),
            ("scaffold1", 50, "+", 1, 9, "CG", "CGT"),
        ]
        mc_path = _write_split_cxreport(tmp_path / "mc.txt", rows)

        result_filtered = _read_split_cxreport(mc_path, None, filter_chromosomes=True)
        result_all = _read_split_cxreport(mc_path, None, filter_chromosomes=False)

        assert "chr1" in result_filtered
        assert "chr1_alt" not in result_filtered
        assert "scaffold1" not in result_filtered

        assert "chr1" in result_all
        assert "chr1_alt" in result_all
        assert "scaffold1" in result_all


# ---------------------------------------------------------------------------
# TestFromSplitCXreportFiles
# ---------------------------------------------------------------------------


class TestFromSplitCXreportFiles:
    """Tests for MethylStore.from_split_cxreport_files."""

    def _mc_rows(self):
        return [
            ("chr1", 100, "+", 5, 5, "CG", "CGT"),
            ("chr1", 101, "-", 3, 7, "CG", "CGC"),
        ]

    def _hmc_rows(self):
        return [
            ("chr1", 100, "+", 1, 9, "CG", "CGT"),
            ("chr1", 101, "-", 2, 8, "CG", "CGC"),
        ]

    def test_both_mc_and_hmc_files(self, tmp_path):
        mc = _write_split_cxreport(tmp_path / "s1.num_mc_cxreport.txt", self._mc_rows())
        hmc = _write_split_cxreport(tmp_path / "s1.num_hmc_cxreport.txt", self._hmc_rows())

        store = MethylStore.from_split_cxreport_files(
            mc_files=[mc],
            hmc_files=[hmc],
            store_path=tmp_path / "methyl",
            sample_names=["s1"],
        )

        assert store.sample_names == ["s1"]
        assert "chr1" in store.chromosomes
        df = store.root["chr1"]
        # n_mc and n_hmc arrays should exist
        assert "n_mc" in df
        assert "n_hmc" in df

    def test_mc_files_only(self, tmp_path):
        mc = _write_split_cxreport(tmp_path / "s1.num_mc_cxreport.txt", self._mc_rows())

        store = MethylStore.from_split_cxreport_files(
            mc_files=[mc],
            hmc_files=None,
            store_path=tmp_path / "methyl",
            sample_names=["s1"],
        )

        assert store.sample_names == ["s1"]
        assert "chr1" in store.chromosomes

    def test_hmc_files_only(self, tmp_path):
        hmc = _write_split_cxreport(tmp_path / "s1.num_hmc_cxreport.txt", self._hmc_rows())

        store = MethylStore.from_split_cxreport_files(
            mc_files=None,
            hmc_files=[hmc],
            store_path=tmp_path / "methyl",
            sample_names=["s1"],
        )

        assert store.sample_names == ["s1"]
        assert "chr1" in store.chromosomes

    def test_neither_mc_nor_hmc_raises(self, tmp_path):
        with pytest.raises(ValueError):
            MethylStore.from_split_cxreport_files(
                mc_files=None,
                hmc_files=None,
                store_path=tmp_path / "methyl",
            )

    def test_length_mismatch_raises(self, tmp_path):
        mc1 = _write_split_cxreport(tmp_path / "s1.num_mc_cxreport.txt", self._mc_rows())
        mc2 = _write_split_cxreport(tmp_path / "s2.num_mc_cxreport.txt", self._mc_rows())
        hmc1 = _write_split_cxreport(tmp_path / "s1.num_hmc_cxreport.txt", self._hmc_rows())

        with pytest.raises(ValueError, match="same length"):
            MethylStore.from_split_cxreport_files(
                mc_files=[mc1, mc2],
                hmc_files=[hmc1],  # length mismatch
                store_path=tmp_path / "methyl",
            )

    def test_default_sample_names_from_filename(self, tmp_path):
        mc = _write_split_cxreport(tmp_path / "mysample.num_mc_cxreport.txt", self._mc_rows())

        store = MethylStore.from_split_cxreport_files(
            mc_files=[mc],
            store_path=tmp_path / "methyl",
            sample_names=None,
        )

        assert store.sample_names == ["mysample"]


# ---------------------------------------------------------------------------
# TestFromMixedFiles
# ---------------------------------------------------------------------------


class TestFromMixedFiles:
    """Tests for MethylStore.from_mixed_files."""

    def _bg_rows(self):
        # bedGraph format: chrom, start, end, methylation_pct, n_unmethylated, n_methylated
        return "chr1\t100\t102\t75.0\t1\t3\nchr1\t200\t202\t50.0\t2\t2\n"

    def _mc_rows(self):
        return [
            ("chr1", 300, "+", 4, 6, "CG", "CGT"),
            ("chr1", 301, "-", 2, 8, "CG", "CGC"),
        ]

    def _hmc_rows(self):
        return [
            ("chr1", 300, "+", 1, 9, "CG", "CGT"),
            ("chr1", 301, "-", 0, 10, "CG", "CGC"),
        ]

    def test_bedgraph_only(self, tmp_path):
        bg = tmp_path / "s1.bedGraph"
        bg.write_text(self._bg_rows())

        store = MethylStore.from_mixed_files(
            methyldackel_files=[bg],
            store_path=tmp_path / "methyl",
            methyldackel_sample_names=["s1"],
        )

        assert store.sample_names == ["s1"]
        assert "chr1" in store.chromosomes

    def test_mc_hmc_only(self, tmp_path):
        mc = _write_split_cxreport(tmp_path / "s1.num_mc_cxreport.txt", self._mc_rows())
        hmc = _write_split_cxreport(tmp_path / "s1.num_hmc_cxreport.txt", self._hmc_rows())

        store = MethylStore.from_mixed_files(
            mc_files=[mc],
            hmc_files=[hmc],
            store_path=tmp_path / "methyl",
            mc_hmc_sample_names=["s1"],
        )

        assert store.sample_names == ["s1"]
        assert "chr1" in store.chromosomes

    def test_bedgraph_and_mc_hmc_combination(self, tmp_path):
        bg = tmp_path / "bg_sample.bedGraph"
        bg.write_text(self._bg_rows())
        mc = _write_split_cxreport(tmp_path / "cx_sample.num_mc_cxreport.txt", self._mc_rows())
        hmc = _write_split_cxreport(tmp_path / "cx_sample.num_hmc_cxreport.txt", self._hmc_rows())

        store = MethylStore.from_mixed_files(
            methyldackel_files=[bg],
            mc_files=[mc],
            hmc_files=[hmc],
            store_path=tmp_path / "methyl",
            methyldackel_sample_names=["bg_sample"],
            mc_hmc_sample_names=["cx_sample"],
        )

        assert set(store.sample_names) == {"bg_sample", "cx_sample"}
        assert "chr1" in store.chromosomes
        # Both bedGraph and split CXreport positions should be in the union
        positions = store.get_positions("chr1")
        assert 100 in positions  # from bedGraph
        assert 300 in positions  # from CXreport

    def test_neither_provided_raises(self, tmp_path):
        with pytest.raises(ValueError):
            MethylStore.from_mixed_files(
                store_path=tmp_path / "methyl",
            )

    def test_duplicate_sample_name_across_categories_raises(self, tmp_path):
        bg = tmp_path / "shared.bedGraph"
        bg.write_text(self._bg_rows())
        mc = _write_split_cxreport(tmp_path / "shared.num_mc_cxreport.txt", self._mc_rows())

        with pytest.raises(ValueError, match="Duplicate"):
            MethylStore.from_mixed_files(
                methyldackel_files=[bg],
                mc_files=[mc],
                store_path=tmp_path / "methyl",
                methyldackel_sample_names=["shared"],
                mc_hmc_sample_names=["shared"],  # same name as bedGraph sample
            )

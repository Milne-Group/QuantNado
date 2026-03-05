"""CLI tests for QuantNado commands using Typer's test runner."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from quantnado.cli import app, combine_metadata_main, make_zarr_main
from quantnado.dataset.bam import BamStore


runner = CliRunner(env={"NO_COLOR": "1"})


# ---------------------------------------------------------------------------
# Version / help
# ---------------------------------------------------------------------------


def test_version_flag():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "quantnado" in result.output.lower()


def test_no_args_shows_help():
    result = runner.invoke(app, [])
    # Typer returns exit code 0 with help text when no_args_is_help=True
    assert "create-dataset" in result.output or "call-peaks" in result.output


def test_create_dataset_help():
    result = runner.invoke(app, ["create-dataset", "--help"])
    assert result.exit_code == 0
    assert "--output" in result.output


def test_call_peaks_help():
    result = runner.invoke(app, ["call-peaks", "--help"])
    assert result.exit_code == 0
    assert "--bigwig-dir" in result.output


# ---------------------------------------------------------------------------
# create-dataset
# ---------------------------------------------------------------------------


def test_create_dataset_basic(tmp_path, monkeypatch):
    """Happy path: create-dataset processes BAM files and exits 0."""
    created_stores = []

    def fake_from_bam_files(**kwargs):
        # Return a real BamStore without doing any BAM processing.
        cs = {"chr1": 4}
        names = [p.rsplit("/", 1)[-1].replace(".bam", "") for p in kwargs["bam_files"]]
        store = BamStore(kwargs["store_path"], cs, names)
        created_stores.append(store)
        return store

    monkeypatch.setattr("quantnado.cli.BamStore.from_bam_files", fake_from_bam_files)

    bam = tmp_path / "sample1.bam"
    bam.write_text("dummy")

    result = runner.invoke(app, [
        "create-dataset", str(bam),
        "--output", str(tmp_path / "out.zarr"),
        "--log-file", str(tmp_path / "log.log"),
    ])

    assert result.exit_code == 0, result.output
    assert len(created_stores) == 1


def test_create_dataset_missing_output_errors(tmp_path):
    bam = tmp_path / "s.bam"
    bam.write_text("dummy")
    result = runner.invoke(app, ["create-dataset", str(bam)])
    # Missing required --output flag should cause non-zero exit
    assert result.exit_code != 0


def test_create_dataset_with_metadata(tmp_path, monkeypatch):
    metadata_csv = tmp_path / "meta.csv"
    pd.DataFrame({"sample_id": ["sample1"], "condition": ["ctrl"]}).to_csv(metadata_csv, index=False)

    calls = []

    def fake_from_bam_files(**kwargs):
        calls.append(kwargs)
        return BamStore(kwargs["store_path"], {"chr1": 4}, ["sample1"])

    monkeypatch.setattr("quantnado.cli.BamStore.from_bam_files", fake_from_bam_files)

    bam = tmp_path / "sample1.bam"
    bam.write_text("dummy")

    result = runner.invoke(app, [
        "create-dataset", str(bam),
        "--output", str(tmp_path / "out.zarr"),
        "--metadata", str(metadata_csv),
        "--log-file", str(tmp_path / "log.log"),
    ])
    assert result.exit_code == 0, result.output
    assert calls[0]["metadata"] == metadata_csv


def test_create_dataset_overwrite_flag(tmp_path, monkeypatch):
    calls = []

    def fake_from_bam_files(**kwargs):
        calls.append(kwargs)
        return BamStore(kwargs["store_path"], {"chr1": 4}, ["s"])

    monkeypatch.setattr("quantnado.cli.BamStore.from_bam_files", fake_from_bam_files)

    bam = tmp_path / "s.bam"
    bam.write_text("dummy")

    runner.invoke(app, [
        "create-dataset", str(bam),
        "--output", str(tmp_path / "out.zarr"),
        "--overwrite",
        "--log-file", str(tmp_path / "log.log"),
    ])
    assert calls[0]["overwrite"] is True


def test_create_dataset_resume_flag(tmp_path, monkeypatch):
    calls = []

    def fake_from_bam_files(**kwargs):
        calls.append(kwargs)
        return BamStore(kwargs["store_path"], {"chr1": 4}, ["s"])

    monkeypatch.setattr("quantnado.cli.BamStore.from_bam_files", fake_from_bam_files)

    bam = tmp_path / "s.bam"
    bam.write_text("dummy")

    runner.invoke(app, [
        "create-dataset", str(bam),
        "--output", str(tmp_path / "out.zarr"),
        "--resume",
        "--log-file", str(tmp_path / "log.log"),
    ])
    assert calls[0]["resume"] is True


def test_create_dataset_propagates_exception(tmp_path, monkeypatch):
    def fake_from_bam_files(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("quantnado.cli.BamStore.from_bam_files", fake_from_bam_files)

    bam = tmp_path / "s.bam"
    bam.write_text("dummy")

    result = runner.invoke(app, [
        "create-dataset", str(bam),
        "--output", str(tmp_path / "out.zarr"),
        "--log-file", str(tmp_path / "log.log"),
    ])
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# call-peaks
# ---------------------------------------------------------------------------


def test_call_peaks_basic(tmp_path, monkeypatch):
    calls = []

    def fake_call_peaks(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        "quantnado.peak_calling.call_quantile_peaks.call_peaks_from_bigwig_dir",
        fake_call_peaks,
    )

    bw_dir = tmp_path / "bigwigs"
    bw_dir.mkdir()
    chromsizes = tmp_path / "hg38.sizes"
    chromsizes.write_text("chr1\t248956422\n")
    out_dir = tmp_path / "peaks"

    result = runner.invoke(app, [
        "call-peaks",
        "--bigwig-dir", str(bw_dir),
        "--output-dir", str(out_dir),
        "--chromsizes", str(chromsizes),
        "--quantile", "0.98",
        "--log-file", str(tmp_path / "peaks.log"),
    ])
    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert calls[0]["quantile"] == 0.98


def test_call_peaks_propagates_exception(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "quantnado.peak_calling.call_quantile_peaks.call_peaks_from_bigwig_dir",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("peak error")),
    )
    result = runner.invoke(app, [
        "call-peaks",
        "--bigwig-dir", str(tmp_path),
        "--output-dir", str(tmp_path / "out"),
        "--chromsizes", str(tmp_path / "cs.txt"),
        "--log-file", str(tmp_path / "log.log"),
    ])
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# make_zarr_main alias entry point
# ---------------------------------------------------------------------------


def test_make_zarr_main_is_callable():
    """make_zarr_main must exist and be callable (it's just app())."""
    assert callable(make_zarr_main)


# ---------------------------------------------------------------------------
# combine_metadata_main entry point
# ---------------------------------------------------------------------------


def test_combine_metadata_main_merges_csvs(tmp_path):
    f1 = tmp_path / "a.csv"
    f2 = tmp_path / "b.csv"
    f1.write_text("sample_id,condition\ns1,ctrl\n")
    f2.write_text("sample_id,condition\ns2,treat\n")
    out = tmp_path / "combined.csv"

    from typer.testing import CliRunner as TR
    from quantnado.cli import combine_metadata_main
    import typer

    # Build a minimal runner against the nested Typer app
    combine_runner = TR()

    # combine_metadata_main() creates a new app internally; invoke it directly
    # by calling BamStore._combine_metadata_files (the underlying logic is tested here)
    from quantnado.dataset.bam import BamStore
    combined = BamStore._combine_metadata_files([str(f1), str(f2)])
    combined.to_csv(out, index=False)

    reloaded = pd.read_csv(out)
    assert set(reloaded["sample_id"]) == {"s1", "s2"}
    assert len(reloaded) == 2

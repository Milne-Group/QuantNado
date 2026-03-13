"""CLI tests for QuantNado commands using Typer's test runner."""
from __future__ import annotations

import re
import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from quantnado.cli import app, combine_metadata_main, make_zarr_main
from quantnado.dataset.store_bam import BamStore
from quantnado.dataset.store_multiomics import MultiomicsStore


runner = CliRunner()


def _strip_ansi(text: str) -> str:
    return re.sub(r'\x1b\[[0-9;]*[mGKHFABCDEFJKST]', '', text)


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
    assert "--output" in _strip_ansi(result.output)


def test_call_peaks_help():
    result = runner.invoke(app, ["call-peaks", "--help"])
    assert result.exit_code == 0
    output = _strip_ansi(result.output)
    assert "--zarr" in output
    assert "--window-overlap" in output


# ---------------------------------------------------------------------------
# create-dataset
# ---------------------------------------------------------------------------


def test_create_dataset_basic(tmp_path, monkeypatch):
    """Happy path: create-dataset processes BAM files and exits 0."""
    calls = []

    def fake_from_files(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("quantnado.dataset.store_multiomics.MultiomicsStore.from_files", fake_from_files)

    bam = tmp_path / "sample1.bam"
    bam.write_text("dummy")

    result = runner.invoke(app, [
        "create-dataset",
        "--bam", str(bam),
        "--output", str(tmp_path / "out.zarr"),
        "--log-file", str(tmp_path / "log.log"),
    ])

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert calls[0]["store_dir"] == tmp_path / "out.zarr"
    assert calls[0]["bam_files"] == [str(bam)]


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

    def fake_from_files(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("quantnado.dataset.store_multiomics.MultiomicsStore.from_files", fake_from_files)

    bam = tmp_path / "sample1.bam"
    bam.write_text("dummy")

    result = runner.invoke(app, [
        "create-dataset",
        "--bam", str(bam),
        "--output", str(tmp_path / "out.zarr"),
        "--metadata", str(metadata_csv),
        "--log-file", str(tmp_path / "log.log"),
    ])
    assert result.exit_code == 0, result.output
    assert calls[0]["metadata"] == metadata_csv


def test_create_dataset_overwrite_flag(tmp_path, monkeypatch):
    calls = []

    def fake_from_files(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("quantnado.dataset.store_multiomics.MultiomicsStore.from_files", fake_from_files)

    bam = tmp_path / "s.bam"
    bam.write_text("dummy")

    runner.invoke(app, [
        "create-dataset",
        "--bam", str(bam),
        "--output", str(tmp_path / "out.zarr"),
        "--overwrite",
        "--log-file", str(tmp_path / "log.log"),
    ])
    assert calls[0]["overwrite"] is True


def test_create_dataset_resume_flag(tmp_path, monkeypatch):
    calls = []

    def fake_from_files(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("quantnado.dataset.store_multiomics.MultiomicsStore.from_files", fake_from_files)

    bam = tmp_path / "s.bam"
    bam.write_text("dummy")

    runner.invoke(app, [
        "create-dataset",
        "--bam", str(bam),
        "--output", str(tmp_path / "out.zarr"),
        "--resume",
        "--log-file", str(tmp_path / "log.log"),
    ])
    assert calls[0]["resume"] is True


def test_create_dataset_chunk_len_flag(tmp_path, monkeypatch):
    calls = []

    def fake_from_files(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("quantnado.dataset.store_multiomics.MultiomicsStore.from_files", fake_from_files)

    bam = tmp_path / "s.bam"
    bam.write_text("dummy")

    result = runner.invoke(app, [
        "create-dataset",
        "--bam", str(bam),
        "--output", str(tmp_path / "out.zarr"),
        "--chunk-len", "131072",
        "--log-file", str(tmp_path / "log.log"),
    ])
    assert result.exit_code == 0, result.output
    assert calls[0]["chunk_len"] == 131072


def test_create_dataset_local_staging_flags(tmp_path, monkeypatch):
    calls = []

    def fake_from_files(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("quantnado.dataset.store_multiomics.MultiomicsStore.from_files", fake_from_files)

    bam = tmp_path / "s.bam"
    bam.write_text("dummy")
    staging_dir = tmp_path / "scratch"

    result = runner.invoke(app, [
        "create-dataset",
        "--bam", str(bam),
        "--output", str(tmp_path / "out.zarr"),
        "--local-staging",
        "--staging-dir", str(staging_dir),
        "--log-file", str(tmp_path / "log.log"),
    ])
    assert result.exit_code == 0, result.output
    assert calls[0]["local_staging"] is True
    assert calls[0]["staging_dir"] == staging_dir


def test_create_dataset_construction_compression_flag(tmp_path, monkeypatch):
    calls = []

    def fake_from_files(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("quantnado.dataset.store_multiomics.MultiomicsStore.from_files", fake_from_files)

    bam = tmp_path / "s.bam"
    bam.write_text("dummy")

    result = runner.invoke(app, [
        "create-dataset",
        "--bam", str(bam),
        "--output", str(tmp_path / "out.zarr"),
        "--construction-compression", "fast",
        "--log-file", str(tmp_path / "log.log"),
    ])
    assert result.exit_code == 0, result.output
    assert calls[0]["construction_compression"] == "fast"


def test_create_dataset_propagates_exception(tmp_path, monkeypatch):
    def fake_from_files(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("quantnado.dataset.store_multiomics.MultiomicsStore.from_files", fake_from_files)

    bam = tmp_path / "s.bam"
    bam.write_text("dummy")

    result = runner.invoke(app, [
        "create-dataset",
        "--bam", str(bam),
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
        "quantnado.peak_calling.call_quantile_peaks.call_peaks_from_zarr",
        fake_call_peaks,
    )

    zarr_path = tmp_path / "input.zarr"
    out_dir = tmp_path / "peaks"

    result = runner.invoke(app, [
        "call-peaks",
        "--zarr", str(zarr_path),
        "--output-dir", str(out_dir),
        "--quantile", "0.98",
        "--log-file", str(tmp_path / "peaks.log"),
    ])
    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert calls[0]["quantile"] == 0.98


def test_call_peaks_propagates_exception(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "quantnado.peak_calling.call_quantile_peaks.call_peaks_from_zarr",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("peak error")),
    )
    result = runner.invoke(app, [
        "call-peaks",
        "--zarr", str(tmp_path / "input.zarr"),
        "--output-dir", str(tmp_path / "out"),
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


def test_create_dataset_methylation_files_passed_through(tmp_path, monkeypatch):
    calls = []

    def fake_from_files(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("quantnado.dataset.store_multiomics.MultiomicsStore.from_files", fake_from_files)

    meth = tmp_path / "sample1_CpG.bedGraph"
    meth.write_text("dummy")

    result = runner.invoke(app, [
        "create-dataset",
        "--methylation", str(meth),
        "--output", str(tmp_path / "out.zarr"),
        "--log-file", str(tmp_path / "log.log"),
    ])
    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert calls[0]["methyldackel_files"] == [str(meth)]


def test_create_dataset_vcf_files_passed_through(tmp_path, monkeypatch):
    calls = []

    def fake_from_files(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("quantnado.dataset.store_multiomics.MultiomicsStore.from_files", fake_from_files)

    vcf = tmp_path / "sample1.vcf.gz"
    vcf.write_text("dummy")

    result = runner.invoke(app, [
        "create-dataset",
        "--vcf", str(vcf),
        "--output", str(tmp_path / "out.zarr"),
        "--log-file", str(tmp_path / "log.log"),
    ])
    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert calls[0]["vcf_files"] == [str(vcf)]


def test_create_dataset_no_inputs_exits_with_error(tmp_path, monkeypatch):
    monkeypatch.setattr("quantnado.dataset.store_multiomics.MultiomicsStore.from_files", lambda **kw: None)

    result = runner.invoke(app, [
        "create-dataset",
        "--output", str(tmp_path / "out.zarr"),
        "--log-file", str(tmp_path / "log.log"),
    ])
    assert result.exit_code == 1


def test_create_dataset_methylation_sample_names_passed_through(tmp_path, monkeypatch):
    calls = []

    def fake_from_files(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("quantnado.dataset.store_multiomics.MultiomicsStore.from_files", fake_from_files)

    meth = tmp_path / "sample1_CpG.bedGraph"
    meth.write_text("dummy")

    result = runner.invoke(app, [
        "create-dataset",
        "--methylation", str(meth),
        "--methylation-sample-names", "my_sample",
        "--output", str(tmp_path / "out.zarr"),
        "--log-file", str(tmp_path / "log.log"),
    ])
    assert result.exit_code == 0, result.output
    assert calls[0]["methyldackel_sample_names"] == ["my_sample"]


def test_create_dataset_vcf_sample_names_passed_through(tmp_path, monkeypatch):
    calls = []

    def fake_from_files(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("quantnado.dataset.store_multiomics.MultiomicsStore.from_files", fake_from_files)

    vcf = tmp_path / "sample1.vcf.gz"
    vcf.write_text("dummy")

    result = runner.invoke(app, [
        "create-dataset",
        "--vcf", str(vcf),
        "--vcf-sample-names", "vcf_sample",
        "--output", str(tmp_path / "out.zarr"),
        "--log-file", str(tmp_path / "log.log"),
    ])
    assert result.exit_code == 0, result.output
    assert calls[0]["vcf_sample_names"] == ["vcf_sample"]


def test_create_dataset_max_workers_flag(tmp_path, monkeypatch):
    calls = []

    def fake_from_files(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("quantnado.dataset.store_multiomics.MultiomicsStore.from_files", fake_from_files)

    bam = tmp_path / "s.bam"
    bam.write_text("dummy")

    result = runner.invoke(app, [
        "create-dataset",
        "--bam", str(bam),
        "--output", str(tmp_path / "out.zarr"),
        "--max-workers", "4",
        "--log-file", str(tmp_path / "log.log"),
    ])
    assert result.exit_code == 0, result.output
    assert calls[0]["max_workers"] == 4


def test_create_dataset_test_flag(tmp_path, monkeypatch):
    calls = []

    def fake_from_files(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("quantnado.dataset.store_multiomics.MultiomicsStore.from_files", fake_from_files)

    bam = tmp_path / "s.bam"
    bam.write_text("dummy")

    result = runner.invoke(app, [
        "create-dataset",
        "--bam", str(bam),
        "--output", str(tmp_path / "out.zarr"),
        "--test",
        "--log-file", str(tmp_path / "log.log"),
    ])
    assert result.exit_code == 0, result.output
    assert calls[0]["test"] is True


def test_create_dataset_multiomics_error_propagation(tmp_path, monkeypatch):
    def fake_from_files(**kwargs):
        raise ValueError("multiomics store error")

    monkeypatch.setattr("quantnado.dataset.store_multiomics.MultiomicsStore.from_files", fake_from_files)

    meth = tmp_path / "sample1_CpG.bedGraph"
    meth.write_text("dummy")

    result = runner.invoke(app, [
        "create-dataset",
        "--methylation", str(meth),
        "--output", str(tmp_path / "out.zarr"),
        "--log-file", str(tmp_path / "log.log"),
    ])
    assert result.exit_code == 1


def test_call_peaks_blacklist_and_merge_flags(tmp_path, monkeypatch):
    calls = []

    def fake_call_peaks(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        "quantnado.peak_calling.call_quantile_peaks.call_peaks_from_zarr",
        fake_call_peaks,
    )

    zarr_path = tmp_path / "input.zarr"
    blacklist = tmp_path / "blacklist.bed"
    blacklist.write_text("")
    out_dir = tmp_path / "peaks"

    result = runner.invoke(app, [
        "call-peaks",
        "--zarr", str(zarr_path),
        "--output-dir", str(out_dir),
        "--blacklist", str(blacklist),
        "--merge",
        "--log-file", str(tmp_path / "peaks.log"),
    ])
    assert result.exit_code == 0, result.output
    assert calls[0]["merge"] is True
    assert calls[0]["blacklist_file"] == blacklist


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
    from quantnado.dataset.store_bam import BamStore
    combined = BamStore._combine_metadata_files([str(f1), str(f2)])
    combined.to_csv(out, index=False)

    reloaded = pd.read_csv(out)
    assert set(reloaded["sample_id"]) == {"s1", "s2"}
    assert len(reloaded) == 2

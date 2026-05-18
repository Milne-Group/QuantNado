from __future__ import annotations

from pathlib import Path
import re
import subprocess

from typer.testing import CliRunner

from quantnado.cli import app


runner = CliRunner()


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _clean(text: str) -> str:
    return ANSI_RE.sub("", text)


def test_dataset_create_help_mentions_direct_inputs():
    result = runner.invoke(app, ["dataset", "create", "--help"])
    assert result.exit_code == 0
    clean = _clean(result.stdout)
    assert "--bamfile" in clean
    assert "--vcf_file" in clean
    assert "--methylation" in clean


def test_dataset_combine_help_mentions_stores_list():
    result = runner.invoke(app, ["dataset", "combine", "--help"])
    assert result.exit_code == 0
    clean = _clean(result.stdout)
    assert "--stores" in clean
    assert "single" in clean
    assert "flag" in clean


def test_dataset_compress_help_mentions_parallel_archive():
    result = runner.invoke(app, ["dataset", "compress", "--help"])
    assert result.exit_code == 0
    clean = _clean(result.stdout)
    assert "--dataset" in clean
    assert "--workers" in clean
    assert "pigz" in clean


def test_dataset_create_dispatches_bam_store(monkeypatch, tmp_path):
    calls: list[dict] = []

    def fake_from_bam_files(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("quantnado.dataset.store_bam.BamStore.from_bam_files", fake_from_bam_files)

    bam = tmp_path / "sample.bam"
    bam.write_text("")

    result = runner.invoke(
        app,
        [
            "dataset",
            "create",
            "--sample",
            "RNA_1",
            "--assay",
            "RNA",
            "--bamfile",
            str(bam),
            "--stranded",
            "R",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 0
    assert len(calls) == 1
    assert calls[0]["bam_path"] == str(bam)
    assert calls[0]["sample"] == "RNA_1"
    assert calls[0]["assay"] == "RNA"
    assert calls[0]["stranded"] == "R"


def test_dataset_create_dispatches_methyl_store(monkeypatch, tmp_path):
    calls: list[dict] = []

    def fake_from_files(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("quantnado.dataset.store_methyl.MethylStore.from_files", fake_from_files)

    bam = tmp_path / "sample.bam"
    methyl = tmp_path / "sample.bedGraph"
    bam.write_text("")
    methyl.write_text("")

    result = runner.invoke(
        app,
        [
            "dataset",
            "create",
            "--sample",
            "METH_1",
            "--assay",
            "METH",
            "--bamfile",
            str(bam),
            "--methylation_file",
            str(methyl),
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 0
    assert len(calls) == 1
    assert calls[0]["bam_path"] == str(bam)
    assert calls[0]["methyl_path"] == str(methyl)
    assert calls[0]["sample"] == "METH_1"


def test_dataset_create_dispatches_variant_store(monkeypatch, tmp_path):
    calls: list[dict] = []

    def fake_from_vcf(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("quantnado.dataset.store_variants.VariantStore.from_vcf", fake_from_vcf)

    vcf = tmp_path / "sample.vcf.gz"
    vcf.write_text("")

    result = runner.invoke(
        app,
        [
            "dataset",
            "create",
            "--sample",
            "SNP_1",
            "--assay",
            "SNP",
            "--vcf_file",
            str(vcf),
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 0
    assert len(calls) == 1
    assert calls[0]["vcf_path"] == str(vcf)
    assert calls[0]["sample"] == "SNP_1"


def test_dataset_combine_accepts_single_flag_store_list(monkeypatch, tmp_path):
    stores = []
    for name in ["ATAC_1.zarr", "RNA_1.zarr", "ChIP_1.zarr"]:
        path = tmp_path / name
        path.mkdir()
        stores.append(path)

    seen: dict[str, object] = {}

    @classmethod
    def fake_combine(cls, src, output, overwrite=True, n_workers=1):
        src_path = Path(src)
        seen["stores"] = sorted(p.name for p in src_path.glob("*.zarr"))
        seen["output"] = Path(output)
        seen["overwrite"] = overwrite
        seen["n_workers"] = n_workers
        return None

    monkeypatch.setattr("quantnado.analysis.core.QuantNadoDataset.combine", fake_combine)

    result = runner.invoke(
        app,
        [
            "dataset",
            "combine",
            "--stores",
            str(stores[0]),
            str(stores[1]),
            str(stores[2]),
            "--output",
            str(tmp_path / "combined.zarr"),
            "--workers",
            "2",
        ],
    )

    assert result.exit_code == 0
    assert seen["stores"] == ["ATAC_1.zarr", "ChIP_1.zarr", "RNA_1.zarr"]
    assert seen["output"] == tmp_path / "combined.zarr"
    assert seen["n_workers"] == 2


def test_dataset_compress_uses_pigz_workers(monkeypatch, tmp_path):
    dataset = tmp_path / "combined.zarr"
    dataset.mkdir()
    (dataset / "zarr.json").write_text("{}")
    output = tmp_path / "combined.zarr.gz"
    seen: dict[str, list[str]] = {}

    def fake_which(name: str) -> str | None:
        if name in {"tar", "pigz"}:
            return f"/usr/bin/{name}"
        return None

    def fake_run(cmd, stdout, stderr, text, check):
        seen["cmd"] = cmd
        archive = Path(cmd[cmd.index("-cf") + 1])
        archive.write_bytes(b"archive")
        return subprocess.CompletedProcess(cmd, 0, "")

    monkeypatch.setattr("quantnado.cli.shutil.which", fake_which)
    monkeypatch.setattr("quantnado.cli.subprocess.run", fake_run)

    result = runner.invoke(
        app,
        [
            "dataset",
            "compress",
            "--dataset",
            str(dataset),
            "--output",
            str(output),
            "--workers",
            "8",
        ],
    )

    assert result.exit_code == 0
    cmd = seen["cmd"]
    assert cmd[:4] == ["/usr/bin/tar", "-I", "pigz -p 8", "-cf"]
    assert cmd[4] == str(output)
    assert cmd[-3:] == ["-C", str(tmp_path), "combined.zarr"]


def test_dataset_compress_requires_pigz_for_parallel_workers(monkeypatch, tmp_path):
    dataset = tmp_path / "combined.zarr"
    dataset.mkdir()
    log_file = tmp_path / "compress.log"

    def fake_which(name: str) -> str | None:
        if name == "tar":
            return "/usr/bin/tar"
        return None

    monkeypatch.setattr("quantnado.cli.shutil.which", fake_which)

    result = runner.invoke(
        app,
        [
            "dataset",
            "compress",
            "--dataset",
            str(dataset),
            "--workers",
            "2",
            "--log-file",
            str(log_file),
        ],
    )

    assert result.exit_code == 1
    assert "pigz" in _clean(log_file.read_text())

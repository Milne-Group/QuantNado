from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from quantnado.cli import app


runner = CliRunner()


def test_dataset_create_help_mentions_direct_inputs():
    result = runner.invoke(app, ["dataset", "create", "--help"])
    assert result.exit_code == 0
    assert "--bamfile" in result.stdout
    assert "--vcf_file" in result.stdout
    assert "--methylation" in result.stdout


def test_dataset_combine_help_mentions_stores_list():
    result = runner.invoke(app, ["dataset", "combine", "--help"])
    assert result.exit_code == 0
    assert "--stores" in result.stdout
    assert "single" in result.stdout
    assert "flag" in result.stdout


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
    def fake_combine(cls, src, output, overwrite=True):
        src_path = Path(src)
        seen["stores"] = sorted(p.name for p in src_path.glob("*.zarr"))
        seen["output"] = Path(output)
        seen["overwrite"] = overwrite
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
        ],
    )

    assert result.exit_code == 0
    assert seen["stores"] == ["ATAC_1.zarr", "ChIP_1.zarr", "RNA_1.zarr"]
    assert seen["output"] == tmp_path / "combined.zarr"

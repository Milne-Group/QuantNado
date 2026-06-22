from __future__ import annotations

from quantnado.api import create_dataset


def test_create_dataset_forwards_single_end_to_bam_store(monkeypatch, tmp_path):
    calls: list[dict] = []

    def fake_from_bam_files(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("quantnado.dataset.store_bam.BamStore.from_bam_files", fake_from_bam_files)

    output_path = tmp_path / "rna.zarr"
    result = create_dataset(
        "RNA_1",
        "RNA",
        output_path,
        bam_path="dummy.bam",
        stranded="R",
        paired=False,
    )

    assert result == output_path
    assert len(calls) == 1
    assert calls[0]["paired"] is False

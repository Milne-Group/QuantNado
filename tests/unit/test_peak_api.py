from __future__ import annotations

from pathlib import Path

from quantnado.analysis.core import QuantNadoDataset


def test_call_peaks_resolves_assay_types_to_array_keys(monkeypatch, tmp_path):
    calls = []

    def fake_quantile(zarr_path, output_dir, assay=None, **kwargs):
        calls.append((Path(zarr_path), Path(output_dir), assay))
        return [str(Path(output_dir) / f"{assay}.bed")]

    monkeypatch.setattr(
        "quantnado.peak_calling.call_quantile_peaks.call_quantile_peaks_from_zarr",
        fake_quantile,
    )

    class FakeQN:
        call_peaks = QuantNadoDataset.call_peaks

        def __init__(self):
            self.path = tmp_path / "dataset.zarr"
            self._combined = True
            self._combined_root = type(
                "Root",
                (),
                {
                    "attrs": {
                        "key_to_samples": {
                            "atac": ["ATAC-1"],
                            "chip_mll": ["ChIP-MLL-1"],
                            "chip_h3k27ac": ["ChIP-H3K27ac-1"],
                        }
                    }
                },
            )()
            self._stores = []

        @property
        def sample_names(self):
            return ["ATAC-1", "ChIP-MLL-1", "ChIP-H3K27ac-1"]

        @property
        def assays(self):
            return ["ATAC", "CHIP"]

        @property
        def array_keys(self):
            return ["atac", "chip_h3k27ac", "chip_mll"]

        def _get_assay_per_sample(self):
            return ["ATAC", "CHIP", "CHIP"]

        def _resolve_samples(self, assay=None, samples=None):
            if samples is not None:
                requested = [samples] if isinstance(samples, str) else list(samples)
                return [s for s in requested if s in self.sample_names]
            if assay is not None:
                assay_upper = {assay} if isinstance(assay, str) else {str(a).upper() for a in assay}
                return [
                    s for s, a in zip(self.sample_names, self._get_assay_per_sample())
                    if a.upper() in assay_upper
                ]
            return list(self.sample_names)

    qn = FakeQN()

    beds = qn.call_peaks(tmp_path / "peaks", method="quantile", assay=["ATAC", "CHIP"])

    assert set(call[2] for call in calls) == {"atac", "chip_mll", "chip_h3k27ac"}
    assert set(call[1] for call in calls) == {
        tmp_path / "peaks" / "atac",
        tmp_path / "peaks" / "chip_mll",
        tmp_path / "peaks" / "chip_h3k27ac",
    }
    assert set(beds) == {"atac", "chip_h3k27ac", "chip_mll"}


def test_call_peaks_uses_collapsed_coverage_with_sample_subsets(monkeypatch, tmp_path):
    calls = []

    def fake_quantile(zarr_path, output_dir, assay=None, samples=None, **kwargs):
        calls.append((Path(zarr_path), Path(output_dir), assay, tuple(samples or [])))
        label = output_dir.name
        return [str(Path(output_dir) / f"{label}.bed")]

    monkeypatch.setattr(
        "quantnado.peak_calling.call_quantile_peaks.call_quantile_peaks_from_zarr",
        fake_quantile,
    )

    class FakeQN:
        call_peaks = QuantNadoDataset.call_peaks

        def __init__(self):
            self.path = tmp_path / "dataset.zarr"
            self._combined = True
            self._combined_root = type(
                "Root",
                (),
                {
                    "attrs": {
                        "key_to_samples": {
                            "atac": ["atac"],
                            "chip": ["chip-rx_MLL"],
                            "coverage": ["atac", "chip-rx_MLL", "rna-1"],
                        }
                    }
                },
            )()
            self._stores = []

        @property
        def sample_names(self):
            return ["atac", "chip-rx_MLL", "rna-1"]

        @property
        def assays(self):
            return ["ATAC", "CHIP", "RNA"]

        @property
        def array_keys(self):
            return ["coverage", "rna_fwd", "rna_rev"]

        def _get_assay_per_sample(self):
            return ["ATAC", "CHIP", "RNA"]

        def _resolve_samples(self, assay=None, samples=None):
            if samples is not None:
                requested = [samples] if isinstance(samples, str) else list(samples)
                return [s for s in requested if s in self.sample_names]
            if assay is not None:
                assay_upper = {assay} if isinstance(assay, str) else {str(a).upper() for a in assay}
                return [
                    s for s, a in zip(self.sample_names, self._get_assay_per_sample())
                    if a.upper() in assay_upper
                ]
            return list(self.sample_names)

    qn = FakeQN()

    beds = qn.call_peaks(tmp_path / "peaks", method="quantile", assay=["ATAC", "CHIP"])

    assert calls == [
        (tmp_path / "dataset.zarr", tmp_path / "peaks" / "atac", "coverage", ("atac",)),
        (tmp_path / "dataset.zarr", tmp_path / "peaks" / "chip", "coverage", ("chip-rx_MLL",)),
    ]
    assert set(beds) == {"atac", "chip"}


def test_call_peaks_supports_multiple_methods(monkeypatch, tmp_path):
    calls = []

    def fake_quantile(zarr_path, output_dir, assay=None, samples=None, **kwargs):
        calls.append(("quantile", Path(output_dir), assay, tuple(samples or [])))
        return [str(Path(output_dir) / f"{Path(output_dir).name}.bed")]

    def fake_lanceotron(zarr_path, output_dir, assay=None, samples=None, **kwargs):
        calls.append(("lanceotron", Path(output_dir), assay, tuple(samples or [])))
        return [str(Path(output_dir) / f"{Path(output_dir).name}.bed")]

    monkeypatch.setattr(
        "quantnado.peak_calling.call_quantile_peaks.call_quantile_peaks_from_zarr",
        fake_quantile,
    )
    monkeypatch.setattr(
        "quantnado.peak_calling.call_lanceotron_peaks.call_lanceotron_peaks_from_zarr",
        fake_lanceotron,
    )

    class FakeQN:
        call_peaks = QuantNadoDataset.call_peaks
        available_peak_methods = QuantNadoDataset.available_peak_methods

        def __init__(self):
            self.path = tmp_path / "dataset.zarr"
            self._combined = True
            self._combined_root = type(
                "Root",
                (),
                {"attrs": {"key_to_samples": {"coverage": ["atac", "chip-rx_MLL"]}}},
            )()
            self._stores = []

        @property
        def sample_names(self):
            return ["atac", "chip-rx_MLL"]

        @property
        def assays(self):
            return ["ATAC", "CHIP"]

        @property
        def array_keys(self):
            return ["coverage"]

        def _get_assay_per_sample(self):
            return ["ATAC", "CHIP"]

        def _resolve_samples(self, assay=None, samples=None, ip=None, group=None):
            if samples is not None:
                requested = [samples] if isinstance(samples, str) else list(samples)
                return [s for s in requested if s in self.sample_names]
            if assay is not None:
                assay_upper = {assay} if isinstance(assay, str) else {str(a).upper() for a in assay}
                return [
                    s for s, a in zip(self.sample_names, self._get_assay_per_sample())
                    if a.upper() in assay_upper
                ]
            return list(self.sample_names)

    qn = FakeQN()
    assert qn.available_peak_methods == ["quantile", "seacr", "lanceotron"]

    beds = qn.call_peaks(
        tmp_path / "peaks",
        method=["quantile", "lanceotron"],
        assay=["ATAC", "CHIP"],
    )

    assert set(beds) == {"quantile", "lanceotron"}
    assert set(beds["quantile"]) == {"atac", "chip"}
    assert set(beds["lanceotron"]) == {"atac", "chip"}
    assert set((method, output_dir.parts[-2], output_dir.parts[-1]) for method, output_dir, *_ in calls) == {
        ("quantile", "quantile", "atac"),
        ("quantile", "quantile", "chip"),
        ("lanceotron", "lanceotron", "atac"),
        ("lanceotron", "lanceotron", "chip"),
    }

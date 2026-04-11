# Command-Line Interface

QuantNado exposes two main commands:

| Command | Purpose |
|---|---|
| [`create-dataset`](create_dataset.md) | Build per-sample `.zarr` stores from a metadata CSV/TSV |
| [`call-peaks`](call_peaks.md) | Call peaks from a QuantNado dataset |

## Global Help

```bash
quantnado --help
quantnado --version
```

## Current CLI Model

- `create-dataset` is metadata-driven
- outputs are written as one `.zarr` store per sample
- test-mode builds can use either the default chromosome trio or repeated `--test-chrom` values
- `call-peaks` works from QuantNado zarr inputs, not from bigWig directories

## Typical Workflow

```bash
quantnado create-dataset \
  --metadata samples.csv \
  --output-dir dataset

quantnado call-peaks \
  --zarr dataset \
  --method quantile \
  --output-dir peaks
```

## More Detail

- [Create Dataset](create_dataset.md)
- [Call Peaks](call_peaks.md)
- [Legacy-style CLI summary](../cli.md)

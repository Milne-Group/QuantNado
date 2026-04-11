# Command-Line Interface

QuantNado exposes a dataset command group plus `call-peaks`:

| Command | Purpose |
|---|---|
| [`dataset create`](create_dataset.md) | Build one per-sample `.zarr` store from direct assay inputs |
| [`dataset combine`](combine_dataset.md) | Merge per-sample `.zarr` stores into one multi-sample dataset |
| [`call-peaks`](call_peaks.md) | Call peaks from a QuantNado dataset |

## Global Help

```bash
quantnado --help
quantnado --version
```

## Current CLI Model

- `dataset create` is per-sample and input-file-driven
- outputs are written as one `.zarr` store per sample
- `dataset combine` accepts one `--stores` flag followed by a list of store paths
- test-mode builds can use either the default chromosome trio or repeated `--test-chrom` values
- `call-peaks` works from QuantNado zarr inputs, not from bigWig directories

## Typical Workflow

```bash
quantnado dataset create \
  --sample ATAC_1 \
  --assay ATAC \
  --bamfile /data/ATAC_1.bam \
  --output-dir dataset

quantnado dataset combine \
  --stores dataset/ATAC_1.zarr dataset/RNA_1.zarr \
  --output dataset/combined.zarr

quantnado call-peaks \
  --zarr dataset/combined.zarr \
  --method quantile \
  --output-dir peaks
```

## More Detail

- [Create Dataset](create_dataset.md)
- [Combine Dataset](combine_dataset.md)
- [Call Peaks](call_peaks.md)
- [Legacy-style CLI summary](../cli.md)

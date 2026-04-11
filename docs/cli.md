# CLI Reference

This page is a compact summary of the current command-line interface.

## Root Command

```bash
quantnado --help
quantnado --version
```

## `quantnado create-dataset`

Build per-sample `.zarr` stores from a metadata CSV/TSV.

```bash
quantnado create-dataset \
  --metadata samples.csv \
  --output-dir dataset
```

Key points:

- metadata-driven, not positional-BAM-driven
- one output `.zarr` per sample
- supports `ATAC`, `ChIP`, `RNA`, `CUT&TAG`, `METH`, `SNP`, and `MCC`

Most important options:

- `--metadata`, `-m`
- `--output-dir`, `-o`
- `--chromsizes`
- `--overwrite`
- `--chunk-len`
- `--construction-compression`
- `--filter-chromosomes / --no-filter-chromosomes`
- `--test`
- `--test-chrom`

## `quantnado call-peaks`

Call peaks from a QuantNado store or store directory.

```bash
quantnado call-peaks \
  --zarr dataset \
  --method quantile \
  --output-dir peaks
```

Key points:

- uses QuantNado zarr inputs, not bigWig directories
- supports `quantile`, `seacr`, and `lanceotron`
- optional `--assay` lets you target a specific array key

Most important options:

- `--zarr`
- `--method`
- `--output-dir`
- `--assay`
- `--blacklist`
- `--n-workers`
- `--device`

## See Also

- [CLI Overview](cli/index.md)
- [Create Dataset](cli/create_dataset.md)
- [Call Peaks](cli/call_peaks.md)

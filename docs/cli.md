# CLI Reference

This page is a compact summary of the current command-line interface.

## Root Command

```bash
quantnado --help
quantnado --version
```

## `quantnado dataset create`

Build one per-sample `.zarr` store from direct assay inputs.

```bash
quantnado dataset create \
  --sample RNA_1 \
  --assay RNA \
  --bamfile /data/RNA_1.bam \
  --stranded R \
  --output-dir dataset
```

Key points:

- one command call per sample
- one output `.zarr` per sample
- supports `ATAC`, `ChIP`, `RNA`, `CUT&TAG`, `METH`, `SNP`, and `MCC`

Most important options:

- `--sample`
- `--assay`
- `--output-dir`, `-o`
- `--bamfile`
- `--vcf_file`
- `--methylation_file`
- `--ip`
- `--stranded`
- `--chromsizes`
- `--overwrite`
- `--chunk-len`
- `--construction-compression`
- `--filter-chromosomes / --no-filter-chromosomes`
- `--test`
- `--test-chrom`

## `quantnado dataset combine`

Combine per-sample stores into one multi-sample dataset.

```bash
quantnado dataset combine \
  --stores dataset/ATAC_1.zarr dataset/RNA_1.zarr dataset/METH_1.zarr \
  --output dataset/combined.zarr
```

Key points:

- accepts one `--stores` flag followed by a list of per-sample `.zarr` stores
- writes one combined multi-sample `.zarr`
- useful before sharing, reopening, or peak calling across many samples

Most important options:

- `--stores`
- `--output`, `-o`
- `--overwrite`
- `--log-file`
- `--verbose`

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
- [Combine Dataset](cli/combine_dataset.md)
- [Call Peaks](cli/call_peaks.md)

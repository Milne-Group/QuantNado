# `call-peaks`

Call peaks directly from a QuantNado dataset.

## Usage

```bash
quantnado call-peaks \
  --zarr dataset/combined.zarr \
  --output-dir peaks
```

## Required Options

- `--zarr`: path to a QuantNado store or a directory of per-sample stores
- `--output-dir`: directory for output BED files

## Methods

- `quantile`: fast default method
- `seacr`: SEACR-style peak calling with optional control store
- `lanceotron`: ML-based peak calling, typically for ChIP-seq

## Common Options

- `--method TEXT`: `quantile`, `seacr`, or `lanceotron`
- `--assay TEXT`: array key to call peaks on, such as `atac` or `chip_h3k27ac`
- `--blacklist PATH`: BED file of excluded regions
- `--log-file PATH`: log destination
- `--verbose`, `-v`: debug logging

## Quantile Options

- `--tilesize INTEGER`
- `--window-overlap INTEGER`
- `--quantile FLOAT`
- `--merge / --no-merge`

Example:

```bash
quantnado call-peaks \
  --zarr dataset \
  --method quantile \
  --assay atac \
  --quantile 0.98 \
  --output-dir peaks
```

## SEACR Options

- `--control-zarr PATH`
- `--fdr FLOAT`
- `--norm TEXT`
- `--stringency TEXT`
- `--n-workers INTEGER`
- `--device TEXT`

Example:

```bash
quantnado call-peaks \
  --zarr chip_dataset \
  --method seacr \
  --assay chip_h3k27ac \
  --control-zarr igg_dataset \
  --output-dir peaks_seacr
```

## LanceOTron Options

- `--score-threshold FLOAT`
- `--smooth-window INTEGER`
- `--batch-size INTEGER`
- `--n-workers INTEGER`
- `--device TEXT`

Example:

```bash
quantnado call-peaks \
  --zarr chip_dataset \
  --method lanceotron \
  --assay chip_h3k27ac \
  --output-dir peaks_lanceotron
```

## Output

QuantNado writes BED files to the requested output directory, one result per eligible sample.

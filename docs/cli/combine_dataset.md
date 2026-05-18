# `dataset combine`

Combine per-sample QuantNado `.zarr` stores into a single multi-sample dataset.

## Usage

```bash
quantnado dataset combine \
  --stores dataset/ATAC_1.zarr dataset/RNA_1.zarr dataset/METH_1.zarr \
  --output dataset/combined.zarr
```

## Required Options

- `--stores`: one or more per-sample `.zarr` stores after a single flag
- `--output`, `-o`: path to the combined multi-sample `.zarr`

## Options

- `--overwrite / --no-overwrite`: replace the output store if it already exists
- `--workers`, `--n-workers`: number of thread workers to use while copying rows
- `--log-file PATH`: log destination
- `--verbose`, `-v`: debug logging

## Notes

- `dataset combine` accepts either:
  - one `--stores` flag followed by a list of store paths, or
  - a single directory path after `--stores` if you want to combine every `.zarr` store in that directory
- incomplete stores are skipped by the combine engine
- the combined store can be opened with the same `QuantNado` / `QuantNadoDataset` API

## Example

```bash
quantnado dataset combine \
  --stores samples/ATAC_1.zarr samples/H3K27ac_1.zarr samples/RNA_1.zarr \
  --output samples/combined.zarr \
  --workers 4
```

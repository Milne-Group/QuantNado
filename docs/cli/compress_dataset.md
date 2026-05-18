# `dataset compress`

Archive a QuantNado dataset directory or combined `.zarr` store as a tar.gz file.

The archive can be opened directly with `QuantNado.open(...)`.

## Usage

```bash
quantnado dataset compress \
  --dataset dataset/combined.zarr \
  --output dataset/combined.zarr.gz \
  --workers 8
```

## Required Options

- `--dataset`, `--input`: QuantNado dataset directory or combined `.zarr` store

## Options

- `--output`, `-o`: output archive path; defaults to `<dataset>.gz`
- `--overwrite / --no-overwrite`: replace the output archive if it already exists
- `--workers`, `--n-workers`: number of `pigz` workers
- `--log-file PATH`: log destination
- `--verbose`, `-v`: debug logging

## Notes

- with `--workers > 1`, `pigz` must be available in the runtime container
- with `--workers 1`, QuantNado can fall back to standard gzip if `pigz` is unavailable
- the output archive is a tar.gz archive even if the filename ends in `.zarr.gz`
- write the archive next to the dataset, not inside the dataset directory

## Snakemake Example

```python
shell:
    """
    quantnado dataset compress \
      --dataset {input.dataset:q} \
      --output {output.zipped_dataset:q} \
      --workers {threads} \
      --log-file {log:q}
    """
```

# Troubleshooting

## Install Problems

### Python version errors

QuantNado currently supports Python 3.12 and 3.13.

```bash
python --version
```

If you are on 3.14+, create a 3.12 or 3.13 environment and reinstall.

### Import errors

```bash
python -c "import quantnado; print(quantnado.__version__)"
```

If that fails:

```bash
pip install quantnado
```

Or from source:

```bash
pip install -e .
```

## Dataset Creation Problems

### BAM index missing

QuantNado expects indexed BAM files.

```bash
samtools index sample.bam
```

### Metadata validation errors

Check that your metadata file has:

- `sample`
- `assay`

And the assay-specific input columns required for each row:

- `bam` for BAM-based assays
- `bam` and `methyl` for `METH`
- `vcf` for `SNP`

### Output already exists

If the target store already exists and you want to replace it:

```bash
quantnado create-dataset \
  --metadata samples.csv \
  --output-dir dataset \
  --overwrite
```

### Chromosome naming mismatches

If your intervals, annotation, and data use different contig naming styles such as `chr21` vs `21`, reduction and extraction may appear empty. Make sure BED/GTF inputs and the underlying stores use matching chromosome names.

## Open / Analysis Problems

### Opening a dataset fails

Confirm you are pointing at either:

- a directory containing per-sample `.zarr` stores, or
- a combined `.zarr` written by `QuantNado.combine(...)`

### Empty assay selection

If `assay="RNA"` or similar returns no samples, inspect:

```python
qn.assays
qn.sample_names
qn.array_keys
```

Remember:

- `assay` is the biological type
- `modality` is the array key

### No data for a requested modality

Inspect available array keys first:

```python
qn.array_keys
```

Then pass a valid key such as `coverage`, `rna_fwd`, `chip_h3k27ac`, or `methyl_pct`.

## Peak Calling Problems

### No peaks are called

Start by loosening thresholds or checking the chosen assay key:

```bash
quantnado call-peaks \
  --zarr dataset \
  --method quantile \
  --assay atac \
  --quantile 0.95 \
  --output-dir peaks
```

Also verify that the selected store actually contains the requested assay key.

### `lanceotron` is unavailable

Install the optional dependency set:

```bash
pip install "quantnado[lanceotron]"
```

## Performance Problems

### Dataset creation is slow

Things to check:

- BAM size and sequencing depth
- filesystem performance
- whether chunk sizing or compression needs tuning for your environment

Useful flags:

```bash
quantnado create-dataset \
  --metadata samples.csv \
  --output-dir dataset \
  --construction-compression fast \
  --chunk-len 131072
```

### Memory pressure during analysis

Most of the API is xarray/dask-backed. Prefer selecting smaller regions, restricting assays or samples, and working chromosome-by-chromosome when exploring large datasets.

## Still Stuck?

- Check [Usage Guide](basic_usage.md)
- Check [CLI Reference](cli.md)
- Check [API Reference](api/quantnado.md)
- Open an issue at [GitHub](https://github.com/Milne-Group/QuantNado/issues)

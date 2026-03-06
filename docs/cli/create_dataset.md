# Create Dataset Command

Convert BAM files into a Zarr dataset for efficient analysis.

## Usage

```bash
quantnado create-dataset [OPTIONS] BAM_FILES...
```

## Description

The `create-dataset` command processes one or more BAM files and creates a Zarr-backed dataset for efficient genomic signal analysis. This is the first step in most QuantNado workflows.

## Arguments

### BAM Files (Required)

Paths to BAM files to process (one or more):

```bash
quantnado create-dataset sample1.bam sample2.bam sample3.bam --output dataset.zarr
```

BAM files must be:
- Coordinate-sorted
- Indexed (.bai file in same directory)
- Valid BAM format

## Options

### Output (`--output`, `-o`) {: #output}

**Required.** Path where the Zarr dataset will be saved:

```bash
quantnado create-dataset *.bam --output dataset.zarr
```

Creates a directory `dataset.zarr/` containing the processed data.

### Chromsizes (`--chromsizes`)

Path to chromsizes file or auto-detect from first BAM:

```bash
# Specify explicit chromsizes
quantnado create-dataset *.bam --output dataset.zarr --chromsizes hg38.chrom.sizes

# Auto-detect from BAM (default)
quantnado create-dataset *.bam --output dataset.zarr
```

Chromsizes file format (tab-separated):
```
chr1    248956422
chr2    242193529
...
```

### Metadata (`--metadata`)

Path to metadata CSV file:

```bash
quantnado create-dataset *.bam --output dataset.zarr --metadata samples.csv
```

CSV format with a sample-matching column:
```
sample_id,condition,replicate,quality
sample1,control,1,high
sample2,control,2,high
sample3,treatment,1,high
```

Metadata rules:
- By default, QuantNado matches metadata rows using the `sample_id` column.
- Use `--sample-column` if your metadata uses a different column name.
- Sample names default to BAM filename stems, or can be set explicitly with repeated `--sample-name` options.
- Metadata can contain any additional columns; all of them are stored.
- Metadata rows may be a subset of samples; missing values are stored as empty.
- An optional `sample_hash` column is accepted for validating that metadata matches the processed BAMs.

### Sample Names (`--sample-name`)

Provide one explicit sample name per BAM file, in the same order as `BAM_FILES`:

```bash
quantnado create-dataset sample1.bam sample2.bam \
  --sample-name ATAC \
  --sample-name H3K27ac \
  --output dataset.zarr
```

This is useful when BAM filenames are not the labels you want to carry through into the dataset and metadata.

### Metadata Sample Column (`--sample-column`)

Use a different metadata column to match rows to the dataset sample names:

```bash
quantnado create-dataset sample1.bam sample2.bam \
  --sample-name ATAC \
  --sample-name H3K27ac \
  --metadata samples.csv \
  --sample-column assay_label \
  --output dataset.zarr
```

### Max Workers (`--max-workers`)

Number of parallel worker threads (default: 1):

```bash
# Use more workers for faster processing (uses more memory)
quantnado create-dataset *.bam --output dataset.zarr --max-workers 8

# Use single worker for memory efficiency
quantnado create-dataset *.bam --output dataset.zarr --max-workers 1
```

### Overwrite (`--overwrite`)

Overwrite existing dataset at same path (default: no overwrite):

```bash
quantnado create-dataset *.bam --output dataset.zarr --overwrite
```

### Log File (`--log-file`)

Path to save processing logs (default: `quantnado_processing.log`):

```bash
quantnado create-dataset *.bam --output dataset.zarr --log-file processing.log
```

### Verbose (`--verbose`, `-v`)

Enable debug logging:

```bash
quantnado create-dataset *.bam --output dataset.zarr --verbose
```

## Examples

### Basic Usage

Create dataset from BAM files:

```bash
quantnado create-dataset sample1.bam sample2.bam sample3.bam \
  --output my_dataset.zarr
```

### With Chromsizes

Specify explicit chromsizes file:

```bash
quantnado create-dataset *.bam \
  --output dataset.zarr \
  --chromsizes /reference/hg38.chrom.sizes
```

### With Metadata

Include sample metadata:

```bash
quantnado create-dataset *.bam \
  --output dataset.zarr \
  --metadata samples.csv
```

### With Explicit Sample Names From a Notebook Mapping

If you already have a mapping like `{label: bam_path}`, pass the BAM paths positionally and the labels with repeated `--sample-name` flags in the same order:

```bash
quantnado create-dataset atac.bam h3k27ac.bam myb.bam \
  --sample-name ATAC \
  --sample-name H3K27ac \
  --sample-name MYB \
  --output dataset.zarr
```

### Parallel Processing

Use multiple workers for faster processing:

```bash
quantnado create-dataset *.bam \
  --output dataset.zarr \
  --max-workers 8 \
  --verbose
```

### Resume Processing

Resume interrupted dataset creation by passing `--resume`. Without this flag, re-running the
command will fail if the store already exists (use `--overwrite` to replace it instead):

```bash
# Resume from where processing left off (skips completed samples)
quantnado create-dataset *.bam --output dataset.zarr --resume

# Or start fresh, overwriting the existing store
quantnado create-dataset *.bam --output dataset.zarr --overwrite
```

## Performance

### Typical Run Times

Creation time depends on sequencing depth:

| Read Count | Time (single sample) |
|------------|----------------------|
| 5M reads | 1-2 minutes |
| 50M reads | 5-10 minutes |
| 100M+ reads | 15-30 minutes |

### Output Size

Zarr dataset size approximates BAM file size:

| Read Depth | Size |
|-----------|------|
| 5M reads | 500 MB |
| 50M reads | 5 GB |
| 100M+ reads | 10+ GB |

## Troubleshooting

### BAM file errors

**Problem**: `FileNotFoundError: BAM file not found`

**Solution**: Verify path and ensure .bai index exists:
```bash
ls -l sample1.bam sample1.bam.bai
samtools index sample1.bam
```

### Chromsizes errors

**Problem**: `ValueError: chromsizes_dict appears empty`

**Solution**: Provide explicit chromsizes file:
```bash
quantnado create-dataset *.bam --output dataset.zarr --chromsizes hg38.chrom.sizes
```

### Memory issues

**Problem**: Out of memory during processing

**Solution**: Use fewer workers:
```bash
quantnado create-dataset *.bam --output dataset.zarr --max-workers 1
```

### Slow processing

**Problem**: Dataset creation is very slow

**Solution**: Use more workers (if memory available):
```bash
quantnado create-dataset *.bam --output dataset.zarr --max-workers 8
```

## Python Equivalent

The CLI `create-dataset` command is equivalent to:

```python
from quantnado import QuantNado
import pandas as pd

qn = QuantNado.from_bam_files(
    bam_files=["sample1.bam", "sample2.bam", "sample3.bam"],
    store_path="dataset.zarr",
    chromsizes="hg38.chrom.sizes",
    metadata="samples.csv",
  sample_names=["ATAC", "H3K27ac", "MYB"],
    overwrite=True
)
```

See [Dataset Creation Guide](../basic_usage.md) for Python API details.

## See Also

- [Call Peaks Command](call_peaks.md)
- [Dataset Creation Guide](../basic_usage.md)
- [Python API: QuantNado.from_bam_files](../api/quantnado.md)
- [Basic Usage](../basic_usage.md)

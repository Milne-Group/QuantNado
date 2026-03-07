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

CSV format with sample_id column:
```
sample_id,condition,replicate,quality
sample1,control,1,high
sample2,control,2,high
sample3,treatment,1,high
```

### Max Workers (`--max-workers`)

Number of parallel worker threads (default: 1):

```bash
# Use more workers for faster processing (uses more memory)
quantnado create-dataset *.bam --output dataset.zarr --max-workers 8

# Use single worker for memory efficiency
quantnado create-dataset *.bam --output dataset.zarr --max-workers 1
```

### Chunk Length (`--chunk-len`)

Override the position-axis Zarr chunk length. If omitted, QuantNado derives a
filesystem-aware default from the output path so network filesystems like CephFS
use much larger write units than local disks.

```bash
# Let QuantNado auto-select a chunk length from the target filesystem
quantnado create-dataset *.bam --output dataset.zarr

# Pin an explicit chunk length for benchmarking or reproducibility
quantnado create-dataset *.bam --output dataset.zarr --chunk-len 131072
```

### Construction Compression (`--construction-compression`)

Control build-time compression separately from the on-disk dataset layout.
This is useful when benchmarking CephFS write throughput, where lower
compression overhead or fully uncompressed construction may outperform the
default profile.

```bash
# Current default profile
quantnado create-dataset *.bam --output dataset.zarr --construction-compression default

# Lower zstd compression overhead
quantnado create-dataset *.bam --output dataset.zarr --construction-compression fast

# Uncompressed construction writes
quantnado create-dataset *.bam --output dataset.zarr --construction-compression none
```

### Local Staging (`--local-staging`, `--staging-dir`)

Build the dataset under local scratch storage and only publish to the final
output path after construction succeeds. This is useful on CephFS-backed
clusters because it converts many incremental writes into one finalize step.

```bash
# Use TMPDIR-backed scratch staging
quantnado create-dataset *.bam \
  --output /ceph/project/dataset.zarr \
  --local-staging \
  --staging-dir "$TMPDIR"

# Let QuantNado choose the system temporary directory when staging is enabled
quantnado create-dataset *.bam \
  --output /ceph/project/dataset.zarr \
  --local-staging
```

`--local-staging` is opt-in. `--staging-dir` can also be supplied directly to
pick a specific scratch filesystem.

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
    overwrite=True
)
```

See [Dataset Creation Guide](../basic_usage.md) for Python API details.

## See Also

- [Call Peaks Command](call_peaks.md)
- [Dataset Creation Guide](../basic_usage.md)
- [Python API: QuantNado.from_bam_files](../api/quantnado.md)
- [Basic Usage](../basic_usage.md)

# Command-Line Interface

QuantNado provides simple, intuitive command-line tools for processing genomic data.

## Main Commands

### `quantnado --version`

Display installed version:

```bash
quantnado --version
```

### `quantnado --help`

Show all available commands:

```bash
quantnado --help
```

## create-dataset

Convert BAM files into an indexed Zarr dataset for fast access.

**Basic Usage:**

```bash
quantnado create-dataset sample1.bam sample2.bam sample3.bam \
  --output dataset.zarr \
  --chromsizes hg38.chrom.sizes
```

**Positional Arguments:**

- `BAM_FILES...` - One or more BAM files (required, must be indexed with `.bai`)

**Options:**

- `--output, -o PATH` - Output Zarr dataset path (required)
- `--chromsizes PATH` - Path to chromsizes file (optional; auto-detected from first BAM if omitted)
- `--metadata PATH` - Path to metadata CSV file (optional)
- `--chunk-len N` - Override the position-axis chunk length; if omitted, QuantNado derives a filesystem-aware default
- `--construction-compression PROFILE` - Build-time compression profile: `default`, `fast`, or `none`
- `--local-staging` - Build under local scratch storage and publish to the output path after completion
- `--staging-dir PATH` - Scratch directory to use for local staging; defaults to `TMPDIR` when local staging is enabled
- `--max-workers N` - Number of parallel threads (default: 1, recommended: 2-16)
- `--verbose, -v` - Enable debug logging
- `--overwrite` - Overwrite existing dataset if it exists
- `--log-file PATH` - Path to log file (default: `quantnado_processing.log`)

**Examples:**

```bash
# Basic: single BAM file
quantnado create-dataset sample.bam \
  --output sample.zarr \
  --chromsizes hg38.chrom.sizes

# Multiple BAM files with parallelization
quantnado create-dataset *.bam \
  --output my_cohort.zarr \
  --chromsizes hg38.chrom.sizes \
  --max-workers 8 \
  --verbose

# Override chunk sizing explicitly
quantnado create-dataset *.bam \
  --output my_cohort.zarr \
  --chromsizes hg38.chrom.sizes \
  --chunk-len 131072

# Stage on local scratch, then publish to the final store path
quantnado create-dataset *.bam \
  --output /ceph/project/cohort.zarr \
  --local-staging \
  --staging-dir "$TMPDIR"

# Use a faster build-time compression profile
quantnado create-dataset *.bam \
  --output /ceph/project/cohort.zarr \
  --construction-compression fast

# With metadata
quantnado create-dataset *.bam \
  --output dataset.zarr \
  --chromsizes hg38.chrom.sizes \
  --metadata samples.csv
```

## call-peaks

Call peaks using quantile-based detection on bigWig signal tracks.

**Basic Usage:**

```bash
quantnado call-peaks \
  --bigwig-dir ./signal_tracks/ \
  --output-dir ./peaks/ \
  --chromsizes hg38.chrom.sizes
```

**Options:**

- `--bigwig-dir PATH` - Directory containing bigWig files (required)
- `--output-dir PATH` - Output directory for BED files (required)
- `--chromsizes PATH` - Path to chromsizes file (required)
- `--blacklist PATH` - Path to BED file with regions to exclude (optional)
- `--quantile FLOAT` - Quantile threshold for peak calling (default: 0.98, range: 0.0-1.0)
- `--merge` - Merge overlapping peaks after calling
- `--tilesize INT` - Size of genomic tiles to use for calling (default: 128)
- `--tmp-dir PATH` - Temporary directory for intermediate files
- `--verbose, -v` - Enable debug logging
- `--log-file PATH` - Path to log file (default: `quantnado_peaks.log`)

**Examples:**

```bash
# Basic peak calling
quantnado call-peaks \
  --bigwig-dir ./bigwigs/ \
  --output-dir ./peaks/ \
  --chromsizes hg38.chrom.sizes

# Stringent threshold with merging
quantnado call-peaks \
  --bigwig-dir ./signals/ \
  --output-dir ./peaks_merged/ \
  --chromsizes hg38.chrom.sizes \
  --quantile 0.98 \
  --merge

# With blacklist regions
quantnado call-peaks \
  --bigwig-dir ./signals/ \
  --output-dir ./peaks/ \
  --chromsizes hg38.chrom.sizes \
  --blacklist hg38.blacklist.bed \
  --quantile 0.95
```

## Getting Help

Display help for any command:

```bash
quantnado --help
quantnado create-dataset --help
quantnado call-peaks --help
```
- `--metadata` - Path to metadata CSV file (optional)
- `--max-workers` - Number of parallel threads (default: 4)
- `--log-file` - Output log file path (default: `quantnado_processing.log`)
- `--verbose, -v` - Enable debug logging
- `--overwrite` - Overwrite existing dataset if it exists

**Example with metadata:**

```bash
quantnado create-dataset sample1.bam sample2.bam \
  --output dataset.zarr \
  --metadata samples.csv \
  --chromsizes hg38.chrom.sizes
```

## Peak Calling

### Call Quantile-Based Peaks

Call peaks from bigWig files using quantile thresholding:

```bash
quantnado call-peaks \
  --bigwig-dir ./bigwigs/ \
  --output-dir ./peaks/ \
  --chromsizes hg38.chrom.sizes \
  --quantile 0.98
```

**Options:**

- `--bigwig-dir` - Directory containing bigWig files (required)
- `--output-dir` - Directory to save peak BED files (required)
- `--chromsizes` - Path to chromsizes file (required)
- `--blacklist` - Path to blacklist BED file (optional)
- `--tilesize` - Genomic tile size for peak calling (default: 128 bp)
- `--quantile` - Quantile threshold for peak calling (default: 0.98)
- `--merge` - Merge overlapping peaks (flag)
- `--tmp-dir` - Temporary directory for intermediate files (default: `tmp`)
- `--log-file` - Output log file (default: `quantnado_peaks.log`)
- `--verbose, -v` - Enable debug logging

**Example with blacklist and merging:**

```bash
quantnado call-peaks \
  --bigwig-dir ./bigwigs/ \
  --output-dir ./peaks/ \
  --chromsizes hg38.chrom.sizes \
  --blacklist blacklist.bed \
  --quantile 0.95 \
  --merge
```

## Common Workflows

### Complete Pipeline

Process BAM files and call peaks:

```bash
# Step 1: Create Zarr dataset
quantnado create-dataset *.bam \
  --output dataset.zarr \
  --chromsizes hg38.chrom.sizes

# Step 2: Convert to bigWig (using external tools)
# for bam in *.bam; do
#   bedtools genomecov -ibam $bam -bg -o ${bam%.bam}.bedgraph
#   bedGraphToBigWig ${bam%.bam}.bedgraph hg38.chrom.sizes ${bam%.bam}.bw
# done

# Step 3: Call peaks
quantnado call-peaks \
  --bigwig-dir ./bigwigs/ \
  --output-dir ./peaks/ \
  --chromsizes hg38.chrom.sizes \
  --quantile 0.98
```

### Working with Multiple Datasets

You can process datasets in batch:

```bash
# Create multiple datasets
for dir in exp1 exp2 exp3; do
  quantnado create-dataset $dir/*.bam \
    --output $dir/dataset.zarr \
    --chromsizes hg38.chrom.sizes
done
```

## Troubleshooting

### Command Not Found

If `quantnado` is not found, ensure QuantNado is installed:

```bash
pip install quantnado
```

Or if installed in development mode, you may need to reinstall:

```bash
pip install -e .
```

### Memory Issues

For large datasets:

```bash
# Use fewer parallel workers
quantnado create-dataset *.bam \
  --output dataset.zarr \
  --max-workers 1
```

### Incomplete Processing

If a dataset creation is interrupted, you can resume:

```bash
# Continue from where it left off
quantnado create-dataset *.bam \
  --output dataset.zarr \
  --chromsizes hg38.chrom.sizes
  # (will resume incomplete samples)
```

## Environment Variables

### Python-specific

- `OMP_NUM_THREADS` - Limit OpenMP threads (recommended: 1)
- `KMP_WARNINGS` - Suppress OpenMP warnings

```bash
export OMP_NUM_THREADS=1
export KMP_WARNINGS=0
quantnado create-dataset *.bam --output dataset.zarr
```

## Getting Help

Get help for any command:

```bash
quantnado --help               # Main help
quantnado create-dataset --help  # Create dataset help
quantnado call-peaks --help      # Peak calling help
```

## See Also

- [Quick Start](quick_start.md) - Python API examples
- [Installation](installation.md) - Installation guide
- [Troubleshooting](troubleshooting.md) - Common issues

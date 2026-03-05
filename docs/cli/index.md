# Command-Line Interface

QuantNado provides command-line tools for common workflows, accessible via the main `quantnado` command.

## Overview

All QuantNado operations can be performed from the terminal without writing Python code:

```bash
quantnado --help
```

## Available Commands

QuantNado provides the following commands:

| Command | Purpose | Use Case |
|---------|---------|----------|
| [`create-dataset`](create_dataset.md) | Convert BAM files to Zarr | Initial data processing |
| [`call-peaks`](call_peaks.md) | Call peaks from bigWig files | Peak identification |

## General Options

All commands support these options:

- `--help, -h` - Show command help and usage
- `--verbose, -v` - Enable debug logging
- `--log-file` - Path to save logs

## Examples

### Quick Start: Create Dataset

```bash
quantnado create-dataset sample1.bam sample2.bam \
  --output dataset.zarr \
  --chromsizes hg38.chrom.sizes
```

### Quick Start: Call Peaks

```bash
quantnado call-peaks \
  --bigwig-dir ./bigwigs/ \
  --output-dir ./peaks/ \
  --chromsizes hg38.chrom.sizes \
  --quantile 0.98
```

## Getting Help

Get help for any command:

```bash
quantnado --help                    # Main help
quantnado create-dataset --help     # Create dataset help
quantnado call-peaks --help         # Peak calling help
```

## Current Limitations

- BAM file input requires local filesystem access
- Zarr output can be written to local or remote storage
- Large datasets may require significant disk space (see [Storage Requirements](../basic_usage.md))

## See Also

- [Create Dataset Command](create_dataset.md)
- [Call Peaks Command](call_peaks.md)
- [Python API](../api/quantnado.md)
- [Basic Usage](../basic_usage.md)

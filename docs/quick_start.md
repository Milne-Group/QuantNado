# Quick Start

Get your first QuantNado analysis running in 10 minutes.

## Prerequisites

- Aligned BAM files (indexed with `.bai`)
- Chromsizes file for your genome (e.g., `hg38.chrom.sizes`)
- QuantNado installed (see [Installation](installation.md))

## Step 1: Create a Dataset from BAM Files

The basic workflow starts with converting BAM files to a Zarr dataset:

```bash
quantnado create-dataset sample1.bam sample2.bam sample3.bam \
  --output my_dataset.zarr \
  --chromsizes hg38.chrom.sizes \
  --max-workers 4
```

This will:
- Count aligned reads at each genomic position
- Create an indexed Zarr dataset for quick access
- Store metadata about each sample
- Complete in a few minutes to hours depending on file size

## Step 2: Call Peaks (Optional)

If you have bigWig signal tracks, call peaks using quantile-based detection:

```bash
quantnado call-peaks \
  --bigwig-dir ./signal_tracks/ \
  --output-dir ./peaks/ \
  --chromsizes hg38.chrom.sizes \
  --quantile 0.98 \
  --merge
```

Output: BED files with called peaks for each sample.

## Next Steps

- **Analyze in Python**: Load the Zarr dataset for downstream analysis
- **More options**: See [CLI Reference](cli.md) for additional parameters
- **Troubleshooting**: Check [FAQ](faq.md) or [Troubleshooting](troubleshooting.md)

## Python Integration

Once your dataset is created:

```python
from quantnado import QuantNado

# Load existing dataset
qn = QuantNado.open("my_dataset.zarr")

# Get data for a specific region
region_data = qn.extract_region("chr1:1000000-2000000")

# Get all chromosomes as a dictionary of xarray DataArrays (lazy)
all_chroms = qn.to_xarray()
chr1_data = all_chroms["chr1"]

# Access sample names and metadata
print(qn.samples)
print(qn.metadata)
```

See [Basic Usage](basic_usage.md) for more analysis examples.

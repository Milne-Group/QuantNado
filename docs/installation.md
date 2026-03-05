# Installation

## System Requirements

- **Python**: 3.12 or 3.13 (3.14+ not yet supported due to PyO3 compatibility)
- **OS**: Linux, macOS, or WSL on Windows
- **Disk**: Sufficient space for BAM files and output Zarr datasets
- **RAM**: 4 GB minimum (8+ GB recommended for large files)

## Installation

### Via Conda (Recommended)

```bash
conda create -n quantnado python=3.13
conda activate quantnado
conda install -c conda-forge quantnado
```

Or with mamba (faster):

```bash
mamba create -n quantnado -c conda-forge quantnado
mamba activate quantnado
```

### Via Pip

```bash
pip install quantnado
```

### From Source (Development)

```bash
git clone https://github.com/Milne-Group/QuantNado.git
cd QuantNado

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

## Verify Installation

```bash
quantnado --version
quantnado --help
```

## Dependencies

Key dependencies installed automatically:

- **pysam** - BAM file handling
- **pyBigWig** - BigWig track handling
- **zarr** - Data storage
- **dask** - Parallel processing
- **xarray** - Multidimensional arrays
- **pandas** - Metadata handling

## Troubleshooting Installation

### Python 3.14 Compatibility Issue

If you see `PyO3's maximum supported version (3.13)` error, downgrade to Python 3.13:

```bash
conda install python=3.13
```

### BAM Indexing Requirement

Your BAM files must be indexed (`.bai` files). If missing:

```bash
samtools index sample.bam
```

### Missing Chromsizes File

Download chromsizes for your genome:

```bash
# Human hg38
curl -O ftp://hgdownload.cse.ucsc.edu/goldenPath/hg38/bigZips/hg38.chrom.sizes

# Or generate from FASTA
faidx --fetch-order genome.fasta | cut -f1,2 > genome.chrom.sizes
```
- **xarray** ≥ 2025.1.1 - Labeled arrays
- **dask** ≥ 2024.1.0 - Lazy evaluation and parallel computing
- **pandas** - Data structures
- **numpy** - Numerical computing
- **pyranges** ≥ 0.1.4 - Genomic ranges
- **scikit-learn** - Machine learning (PCA)

### Optional Dependencies

Development tools:

```bash
pip install quantnado[dev]
```

## Verify Installation

Check that QuantNado is properly installed:

```bash

quantnado --version

quantnado --help

```

## Next Steps

After installation, see:

- [Quick Start](quick_start.md) - Get running in 5 minutes
- [Basic Usage](basic_usage.md) - Learn core concepts
- [Examples](examples.md) - Full workflow examples

# QuantNado

QuantNado is a Python package for genomic quantification and signal analysis, with support for:

- **Dataset creation**: Efficient Zarr-backed storage for per-sample BAM signal across the genome
- **Signal reduction**: Aggregate signal over genomic ranges (genes, promoters, peaks, etc.)
- **Feature counting**: Generate DESeq2-compatible count matrices
- **Dimensionality reduction**: PCA analysis with dask support
- **Command-line tools**: Fast peak calling and dataset generation

---

## 📦 Installation

Install from PyPI:

```bash
pip install quantnado
```

Requires Python ≥ 3.8

---

## 🚀 Quick Start

### Create a dataset from BAM files

```python
from quantnado import QuantNado

# Create a new dataset
qn = QuantNado.from_bam_files(
    bam_files=["sample1.bam", "sample2.bam", "sample3.bam"],
    store_path="dataset.zarr",
)
```

### Load existing dataset and analyze

```python
from quantnado import QuantNado

# Open existing dataset
qn = QuantNado.open("dataset.zarr")

# Extract signal over genomic ranges (e.g., promoters)
promoters = qn.reduce("promoters.bed", reduction="mean")
print(promoters["mean"].shape)  # (n_promoters, n_samples)

# Run PCA on reduced signal
pca_obj, transformed = qn.pca(promoters["mean"], n_components=10)
print(transformed.shape)  # (n_samples, 10)

# Generate feature counts for DESeq2
counts, features = qn.feature_counts("genes.gtf", feature_type="gene")
print(counts.shape)  # (n_genes, n_samples)

# Extract specific genomic region
region = qn.extract_region("chr1:1000-5000")
print(region.shape)  # (n_samples, 4000)
```

### Access metadata and sample information

```python
qn = QuantNado.open("dataset.zarr")

# View sample names
print(qn.samples)  # ['sample1', 'sample2', 'sample3']

# View sample metadata
print(qn.metadata)

# View available chromosomes
print(qn.chromosomes)
```

---

## 📚 Full API Reference

### `QuantNado` class

**Factory methods:**
- `QuantNado.open(store_path, read_only=True)` - Open existing dataset
- `QuantNado.from_bam_files(bam_files, store_path, ...)` - Create dataset from BAM files

**Data access:**
- `.extract_region(region)` - Extract signal for a genomic region
- `.to_xarray(chromosomes)` - Load dataset as xarray DataArrays with lazy evaluation

**Analysis methods:**
- `.reduce(ranges, reduction="mean")` - Aggregate signal over genomic ranges
- `.feature_counts(gtf_file, feature_type="gene")` - Generate feature count matrix
- `.pca(data, n_components=10)` - Run PCA on reduced signal

**Properties:**
- `.samples` - List of sample names
- `.metadata` - Sample metadata (DataFrame)
- `.chromosomes` - Available chromosomes
- `.chromsizes` - Chromosome sizes
- `.store_path` - Path to Zarr store

---

## 🔧 Command-line Tools

QuantNado also provides CLI commands for common tasks:

### `quantnado-make-zarr` - Create dataset from BAM files

```bash
quantnado-make-zarr \
  --bam-dir path/to/bams/ \
  --store-path dataset.zarr \
  --chromsizes hg38.chrom.sizes
```

### `quantnado-call-peaks` - Call peaks from bigWig files

```bash
quantnado-call-peaks \
  --bigwig-dir path/to/bigwigs/ \
  --output-dir peaks/ \
  --chromsizes hg38.chrom.sizes \
  --quantile 0.98
```

---

## 📁 Package Structure

- **`quantnado/api.py`**: High-level `QuantNado` facade (recommended entry point)
- **`quantnado/dataset/`**: Core modules
  - `bam.py`: `BamStore` class for Zarr storage and manipulation
  - `reduce.py`: Signal reduction over genomic ranges
  - `counts.py`: Feature counting for DESeq2
  - `pca.py`: PCA analysis
  - `features.py`: GTF parsing and feature extraction
  - `metadata.py`: Sample metadata handling

---

## 💡 Example Workflows

### Workflow 1: Create dataset and explore

```python
from quantnado import QuantNado

# Create from BAM files
qn = QuantNado.from_bam_files(
    bam_files=["s1.bam", "s2.bam"],
    store_path="data.zarr",
    metadata="samples.csv",  # optional
)

# Inspect
print(qn.samples)
print(qn.metadata)
print(qn.chromosomes)
```

### Workflow 2: Signal reduction and PCA

```python
from quantnado import QuantNado

qn = QuantNado.open("data.zarr")

# Reduce over promoters
promoter_signal = qn.reduce("promoters.bed", reduction="mean")

# Visualize with PCA
pca_obj, transformed = qn.pca(promoter_signal["mean"], n_components=10)
```

### Workflow 3: Generate counts for differential analysis

```python
from quantnado import QuantNado

qn = QuantNado.open("data.zarr")

# Create count matrix
counts, features = qn.feature_counts(
    gtf_file="genes.gtf",
    feature_type="gene",
    integerize=True
)

# Export for DESeq2 (R)
counts.to_csv("counts.csv")
features.to_csv("features.csv")
```

---

## 📋 Requirements

- Python ≥ 3.8
- pysam, bamnado - BAM file handling
- zarr - HDF5-like storage
- xarray - Labeled arrays
- pandas - Data structures
- dask - Lazy evaluation
- scikit-learn - PCA
- typer - CLI

Optional dependencies for peak calling:
- `bbi` - bigWig I/O
- `torch`, `modisco-lite` - Deep learning (for advanced peak calling)

---

## 📝 License

See LICENSE file.

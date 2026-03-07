# QuantNado

<p align="center">
  <img src="docs/assets/images/logo.png" alt="QuantNado logo" width="192">
</p>

**QuantNado provides efficient Zarr-backed storage and analysis of genomic signal from BAM and bigWig files, with support for signal reduction, feature counting, dimensionality reduction, and quantile-based peak calling.**

[![CI](https://github.com/Milne-Group/QuantNado/actions/workflows/python-tests.yml/badge.svg)](https://github.com/Milne-Group/QuantNado/actions/workflows/python-tests.yml)
[![PyPI](https://img.shields.io/pypi/v/quantnado)](https://pypi.org/project/quantnado)
[![Docs](https://img.shields.io/badge/docs-milne--group.github.io-blue)](https://milne-group.github.io/QuantNado/)
[![License: GPLv3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)](https://pypi.org/project/quantnado)

---

## Installation

```bash
pip install quantnado
```

Requires Python 3.12 or 3.13.

---

## Quick Start

### Create a dataset from BAM files

```python
from quantnado import QuantNado

qn = QuantNado.from_bam_files(
    bam_files=["sample1.bam", "sample2.bam", "sample3.bam"],
    store_path="dataset.zarr",
    metadata="samples.csv",  # optional
)
```

### Load and analyse an existing dataset

```python
from quantnado import QuantNado

qn = QuantNado.open("dataset.zarr")

# Aggregate signal over genomic ranges
promoter_signal = qn.reduce("promoters.bed", reduction="mean")
print(promoter_signal["mean"].shape)  # (n_promoters, n_samples)

# PCA on reduced signal
pca_obj, transformed = qn.pca(promoter_signal["mean"], n_components=10)
print(transformed.shape)  # (n_samples, 10)

# Generate a count matrix for DESeq2
counts, features = qn.count_features("genes.gtf", feature_type="gene")
counts.to_csv("counts.csv")

# Extract signal over a specific region
region = qn.extract_region("chr1:1000-5000")
print(region.shape)  # (n_samples, 4000)
```

---

## Command-line Interface

QuantNado installs a `quantnado` command with two subcommands.

### `create-dataset` — build a multi-omics store from BAM/bedGraph/VCF files

At least one of `--bam`, `--bedgraph`, or `--vcf` is required. File lists are comma-separated.

```bash
quantnado create-dataset \
  --output dataset \
  --bam sample1.bam,sample2.bam,sample3.bam \
  --bedgraph meth_rep1.bedGraph,meth_rep2.bedGraph \
  --vcf sample1.vcf.gz,sample2.vcf.gz \
  --metadata samples.csv \
  --max-workers 8
```

### `call-peaks` — call quantile-based peaks from bigWig files

```bash
quantnado call-peaks \
  --bigwig-dir path/to/bigwigs/ \
  --output-dir peaks/ \
  --chromsizes hg38.chrom.sizes \
  --quantile 0.98
```

Run `quantnado --help` or `quantnado <subcommand> --help` for full option listings.

---

## API Reference

Full documentation is available at [milne-group.github.io/QuantNado](https://milne-group.github.io/QuantNado/).

### `QuantNado`

| Method / Property | Description |
|---|---|
| `QuantNado.from_bam_files(bam_files, store_path, ...)` | Create a new dataset from BAM files |
| `QuantNado.open(store_path, read_only=True)` | Open an existing dataset |
| `.reduce(ranges, reduction="mean")` | Aggregate signal over genomic ranges (BED) |
| `.count_features(gtf_file, feature_type="gene")` | Generate a DESeq2-compatible count matrix |
| `.pca(data, n_components=10)` | Run PCA on a signal matrix |
| `.extract_region(region)` | Extract raw signal for a genomic region |
| `.to_xarray(chromosomes)` | Load dataset as lazy xarray DataArrays |
| `.samples` | List of sample names |
| `.metadata` | Sample metadata (DataFrame) |
| `.chromosomes` | Available chromosome names |
| `.chromsizes` | Chromosome sizes (dict) |
| `.store_path` | Path to the underlying Zarr store |

---

## Requirements

| Dependency | Purpose |
|---|---|
| `zarr`, `icechunk` | Zarr v3 storage backend |
| `xarray`, `dask` | Lazy array operations |
| `pandas`, `numpy` | Data structures |
| `pysam`, `bamnado` | BAM file I/O |
| `pyBigWig` | bigWig I/O |
| `pyranges` | Genomic range operations |
| `scikit-learn` (via `dask-ml`) | PCA |
| `typer`, `loguru` | CLI and logging |
| `ipykernel`, `jupyterlab`, `matplotlib` | Example notebook (`pip install "quantnado[example]"`) |


---

## License

GNU GPL v3.0 — see [LICENSE](LICENSE).

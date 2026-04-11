# QuantNado

[![Docs](https://img.shields.io/badge/docs-milne--group.github.io-blue)](https://milne-group.github.io/QuantNado/)
[![CI](https://github.com/Milne-Group/QuantNado/actions/workflows/python-tests.yml/badge.svg)](https://github.com/Milne-Group/QuantNado/actions/workflows/python-tests.yml)
[![Release](https://img.shields.io/github/v/release/Milne-Group/QuantNado?sort=semver)](https://github.com/Milne-Group/QuantNado/releases)
[![License](https://img.shields.io/badge/license-GPLv3-blue.svg)](LICENSE)
[![PyPI Version](https://img.shields.io/pypi/v/quantnado)](https://pypi.org/project/quantnado)

<p align="center">
  <img src="docs/assets/images/logo.png" alt="QuantNado logo" width="192">
</p>

**QuantNado builds per-sample Zarr stores from genomic assays and exposes a unified Python API for region selection, reduction, feature counting, normalisation, PCA, and peak calling.**

## Installation

```bash
pip install quantnado
```

Requires Python 3.12 or 3.13.

## Quick Start

### 1. Create per-sample stores from a metadata table

```csv
sample,assay,bam,methyl,vcf,ip
ATAC_1,ATAC,/data/ATAC_1.bam,,,
H3K27ac_1,ChIP,/data/H3K27ac_1.bam,,,H3K27ac
RNA_1,RNA,/data/RNA_1.bam,,,
METH_1,METH,/data/METH_1.bam,/data/METH_1.bedGraph,,
SNP_1,SNP,,,/data/SNP_1.vcf.gz,
```

```bash
quantnado create-dataset \
  --metadata samples.csv \
  --output-dir dataset \
  --chromsizes hg38.chrom.sizes
```

This writes one `.zarr` store per sample into `dataset/`.

For quick test builds, you can either use the default test chromosomes:

```bash
quantnado create-dataset \
  --metadata samples.csv \
  --output-dir dataset \
  --test
```

or provide an explicit list:

```bash
quantnado create-dataset \
  --metadata samples.csv \
  --output-dir dataset \
  --test-chrom chr21 \
  --test-chrom chr9
```

### 2. Open the dataset in Python

```python
from quantnado import QuantNado

qn = QuantNado.open("dataset/")

print(qn.sample_names)
print(qn.assays)
print(qn.array_keys)
print(qn.info)
```

### 3. Run common analyses

```python
# Select a genomic region
region = qn.sel("chr1", 1_000_000, 1_010_000)

# Reduce signal over intervals
promoters = qn.reduce(
    intervals_path="promoters.bed",
    reduction="mean",
    modality="coverage",
)

# Quantify stored signal over genes
gene_signal, gene_meta = qn.quantify_signal(
    gtf_file="genes.gtf",
    feature_type="gene",
    assay="RNA",
    modality="coverage",
)

# Count features using the current signal backend
counts, features = qn.count_features(
    gtf_file="genes.gtf",
    feature_type="gene",
    engine="signal",
    assay="RNA",
)

# PCA on reduced signal
pca_obj, pca_result = qn.pca(promoters["mean"], n_components=10)
```

### 4. Optionally combine stores

```python
from quantnado import QuantNado

combined = QuantNado.combine("dataset/", "dataset/combined.zarr")
```

You can open either `dataset/` or `dataset/combined.zarr` with the same API.

## CLI

QuantNado installs a `quantnado` command with two main workflows.

### `create-dataset`

Creates per-sample Zarr stores from a metadata CSV/TSV.

```bash
quantnado create-dataset \
  --metadata samples.csv \
  --output-dir dataset \
  --overwrite
```

Supported assays: `ATAC`, `ChIP`, `RNA`, `CUT&TAG`, `METH`, `SNP`, `MCC`.

### `call-peaks`

Calls peaks directly from a QuantNado dataset.

```bash
quantnado call-peaks \
  --zarr dataset/combined.zarr \
  --method quantile \
  --assay atac \
  --output-dir peaks/
```

Available methods: `quantile`, `seacr`, and `lanceotron`.

## Python API

The main entry points are:

| Object / function | Purpose |
|---|---|
| `QuantNado.open(path)` | Open a directory of per-sample stores or a combined `.zarr` |
| `QuantNado.combine(src, output)` | Combine per-sample stores into one multi-sample store |
| `QuantNadoDataset(path)` | Lower-level analysis object used by the facade |
| `create_dataset(...)` | Build a single per-sample store programmatically |
| `metadata_from_seqnado(...)` | Generate a QuantNado metadata table from a SeqNado project |

Common analysis methods on `QuantNado` / `QuantNadoDataset`:

| Method | Purpose |
|---|---|
| `.sel(chrom, start, end, ...)` | Extract a genomic region as `xr.Dataset` |
| `.reduce(...)` | Summarise signal over BED/GTF intervals |
| `.quantify_signal(...)` | Quantify stored signal over features |
| `.count_features(...)` | Count features via the selected engine (`signal` today, `bam` planned) |
| `.extract(...)` | Bin signal around promoters, genes, transcripts, or exons |
| `.normalise(...)` | Apply CPM/RPKM/TPM normalisation |
| `.group_by(...)`, `.subset(...)`, `.info` | Notebook-friendly sample grouping, filtering, and dataset summaries |
| `.pca(...)` | Run PCA on reduced or selected signal |
| `.metaplot(...)`, `.tornadoplot(...)`, `.heatmap(...)`, `.correlate(...)` | Visualisation helpers |

## Documentation

Full documentation is available at [milne-group.github.io/QuantNado](https://milne-group.github.io/QuantNado/).

## License

GNU GPL v3.0. See [LICENSE](LICENSE).

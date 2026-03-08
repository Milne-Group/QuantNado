# QuantNado

**Multi-modal genomic signal storage and analysis**

![logo](assets/images/logo.png){: style="display: block; margin: 0 auto; width: 200px;"}

QuantNado converts aligned genomic files (BAM, bedGraph, VCF) into efficient Zarr-backed stores and provides a unified Python API for signal extraction, methylation analysis, variant handling, and visualisation.

## What QuantNado Does

| Modality | Input | Stored data |
|---|---|---|
| **Coverage** | BAM files | per-base read depth, dense (sample × position) |
| **Methylation** | MethylDackel bedGraph | CpG methylation %, counts, sparse |
| **Variants** | VCF.gz files | genotype, allele depths, quality, sparse |

All modalities live in a single directory (`MultiomicsStore`) and are accessed through one Python object.

## Quick Example

```python
import quantnado as qn

# Create a multi-modal dataset
qn.create_dataset(
    store_dir="dataset/",
    bam_files=["atac.bam", "chip.bam"],
    bedgraph_files=["meth-rep1.bedGraph", "meth-rep2.bedGraph"],
    vcf_files=["snp.vcf.gz"],
    bedgraph_sample_names=lambda p: p.stem.split("_hg38")[0],
)

# Open and explore
ds = qn.open("dataset/")
print(ds.modalities)   # ['coverage', 'methylation', 'variants']

# Coverage analysis
reduced = ds.reduce("promoters.bed", reduction="mean")
pca_obj, pca_result = ds.pca(reduced["mean"])
counts, features = ds.count_features("genes.gtf", feature_type="gene")

# Visualise
binned = ds.extract(feature_type="transcript", gtf_path="genes.gtf",
                    upstream=2000, downstream=2000, anchor="start", bin_size=50)
ds.metaplot(binned, modality="coverage", title="Metagene profile")

# Methylation
binned_meth = ds.extract(modality="methylation", feature_type="transcript",
                         gtf_path="genes.gtf", upstream=1000, downstream=1000,
                         anchor="start", bin_size=50)
ds.metaplot(binned_meth, modality="methylation", title="CpG methylation at TSS")

# Sub-store access
ds.coverage    # BamStore
ds.methylation # MethylStore
ds.variants    # VariantStore
```

## Key Features

- **Unified entry point** — `qn.open()` / `qn.create_dataset()` for all modalities
- **Lazy loading** — Zarr + Dask; only reads what you ask for
- **Multi-modal** — coverage, methylation, and variant data in one store
- **Analysis built-in** — reduce, PCA, feature counts, metaplots, tornado plots
- **Resumable** — skip completed samples when re-running

## Documentation

- **[Installation](installation.md)** — Setup and dependencies
- **[Quick Start](quick_start.md)** — First analysis in minutes
- **[Usage Guide](basic_usage.md)** — Coverage, methylation, and variant workflows
- **[CLI Reference](cli/create_dataset.md)** — Command-line interface
- **[API Reference](api/quantnado.md)** — Full Python API

## Citation

If you use QuantNado, please cite:

```
QuantNado: Efficient genomic signal quantification and analysis
Milne Group, University of Oxford
```

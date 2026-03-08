# Quick Start

Get your first QuantNado analysis running in minutes.

## Prerequisites

- Aligned BAM files (indexed with `.bai`)
- QuantNado installed (see [Installation](installation.md))

## Step 1: Create a Dataset

Build a multi-modal Zarr store from your genomic files. Provide whichever modalities you have — all are optional independently.

```python
import quantnado as qn

qn.create_dataset(
    store_dir="dataset/",
    bam_files=["atac.bam", "chip.bam", "rna.bam"],      # coverage
    methyldackel_files=["meth-rep1.bedGraph"],                # methylation
    vcf_files=["snp.vcf.gz"],                            # variants
    bedgraph_sample_names=lambda p: p.stem.split("_hg38")[0],
    max_workers=4,
)
```

Or from the command line:

```bash
quantnado create-dataset \
  --output dataset/ \
  --bam atac.bam,chip.bam \
  --bedgraph meth-rep1.bedGraph \
  --vcf snp.vcf.gz \
  --max-workers 4
```

## Step 2: Open and Inspect

```python
ds = qn.open("dataset/")
print(ds)
# MultiomicsStore at 'dataset/'
#   modalities : ['coverage', 'methylation', 'variants']
#   coverage   : 3 samples, 24 chrom(s)
#   methylation: 1 samples, 24 chrom(s)
#   variants   : 1 samples, 24 chrom(s)

print(ds.modalities)   # ['coverage', 'methylation', 'variants']
print(ds.chromosomes)  # ['chr1', 'chr2', ..., 'chrX']

metadata = ds.get_metadata()  # combined metadata across all modalities
```

## Step 3: Coverage Analysis

```python
# Reduce signal over promoter windows → (regions × samples) matrix
reduced = ds.reduce("promoters.bed", reduction="mean")

# PCA for QC
pca_obj, pca_result = ds.pca(reduced["mean"], n_components=8)
qn.plot_pca_scatter(pca_obj, pca_result, colour_by="assay", metadata_df=metadata)

# Feature counts (DESeq2-compatible)
counts, features = ds.count_features("genes.gtf", feature_type="gene", integerize=True)

# Extract signal over TSS windows → metagene / tornado plots
binned = ds.extract(
    feature_type="transcript",
    gtf_path="genes.gtf",
    upstream=2000,
    downstream=2000,
    anchor="start",
    bin_size=50,
)
ds.metaplot(binned, modality="coverage", title="Coverage at TSS")
ds.tornadoplot(binned, modality="coverage", sort_by="mean")
```

## Step 4: Methylation Analysis

```python
meth = ds.methylation

# CpG sites per chromosome
for chrom in meth.chromosomes:
    print(f"{chrom}: {len(meth.get_positions(chrom)):,} CpG sites")

# Extract methylation over TSS windows
binned_meth = ds.extract(
    modality="methylation",
    feature_type="transcript",
    gtf_path="genes.gtf",
    upstream=1000,
    downstream=1000,
    anchor="start",
    bin_size=50,
)
ds.metaplot(binned_meth, modality="methylation", title="CpG methylation at TSS")

# Feature-level methylation stats
stats, features = ds.methylation.count_features(gtf_file="genes.gtf", feature_type="gene")
# stats keys: n_methylated, n_unmethylated, n_cpg_covered, methylation_ratio, methylation_pct
```

## Step 5: Variant Analysis

```python
var = ds.variants

# Genotype DataArrays
gt_xr = var.to_xarray(variable="genotype")  # dict[chrom → DataArray]

# Extract region
region_gt = var.extract_region("chr21:5000000-6000000", variable="genotype")
refs, alts = var.get_alleles("chr21")
```

## Next Steps

- **[Usage Guide](basic_usage.md)** — More detailed workflows for each modality
- **[API Reference](api/quantnado.md)** — Complete Python API documentation
- **[CLI Reference](cli/create_dataset.md)** — All command-line options
- **[Example Notebook](example_dataset.ipynb)** — Full worked example

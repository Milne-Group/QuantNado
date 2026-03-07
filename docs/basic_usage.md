# Usage Guide

Detailed workflows for each modality — coverage, methylation, and variants.

## Opening a Dataset

```python
import quantnado as qn

ds = qn.open("dataset/")
print(ds)
# MultiomicsStore at 'dataset/'
#   modalities : ['coverage', 'methylation', 'variants']
#   coverage   : 3 samples, 24 chrom(s)
#   methylation: 1 samples, 24 chrom(s)
#   variants   : 1 samples, 24 chrom(s)

print(ds.modalities)   # ['coverage', 'methylation', 'variants']
print(ds.chromosomes)  # ['chr1', 'chr2', ..., 'chrX']
print(ds.samples)      # list of coverage sample IDs (if coverage available)
```

Sub-stores are accessed as properties:

```python
ds.coverage    # BamStore  (or None if not present)
ds.methylation # MethylStore (or None if not present)
ds.variants    # VariantStore (or None if not present)
```

---

## Coverage Analysis

Coverage is stored as dense per-base read depth (sample × position) in Zarr.

### Metadata

```python
metadata = ds.get_metadata()    # combined metadata across all modalities
print(metadata.columns)

# Update or extend
ds.set_metadata(new_df)
ds.update_metadata({"batch": "batch1"}, sample_ids=["sample1", "sample2"])
```

### Reduce Signal Over Regions

Collapse per-base signal over a BED file of regions into a (regions × samples) matrix:

```python
# Returns dict with one key per reduction method
reduced = ds.reduce("promoters.bed", reduction="mean")
mat = reduced["mean"]          # numpy array: (n_regions, n_samples)

# Multiple reductions at once
reduced = ds.reduce("peaks.bed", reduction=["mean", "max", "sum"])
```

### PCA

```python
pca_obj, pca_result = ds.pca(reduced["mean"], n_components=8)

# Scree plot
qn.plot_pca_scree(pca_obj)

# Scatter coloured by a metadata column
qn.plot_pca_scatter(pca_obj, pca_result, colour_by="assay", metadata_df=metadata)
```

### Feature Counts (DESeq2-compatible)

Count reads over genomic features from a GTF:

```python
counts, features = ds.count_features(
    "genes.gtf",
    feature_type="gene",
    integerize=True,   # round to integers for DESeq2
)
# counts: (n_genes, n_samples) array
# features: DataFrame with chr, start, end, gene_id, ...
```

### Metagene and Tornado Plots

Extract binned signal over windows anchored to genomic features:

```python
binned = ds.extract(
    feature_type="transcript",
    gtf_path="genes.gtf",
    upstream=2000,
    downstream=2000,
    anchor="start",
    bin_size=50,
)

# Metagene (average profile across all features)
ds.metaplot(binned, modality="coverage", title="Coverage at TSS")

# Tornado (per-feature heatmap)
ds.tornadoplot(binned, modality="coverage", sort_by="mean")
```

You can also supply a BED file or a pre-built DataFrame of ranges:

```python
binned = ds.extract(intervals_path="peaks.bed", upstream=500, downstream=500, bin_size=25)
```

---

## Methylation Analysis

Methylation is stored sparsely: only covered CpG positions are kept per chromosome.

### Inspect the Store

```python
meth = ds.methylation

print(meth.chromosomes)
print(meth.samples)

# Number of CpG sites per chromosome
for chrom in meth.chromosomes:
    print(f"{chrom}: {len(meth.get_positions(chrom)):,} CpG sites")
```

### Available Variables

Each CpG site stores five variables:

| Variable | Description |
|---|---|
| `methylation_pct` | Methylation percentage (0–100) |
| `methylation_ratio` | Methylation ratio (0–1) |
| `n_methylated` | Count of methylated reads |
| `n_unmethylated` | Count of unmethylated reads |
| `n_cpg_covered` | Total reads covering this site |

### Metagene and Tornado Plots

Use `ds.extract()` with `modality="methylation"`:

```python
binned_meth = ds.extract(
    modality="methylation",
    variable="methylation_pct",   # default
    feature_type="transcript",
    gtf_path="genes.gtf",
    upstream=1000,
    downstream=1000,
    anchor="start",
    bin_size=50,
)

ds.metaplot(binned_meth, modality="methylation", title="CpG methylation at TSS")
ds.tornadoplot(binned_meth, modality="methylation", sort_by="mean")
```

### Feature-level Methylation Stats

```python
stats, features = ds.methylation.count_features("genes.gtf", feature_type="gene")
# stats is a dict with keys:
#   n_methylated, n_unmethylated, n_cpg_covered,
#   methylation_ratio, methylation_pct
```

---

## Variant Analysis

Variants are stored sparsely from VCF.gz files, indexed per chromosome.

### Inspect the Store

```python
var = ds.variants

print(var.chromosomes)
print(var.samples)
```

### Available Variables

| Variable | Description |
|---|---|
| `genotype` | Integer-encoded genotype (0=hom-ref, 1=het, 2=hom-alt) |
| `allele_depth` | Read depth per allele |
| `quality` | Variant quality score |

### Extract as xarray

```python
# Full chromosome as a DataArray (lazy)
gt_xr = var.to_xarray(variable="genotype")     # dict[chrom → DataArray]

# Specific region
region_gt = var.extract_region("chr21:5000000-6000000", variable="genotype")

# Allele information
refs, alts = var.get_alleles("chr21")
```

---

## Creating Datasets

### Python API

```python
import quantnado as qn

qn.create_dataset(
    store_dir="dataset/",
    bam_files=["atac.bam", "chip.bam", "rna.bam"],
    bedgraph_files=["meth-rep1.bedGraph", "meth-rep2.bedGraph"],
    vcf_files=["snp.vcf.gz"],
    # Callable to derive sample names from file paths
    bedgraph_sample_names=lambda p: p.stem.split("_hg38")[0],
    max_workers=4,
)
```

All modalities are optional — pass only what you have.

### Command Line

```bash
quantnado create-dataset \
  --output dataset/ \
  --bam atac.bam,chip.bam,rna.bam \
  --bedgraph meth-rep1.bedGraph,meth-rep2.bedGraph \
  --vcf snp.vcf.gz \
  --max-workers 4
```

---

## Metadata Management

```python
import pandas as pd

# Read current metadata
meta = ds.get_metadata()

# Set metadata from a DataFrame (replaces existing)
new_meta = pd.DataFrame({
    "sample_id": ds.samples,
    "condition": ["control", "control", "treatment"],
    "assay": ["ATAC", "ChIP", "RNA"],
})
ds.set_metadata(new_meta)

# Add/update individual columns
ds.update_metadata({"batch": "batch1"})

# Inspect columns
print(ds.list_metadata_columns())

# Export / import
ds.metadata_to_csv("metadata.csv")
ds.metadata_from_csv("updated_metadata.csv")
```

---

## Common Questions

**Can I resume an interrupted dataset creation?**
Yes — pass `resume=True` (Python) or `--resume` (CLI). QuantNado skips samples that are already written.

**How long does creation take?**
Coverage stores are the slowest (20 min – 2 h per whole-genome BAM). Use `max_workers` to parallelise. Methylation and variant stores are typically much faster.

**What if I only have one modality?**
`qn.open()` works with any combination. If you only have BAM files, pass them directly to `BamStore.from_bam_files()` or use `qn.create_dataset(bam_files=...)`.

See [Troubleshooting](troubleshooting.md) for more help.

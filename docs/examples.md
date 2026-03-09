# Examples and Workflows

For a complete, runnable end-to-end walkthrough using a real multi-omics dataset, see the
**[Example Notebook](../example/example_dataset.ipynb)** — it covers dataset creation, signal
reduction, PCA, feature counts, normalisation, and visualisation over a chr21 test dataset.

---

The snippets below show focused workflows for common QuantNado tasks.

## 1. Create a Multi-Modal Dataset

```python
import quantnado as qn

ds = qn.create_dataset(
    store_dir="dataset/",
    bam_files=["atac.bam", "chip.bam", "rna_ctrl.bam", "rna_treat.bam"],
    methylation_files=["meth_rep1.bedGraph", "meth_rep2.bedGraph"],
    methylation_sample_names=["meth-rep1", "meth-rep2"],
    vcf_files=["snp.vcf.gz"],
    filter_chromosomes=True,      # keep only canonical chrN / chrX / chrY
    max_workers=8,
    chr_workers=4,
    chromsizes="hg38.chrom.sizes",
)
```

Omit modalities you don't have — the store is created with whatever is supplied.
To open an existing store:

```python
ds = qn.open("dataset/")
print(ds)
# MultiomicsStore at 'dataset/'
#   modalities : ['coverage', 'methylation', 'variants']
#   coverage    : 4 samples, 25 chrom(s)
#   ...
```

---

## 2. Inspect the Dataset

```python
print("Samples    :", ds.samples)
print("Modalities :", ds.modalities)
print("Chromosomes:", ds.chromosomes)

metadata = ds.get_metadata()
print(metadata)
```

---

## 3. QC — PCA, Heatmap, Correlation

### Reduce signal over promoters

```python
promoter_signal = ds.reduce(intervals_path="promoters.bed", reduction="mean")
# xr.Dataset with dims (ranges, sample) and variables: sum, count, mean
```

### PCA

```python
pca_obj, pca_result = qn.pca(
    promoter_signal["mean"],
    n_components=10,
    chromosome="chr21",          # subsample one chrom for speed
    nan_handling_strategy="drop",
    random_state=42,
)

qn.plot_pca_scree(pca_obj, filepath="figures/pca_scree.png")

import pandas as pd
metadata["assay"] = metadata.index.str.split("-").str[0]

qn.plot_pca_scatter(
    pca_obj,
    pca_result,
    xaxis_pc=1,
    yaxis_pc=2,
    metadata_df=metadata,
    colour_by="assay",
    filepath="figures/pca_scatter.png",
)
```

### Clustered heatmap

```python
g = qn.heatmap(
    promoter_signal,          # xr.Dataset from reduce()
    variable="mean",          # which variable to plot
    log_transform=True,       # log1p before clustering
    cmap="mako",
    figsize=(5, 6),
    title="Promoter signal — clustered heatmap",
    filepath="figures/heatmap.png",
)
```

`qn.heatmap` also accepts a `pd.DataFrame` (e.g. from `count_features()`):

```python
counts, features = ds.count_features(gtf_file="genes.gtf", feature_type="gene")
g = qn.heatmap(counts, log_transform=True, title="Gene counts")
```

### Sample–sample correlation

```python
corr_df, g = qn.correlate(
    promoter_signal,
    variable="mean",
    method="pearson",           # or "spearman"
    log_transform=True,
    annotate=True,
    figsize=(6, 6),
    title="Sample–sample correlation (promoter signal)",
    filepath="figures/correlation.png",
)
print(corr_df)                  # pd.DataFrame (samples × samples)
```

---

## 4. Coverage Analysis

### Reduce signal over any BED file

```python
signal = ds.reduce(intervals_path="peaks.bed", reduction="mean")
# signal["mean"]  → (ranges, sample) DataArray
# signal["sum"]   → (ranges, sample) DataArray
# signal["count"] → (ranges, sample) DataArray (non-zero positions)
```

### Count features for DESeq2 / edgeR

```python
rna_samples = ["rna_ctrl", "rna_treat"]

counts, features = ds.count_features(
    gtf_file="genes.gtf",
    feature_type="exon",
    feature_id_col=["gene_id"],
    samples=rna_samples,        # only count these samples
    strand=2,                   # 1 = stranded, 2 = reverse stranded, 0 = unstranded
    integerize=True,
)

# Filter low-count genes
keep = counts.sum(axis=1) >= 10
counts.loc[keep].to_csv("counts.csv")
features.loc[keep].to_csv("features.csv")
```

### Extract binned signal around TSSs

```python
binned = ds.extract(
    modality="coverage",
    feature_type="transcript",
    gtf_path="genes.gtf",
    upstream=1000,
    downstream=1000,
    anchor="start",             # strand-aware TSS
    bin_size=50,
)
# DataArray with dims (interval, bin, sample)
```

### Metagene profile

```python
groups = {"ATAC": ["atac"], "ChIP": ["chip"], "RNA": ["rna_ctrl", "rna_treat"]}

ax = ds.metaplot(
    binned,
    modality="coverage",
    groups=groups,
    flip_minus_strand=True,
    error_stat="sem",
    reference_point=0,
    reference_label="TSS",
    xlabel="Distance from TSS (bp)",
    title="Coverage metagene",
    filepath="figures/metaplot.png",
)
```

### Tornado plot

```python
axes = ds.tornadoplot(
    binned,
    modality="coverage",
    samples=["atac", "chip", "rna_ctrl", "rna_treat"],
    sample_names=["ATAC", "ChIP", "RNA ctrl", "RNA treat"],
    flip_minus_strand=True,
    sort_by="mean",
    ylabel="Intervals",
    title="Coverage tornado — TSS ± 1 kb",
    filepath="figures/tornado.png",
)
```

---

## 5. Normalisation

`qn.normalise` works on the output of `reduce()`, `extract()`, or `count_features()`.

```python
library_sizes = qn.get_library_sizes(ds)   # pd.Series (sample → total_reads)
```

### CPM — per-position signal (metaplots, tornado plots)

CPM is the correct normalisation for binned/per-position signal from `extract()`:

```python
cpm_binned = ds.normalise(binned, method="cpm", library_sizes=library_sizes)

axes = ds.tornadoplot(
    cpm_binned,
    modality="coverage",
    title="CPM-normalised tornado — TSS ± 1 kb",
)
```

### RPKM / TPM — feature count matrices

```python
counts, features = ds.count_features(gtf_file="genes.gtf", feature_type="gene")

rpkm = qn.normalise(
    counts,
    dataset=ds,               # auto-reads library sizes
    method="rpkm",
    feature_lengths=features["range_length"],
)

tpm = qn.normalise(
    counts,
    method="tpm",             # self-normalising — no dataset needed
    feature_lengths=features["range_length"],
)
```

---

## 6. Methylation Analysis

```python
meth = ds.methylation
print(meth.sample_names)      # ['meth-rep1', 'meth-rep2']

# Per-chromosome sparse DataArray (sample × CpG_positions)
meth_xr = meth.to_xarray(variable="methylation_pct")

# Extract a single locus
region_meth = meth.extract_region("chr21:36000000-36100000", variable="methylation_pct")

# Aggregate over features
stats, features = ds.methylation.count_features(
    gtf_file="genes.gtf",
    feature_type="transcript",
    feature_id_col=["gene_id", "transcript_id"],
)
# stats is a dict: {stat_name → pd.DataFrame (features × samples)}

# Metaplot / tornado plot
binned_meth = ds.extract(
    modality="methylation",
    feature_type="transcript",
    gtf_path="genes.gtf",
    upstream=1000,
    downstream=1000,
    anchor="start",
    bin_size=50,
)

ax = ds.metaplot(binned_meth, modality="methylation", flip_minus_strand=True)
axes = ds.tornadoplot(binned_meth, modality="methylation", sort_by="mean")
```

---

## 7. Variant Analysis

```python
var = ds.variants
gt_xr = var.to_xarray(variable="genotype")
# genotype encoding: -1 missing, 0 hom-ref, 1 het, 2 hom-alt

# Region-level extraction
REGION = "chr21:36000000-36100000"
gt   = var.extract_region(REGION, variable="genotype")
adr  = var.extract_region(REGION, variable="allele_depth_ref")
ada  = var.extract_region(REGION, variable="allele_depth_alt")
```

---

## 8. Multi-Modal Locus Browser

View all modalities together in a genome-browser-style layout:

```python
REGION = "chr21:36193575-36260996"

gt  = ds.variants.extract_region(REGION, variable="genotype").compute()
adr = ds.variants.extract_region(REGION, variable="allele_depth_ref").compute()
ada = ds.variants.extract_region(REGION, variable="allele_depth_alt").compute()

axes = ds.locus_plot(
    locus=REGION,
    genotype=gt,
    allele_depth_ref=adr,
    allele_depth_alt=ada,
    sample_names=["atac", "chip", "meth-rep1", "meth-rep2", "snp"],
    modality=["coverage", "coverage", "methylation", "methylation", "variant"],
    palette="tab10",
    title=f"Multi-omics locus — {REGION}",
    figsize=(12, 8),
    filepath="figures/locus_plot.png",
)
```

Coverage and methylation tracks are fetched automatically from the store when not supplied
explicitly. Only the variant arrays need to be pre-computed.

---

## Tips

### Lazy computation
All `reduce()` / `extract()` outputs are backed by lazy Dask arrays. Call `.compute()` only
when you need a concrete NumPy array; plotting functions handle this internally.

### Chromosome filtering
Pass `filter_chromosomes=True` to `create_dataset()` to discard non-canonical contigs (patches,
unlocalized sequences) automatically.

### Selective sample processing
`count_features()` and `reduce()` accept a `samples=` list so you can operate on a subset without
re-running the full pipeline.

### Parallel dataset construction
```python
qn.create_dataset(
    ...,
    max_workers=8,   # samples processed in parallel
    chr_workers=4,   # chromosomes processed in parallel per sample
)
```

### DESeq2 / edgeR handoff
```python
counts, features = ds.count_features(
    gtf_file="genes.gtf",
    feature_type="exon",
    feature_id_col=["gene_id"],
    integerize=True,
)
counts.to_csv("counts_for_deseq2.csv")
```
The output is a plain `pd.DataFrame` (genes × samples) with integer counts, ready to pass
directly to `DESeqDataSetFromMatrix` or `DGEList`.

---

## See Also

- [Quick Start](quick_start.md)
- [API Reference](api/quantnado.md)
- [Example Notebook](../example/example_dataset.ipynb)

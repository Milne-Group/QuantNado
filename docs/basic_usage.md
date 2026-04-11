# Usage Guide

This page reflects the current post-refactor API: datasets are opened from a directory of per-sample `.zarr` stores or from a combined `.zarr`, and most analysis methods work through `QuantNado` or `QuantNadoDataset`.

## Opening a Dataset

```python
from quantnado import QuantNado, QuantNadoDataset

qn = QuantNado.open("dataset/")
raw = QuantNadoDataset("dataset/")
combined = QuantNado.open("dataset/combined.zarr")
```

Useful properties:

```python
qn.sample_names
qn.assays
qn.array_keys
qn.chromosomes
qn.chromsizes
qn.info
```

`qn.info` returns a compact dataset summary that includes assay-level sample counts, array keys, inferred IPs, and any cached sample-group namespaces.

## Region Selection

```python
region = qn.sel("chr21", 36_000_000, 36_010_000)
```

Filter by assay type or explicit sample names:

```python
rna_region = qn.sel("chr21", 36_000_000, 36_010_000, assay="RNA")
atac_region = qn.sel("chr21", 36_000_000, 36_010_000, samples=["ATAC_1"])
```

You can also build reusable sample groups and intersect them in `subset()`:

```python
qn.group_by(
    ip="ip",
    treatment={"control": ["control"], "treated": ["treated"]},
    replicate={"rep1": ["rep1"], "rep2": ["rep2"]},
    spikein={"spikein": ["spikein", "rx"]},
    match="contains",
)

chip_subset = qn.subset(
    assay="CHIP",
    ip="MLL",
    group={"spikein": "spikein"},
)

rna_subset = qn.subset(
    assay="RNA",
    group={
        "treatment": ["treated", "control"],
        "spikein": "spikein",
        "replicate": ["rep1", "rep2"],
    },
)
```

`subset()` intersects all requested filters. If a combination resolves to no samples, the error message reports which group namespace removed the remaining samples.

## Assay vs Modality

- `assay` means a biological assay such as `ATAC`, `RNA`, or `METH`
- `modality` means an array key such as `coverage`, `rna_fwd`, `chip_h3k27ac`, `methyl_pct`, or `GT`

In practice, use `assay=` to choose samples and `modality=` to choose which signal array to analyse.

## Reducing Signal Over Intervals

Use BED intervals directly:

```python
signal = qn.reduce(
    intervals_path="promoters.bed",
    reduction="mean",
    modality="coverage",
)
```

Or derive intervals from a GTF:

```python
gene_signal = qn.reduce(
    gtf_path="genes.gtf",
    feature_type="gene",
    reduction="sum",
    assay="RNA",
    modality="coverage",
)
```

Returned value: `xr.Dataset` with variables such as `sum`, `count`, and `mean`.
The sample axis remains sample-oriented, so the usual next step is `signal["mean"]` for PCA, heatmaps, or correlations.

## Counting Features

```python
counts, features = qn.count_features(
    gtf_file="genes.gtf",
    feature_type="gene",
    engine="signal",
    assay="RNA",
)
```

`counts` is a feature-by-sample `pd.DataFrame`. `features` is the aligned feature metadata table.

For explicit signal quantification without the counting terminology:

```python
signal_matrix, signal_features = qn.quantify_signal(
    gtf_file="genes.gtf",
    feature_type="gene",
    assay="RNA",
    modality="coverage",
)
```

`quantify_signal()` is the recommended API when you want a feature-by-sample matrix derived from stored QuantNado signal rather than BAM-backed read assignment.

`count_features(engine="signal")` is still a signal-derived summarisation path. A BAM-backed `engine="bam"` mode is planned for featureCounts-style read assignment semantics.

You can also count over BED intervals:

```python
counts, features = qn.count_features(
    bed_file="peaks.bed",
    samples=["ATAC_1", "ATAC_2"],
    modality="coverage",
)
```

## Extracting Binned Signal

`extract()` bins signal around promoters, genes, transcripts, or exons.

```python
binned = qn.extract(
    feature_type="promoter",
    GTF_FILE="genes.gtf",
    assay="ATAC",
    modality="coverage",
    upstream=2000,
    downstream=2000,
    bin_size=50,
)
```

Returned value: `xr.DataArray` with dimensions `(interval, bin, sample)`.
Use this output with `metaplot()` or `tornadoplot()`.

## Normalisation

Normalise reduced signal, extracted signal, or count matrices:

```python
cpm_signal = qn.normalise(signal, method="cpm")
rpkm_counts = qn.normalise(
    counts,
    method="rpkm",
    feature_lengths=features["range_length"],
)
```

Library sizes are available directly:

```python
qn.library_sizes()
qn.library_sizes(assay="RNA")
```

## PCA

On a reduced matrix:

```python
pca_obj, pca_result = qn.pca(signal["mean"], n_components=8)
```

Here `pca_result` is sample-by-component output, ready for `pca_scatter()`.

Or directly from one chromosome and modality:

```python
pca_obj, pca_result = qn.pca(
    "chr21",
    assay="ATAC",
    modality="coverage",
    n_components=5,
)
```

## Visualisation

```python
qn.heatmap(signal, variable="mean", title="Promoter signal")
qn.correlate(signal, variable="mean", title="Sample correlation")
qn.metaplot(binned, modality="coverage", title="Metaplot")
qn.tornadoplot(binned, modality="coverage", title="Tornado plot")
```

Use `heatmap()` and `correlate()` on reduced matrices, and `metaplot()` / `tornadoplot()` on extracted binned signal.

## Annotation-Aware Queries

```python
qn = QuantNadoDataset("dataset/", annotation="genes.gtf")
info = qn.gene_info("GNAQ")
gene_region = qn.sel_gene("GNAQ", padding=2000)
```

## Combining Per-Sample Stores

```python
from quantnado import QuantNado

combined = QuantNado.combine("dataset/", "dataset/combined.zarr")
```

The combined store keeps the same read API and is often easier to move or share.

## SeqNado Integration

```python
from quantnado import metadata_from_seqnado

metadata = metadata_from_seqnado("my_seqnado_project", output_dir=".")
```

This writes a `quantnado_metadata.csv` that can be fed into `quantnado create-dataset`.

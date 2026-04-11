# Examples

These examples mirror the current API and CLI.

## Build a Dataset from Metadata

```bash
quantnado create-dataset \
  --metadata samples.csv \
  --output-dir dataset \
  --chromsizes hg38.chrom.sizes
```

Example metadata:

```csv
sample,assay,bam,methyl,vcf,ip
ATAC_1,ATAC,/data/ATAC_1.bam,,,
H3K27ac_1,ChIP,/data/H3K27ac_1.bam,,,H3K27ac
RNA_1,RNA,/data/RNA_1.bam,,,
```

## Open and Inspect

```python
from quantnado import QuantNado

qn = QuantNado.open("dataset/")

print(qn.sample_names)
print(qn.assays)
print(qn.array_keys)
print(qn.info)
```

## Cache Sample Groups

```python
qn.group_by(
    ip="ip",
    treatment={"control": ["control"], "treated": ["treated"]},
    replicate={"rep1": ["rep1"], "rep2": ["rep2"]},
    spikein={"spikein": ["spikein", "rx"]},
    match="contains",
)

print(qn.info)
```

With `match="contains"`, one label can match multiple substrings. Here the single `spikein` label groups both `spikein` and `rx` sample names.

## Reduce Promoter Signal

```python
promoter_signal = qn.reduce(
    intervals_path="promoters.bed",
    reduction="mean",
    modality="coverage",
)
```

This returns an `xr.Dataset`; `promoter_signal["mean"]` is usually the matrix you pass into PCA or clustering.

## RNA Signal Quantification

```python
signal_matrix, feature_meta = qn.quantify_signal(
    gtf_file="genes.gtf",
    feature_type="gene",
    assay="RNA",
    modality="coverage",
)
```

This returns a feature-by-sample matrix plus aligned feature metadata from stored QuantNado signal.

If you want the explicit counting API, use:

```python
counts, features = qn.count_features(
    gtf_file="genes.gtf",
    feature_type="gene",
    engine="signal",
    assay="RNA",
)
```

## PCA and QC

```python
pca_obj, pca_result = qn.pca(promoter_signal["mean"], n_components=10)
qn.pca_scree(pca_obj)
qn.pca_scatter(pca_obj, pca_result)

qn.heatmap(promoter_signal, variable="mean", title="Promoter signal")
qn.correlate(promoter_signal, variable="mean", title="Promoter correlation")
```

## Metaplot / Tornado Plot

```python
binned = qn.extract(
    feature_type="promoter",
    GTF_FILE="genes.gtf",
    assay="ATAC",
    modality="coverage",
    upstream=1000,
    downstream=1000,
    bin_size=50,
)

qn.metaplot(binned, modality="coverage", title="ATAC promoter profile")
qn.tornadoplot(binned, modality="coverage", title="ATAC promoter heatmap")
```

These plotting helpers expect the `xr.DataArray` returned by `extract()`.

## Combine Stores

```python
from quantnado import QuantNado

combined = QuantNado.combine("dataset/", "dataset/combined.zarr")
```

## SeqNado Helper

```python
from quantnado import metadata_from_seqnado

metadata = metadata_from_seqnado("seqnado_project", output_dir=".")
```

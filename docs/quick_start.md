# Quick Start

This guide walks through the current QuantNado workflow: per-sample inputs in, per-sample stores out, optional combined dataset, unified analysis API on top.

## Prerequisites

- QuantNado installed
- Indexed BAM files for BAM-based assays
- Optional methylation bedGraph files and VCF.gz files

## 1. Create Per-Sample Stores

```bash
quantnado dataset create \
  --sample ATAC_1 \
  --assay ATAC \
  --bamfile /data/ATAC_1.bam \
  --output-dir dataset \
  --chromsizes hg38.chrom.sizes

quantnado dataset create \
  --sample H3K27ac_1 \
  --assay ChIP \
  --bamfile /data/H3K27ac_1.bam \
  --ip H3K27ac \
  --output-dir dataset

quantnado dataset create \
  --sample RNA_1 \
  --assay RNA \
  --bamfile /data/RNA_1.bam \
  --stranded R \
  --output-dir dataset

quantnado dataset create \
  --sample METH_1 \
  --assay METH \
  --bamfile /data/METH_1.bam \
  --methylation_file /data/METH_1.bedGraph \
  --output-dir dataset

quantnado dataset create \
  --sample SNP_1 \
  --assay SNP \
  --vcf_file /data/SNP_1.vcf.gz \
  --output-dir dataset
```

This creates one `.zarr` store per sample:

```text
dataset/
  ATAC_1.zarr
  H3K27ac_1.zarr
  RNA_1.zarr
  METH_1.zarr
  SNP_1.zarr
```

For quick test builds, use either:

```bash
quantnado dataset create \
  --sample ATAC_1 \
  --assay ATAC \
  --bamfile /data/ATAC_1.bam \
  --output-dir dataset \
  --test
```

or an explicit chromosome list:

```bash
quantnado dataset create \
  --sample ATAC_1 \
  --assay ATAC \
  --bamfile /data/ATAC_1.bam \
  --output-dir dataset \
  --test-chrom chr21 \
  --test-chrom chr9
```

## 2. Optionally Combine Stores

```bash
quantnado dataset combine \
  --stores dataset/ATAC_1.zarr dataset/H3K27ac_1.zarr dataset/RNA_1.zarr dataset/METH_1.zarr dataset/SNP_1.zarr \
  --output dataset/combined.zarr
```

You can open either `dataset/` or `dataset/combined.zarr`.

## 3. Open the Dataset in Python

```python
from quantnado import QuantNado

qn = QuantNado.open("dataset/")

print(qn.sample_names)
print(qn.assays)
print(qn.array_keys)
print(qn.chromosomes[:5])
print(qn.info)
```

## 4. Select a Region

```python
region = qn.sel("chr21", 36_000_000, 36_010_000)
print(region)
```

`sel()` returns an `xr.Dataset` with one data variable per array key and shared `sample` / `position` coordinates.

## 5. Reduce Signal Over Intervals

```python
promoters = qn.reduce(
    intervals_path="promoters.bed",
    reduction="mean",
    modality="coverage",
)

print(promoters["mean"])
```

`reduce()` returns an `xr.Dataset` with summary variables such as `sum`, `count`, and `mean`.

For RNA-only signal quantification or assay-restricted analysis:

```python
rna_signal, features = qn.quantify_signal(
    gtf_file="genes.gtf",
    feature_type="gene",
    assay="RNA",
    modality="coverage",
)
```

`quantify_signal()` returns `(matrix_df, feature_metadata_df)`.

## 5b. Cache Sample Groups

```python
qn.group_by(
    ip="ip",
    treatment={"control": ["control"], "treated": ["treated"]},
    replicate={"rep1": ["rep1"], "rep2": ["rep2"]},
    spikein={"spikein": ["spikein", "rx"]},
    match="contains",
)

qn.info
```

With `match="contains"`, each label can match one or many substrings. For example, `"spikein": ["spikein", "rx"]` groups both RNA spike-in samples and ChIP spike-in `rx` samples under the same label.

## 6. Run PCA

```python
pca_obj, pca_result = qn.pca(promoters["mean"], n_components=8)
qn.pca_scree(pca_obj)
qn.pca_scatter(pca_obj, pca_result)
```

## 7. Extract Binned Signal for Plots

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

qn.metaplot(binned, modality="coverage", title="ATAC around promoters")
qn.tornadoplot(binned, modality="coverage", title="ATAC promoter heatmap")
```

`extract()` returns an `xr.DataArray` with dimensions `(interval, bin, sample)`.

## Next Steps

- [Usage Guide](basic_usage.md)
- [CLI Reference](cli/index.md)
- [API Reference](api/quantnado.md)
- [Examples](examples.md)

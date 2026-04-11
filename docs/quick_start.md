# Quick Start

This guide walks through the current QuantNado workflow: metadata table in, per-sample stores out, unified analysis API on top.

## Prerequisites

- QuantNado installed
- Indexed BAM files for BAM-based assays
- Optional methylation bedGraph files and VCF.gz files

## 1. Create a Metadata Table

```csv
sample,assay,bam,methyl,vcf,ip
ATAC_1,ATAC,/data/ATAC_1.bam,,,
H3K27ac_1,ChIP,/data/H3K27ac_1.bam,,,H3K27ac
RNA_1,RNA,/data/RNA_1.bam,,,
METH_1,METH,/data/METH_1.bam,/data/METH_1.bedGraph,,
SNP_1,SNP,,,/data/SNP_1.vcf.gz,
```

Required columns:

- `sample`
- `assay`

Input columns depend on assay:

- BAM-based assays use `bam`
- `METH` uses `bam` and `methyl`
- `SNP` uses `vcf`
- `ChIP` and `CUT&TAG` can also use `ip`

## 2. Build the Dataset

```bash
quantnado create-dataset \
  --metadata samples.csv \
  --output-dir dataset \
  --chromsizes hg38.chrom.sizes
```

This creates one `.zarr` store per metadata row:

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
quantnado create-dataset \
  --metadata samples.csv \
  --output-dir dataset \
  --test
```

or an explicit chromosome list:

```bash
quantnado create-dataset \
  --metadata samples.csv \
  --output-dir dataset \
  --test-chrom chr21 \
  --test-chrom chr9
```

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

## 8. Optionally Combine Stores

```python
from quantnado import QuantNado

combined = QuantNado.combine("dataset/", "dataset/combined.zarr")
```

You can then open `dataset/combined.zarr` with the same API.

## Next Steps

- [Usage Guide](basic_usage.md)
- [CLI Reference](cli/index.md)
- [API Reference](api/quantnado.md)
- [Examples](examples.md)

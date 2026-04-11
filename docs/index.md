# QuantNado

![logo](assets/images/logo.png){: style="display: block; margin: 0 auto; width: 200px;"}

QuantNado converts BAM-, bedGraph-, and VCF-derived assays into Zarr-backed stores and gives you one analysis API for region queries, feature reduction, counting, normalisation, PCA, plotting, and peak calling.

## Current Workflow

1. Prepare a metadata CSV/TSV describing your samples and assay types.
2. Run `quantnado create-dataset` to build one `.zarr` store per sample.
3. Open the dataset with `QuantNado.open(...)` or `QuantNadoDataset(...)`.
4. Optionally combine per-sample stores into a single multi-sample `.zarr`.

## Supported Assays

| Assay | Inputs | Typical array keys |
|---|---|---|
| `ATAC`, `ChIP`, `CUT&TAG` | BAM | `atac`, `chip_<ip>`, `cat_<ip>`, `coverage` |
| `MCC` | BAM | viewpoint-specific arrays |
| `METH` | BAM + methylation bedGraph | `methyl_pct`, `n_methylated`, `n_total`, `coverage` |
| `RNA` | BAM | `rna_fwd`, `rna_rev`, `coverage` |
| `SNP` | VCF.gz | `GT`, `AF`, `DP`, `MQ`, `coverage` |

## Minimal Example

### Metadata

samples.csv with the following:

|assay|sample_id|ip|bam_path|stranded|methylation_path|variant_path|
|---|---|---|---|---|---|---|
|ATAC |Sample-1||/data/ATAC-Sample-1.bam||||
|ChIP |Sample-1_H3K27ac|H3K27ac|/data/Sample-1_H3K27ac.bam||||
|MCC  |Sample-1||/data/MCC-Sample-1.bam||||
|METH |Sample-1||/data/METH-Sample-1.bam||/data/METH-Sample-1.bedGraph||
|RNA  |Sample-1||/data/RNA-Sample-1.bam|R|||
|SNP  |Sample-1||/data/SNP-Sample-1.bam|||/data/SNP-Sample-1.vcf.gz|

```bash
quantnado create-dataset \
  --metadata samples.csv \
  --output-dir dataset.zarr \
  --chromsizes hg38.chrom.sizes
```

```python
from quantnado import QuantNado

qn = QuantNado.open("dataset.zarr")

region = qn.sel("chr21", 36_000_000, 36_010_000)
signal = qn.reduce("promoters.bed", reduction="mean", modality="coverage")
matrix, features = qn.quantify_signal("genes.gtf", feature_type="gene", assay="RNA", modality="coverage")
qn.group_by(
    ip="ip",
    treatment={"control": ["control"], "treated": ["treated"]},
    match="contains",
)
qn.info
```

## Key Concepts

- A dataset can be either a directory of per-sample `.zarr` stores or a combined `.zarr`.
- `assay` filters samples by biological type such as `RNA` or `ATAC`.
- `modality` selects a concrete array key such as `coverage`, `rna_fwd`, or `chip_h3k27ac`.
- `group_by()` caches reusable sample-group namespaces, and `subset()` can intersect assay, sample, IP, and group filters.
- Most analysis methods return xarray objects, so they stay lazy until you compute or plot.
- `reduce()` returns an `xr.Dataset`, `extract()` returns an `xr.DataArray`, `quantify_signal()` returns a pandas feature matrix plus metadata from stored signal, and `count_features()` reserves room for backend-specific counting semantics.

## Documentation

- [Installation](installation.md)
- [Quick Start](quick_start.md)
- [Usage Guide](basic_usage.md)
- [CLI Overview](cli/index.md)
- [API Reference](api/quantnado.md)
- [Examples](examples.md)

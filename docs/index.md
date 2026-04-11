# QuantNado

![logo](assets/images/logo.png){: style="display: block; margin: 0 auto; width: 200px;"}

QuantNado converts BAM-, bedGraph-, and VCF-derived assays into Zarr-backed stores and gives you one analysis API for region queries, feature reduction, counting, normalisation, PCA, plotting, and peak calling.

## Current Workflow

1. Run `quantnado dataset create` once per sample to build per-sample `.zarr` stores.
2. Optionally run `quantnado dataset combine` to merge those stores into a single multi-sample `.zarr`.
3. Open the dataset with `QuantNado.open(...)` or `QuantNadoDataset(...)`.

## Supported Assays

| Assay | Inputs | Typical array keys |
|---|---|---|
| `ATAC`, `ChIP`, `CUT&TAG` | BAM | `atac`, `chip_<ip>`, `cat_<ip>`, `coverage` |
| `MCC` | BAM | viewpoint-specific arrays |
| `METH` | BAM + methylation bedGraph | `methyl_pct`, `n_methylated`, `n_total`, `coverage` |
| `RNA` | BAM | `rna_fwd`, `rna_rev`, `coverage` |
| `SNP` | VCF.gz | `GT`, `AF`, `DP`, `MQ`, `coverage` |

## Minimal Example

```bash
quantnado dataset create \
  --sample Sample-1 \
  --assay ATAC \
  --bamfile /data/ATAC-Sample-1.bam \
  --output-dir dataset \
  --chromsizes hg38.chrom.sizes

quantnado dataset create \
  --sample Sample-1_H3K27ac \
  --assay ChIP \
  --bamfile /data/Sample-1_H3K27ac.bam \
  --ip H3K27ac \
  --output-dir dataset

quantnado dataset combine \
  --stores dataset/Sample-1.zarr dataset/Sample-1_H3K27ac.zarr \
  --output dataset/combined.zarr
```

```python
from quantnado import QuantNado

qn = QuantNado.open("dataset/combined.zarr")

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

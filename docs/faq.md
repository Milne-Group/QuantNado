# FAQ

## What does QuantNado create now?

QuantNado currently creates one `.zarr` store per sample from a metadata CSV/TSV. You can then open the directory directly or combine those stores into a single multi-sample `.zarr`.

## What columns are required in the metadata file?

Always:

- `sample`
- `assay`

Depending on assay:

- BAM-based assays use `bam`
- `METH` uses `bam` and `methyl`
- `SNP` uses `vcf`
- `ChIP` and `CUT&TAG` can also use `ip`

## Which assays are supported?

`ATAC`, `ChIP`, `RNA`, `CUT&TAG`, `METH`, `SNP`, and `MCC`.

## How do I open a dataset in Python?

```python
from quantnado import QuantNado

qn = QuantNado.open("dataset/")
```

You can also open a combined store:

```python
qn = QuantNado.open("dataset/combined.zarr")
```

## What is the difference between `assay` and `modality`?

- `assay` filters samples by biological type, such as `RNA` or `ATAC`
- `modality` selects a concrete array key, such as `coverage`, `rna_fwd`, or `chip_h3k27ac`

Most analysis methods accept both.

## How do I combine per-sample stores?

```python
from quantnado import QuantNado

combined = QuantNado.combine("dataset/", "dataset/combined.zarr")
```

## How do I count RNA features?

```python
signal_matrix, features = qn.quantify_signal(
    gtf_file="genes.gtf",
    feature_type="gene",
    assay="RNA",
    modality="coverage",
)
```

Use `quantify_signal()` when you want a feature-by-sample matrix derived from stored QuantNado signal.

If you specifically want the counting API:

```python
counts, features = qn.count_features(
    gtf_file="genes.gtf",
    feature_type="gene",
    engine="signal",
    assay="RNA",
)
```

## What is the difference between `reduce()`, `extract()`, `quantify_signal()`, and `count_features()`?

- `reduce()` summarises signal over intervals and returns an `xr.Dataset`
- `extract()` bins signal around genomic features and returns an `xr.DataArray`
- `quantify_signal()` returns a feature-by-sample matrix plus feature metadata from stored signal
- `count_features(engine="signal")` uses the counting API shape but still summarises stored signal today

## How do grouping and subsetting work?

Use `group_by()` to cache reusable sample-group namespaces, then intersect them in `subset()`:

```python
qn.group_by(
    ip="ip",
    treatment={"control": ["control"], "treated": ["treated"]},
    spikein={"spikein": ["spikein", "rx"]},
    match="contains",
)

subset = qn.subset(
    assay="RNA",
    group={"treatment": "treated", "spikein": "spikein"},
)
```

All filters in `subset()` are combined with `AND`.

## Does QuantNado support peak calling?

Yes. `quantnado call-peaks` works directly from a QuantNado dataset and supports `quantile`, `seacr`, and `lanceotron`.

## Do BAM files need indexes?

Yes. BAM inputs should be coordinate-sorted and indexed with `.bai`.

```bash
samtools index sample.bam
```

## Does QuantNado support cloud storage?

The current workflow is documented around local filesystem paths. If you are working on remote storage, it is usually safest to build stores on the mounted filesystem you intend to use and validate performance there.

## Is the dataset writable after creation?

The read API is designed for analysis. Typical workflows treat created stores as analysis inputs and write derived results to separate files.

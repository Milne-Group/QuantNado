# QuantNado — Codebase Guide

High-performance genomic signal quantification. Stores per-bp coverage in Zarr v3 format with a unified multiomics read API backed by xarray.

---

## Dev setup

```bash
uv sync --all-extras
```

## Commands

```bash
# CLI
quantnado --help
quantnado create-dataset --help
quantnado call-peaks --help

# Tests
uv run pytest tests               # all
uv run pytest tests -m unit       # fast, no file I/O
uv run pytest tests -m integration
uv run pytest tests -m cli

# Lint
uv run ruff check .
uv run ruff format .
```

---

## Project layout

```
quantnado/
├── dataset/              # Write path — one builder per assay type
│   ├── store_bam.py      # BamStore — ATAC, ChIP, CUT&TAG, RNA, MCC
│   ├── store_methyl.py   # MethylStore — TAPS/WGBS (coverage + methylation)
│   ├── store_variants.py # VariantStore — SNP/VCF
│   └── metadata.py       # Zarr metadata helpers + domain enums
├── analysis/             # Read path — use this, not dataset/ directly
│   ├── core.py           # QuantNadoDataset — unified read-only xarray view
│   ├── normalise.py      # get_library_sizes(), normalise() (CPM/RPKM/TPM)
│   ├── reduce.py         # Signal aggregation over BED/GTF ranges
│   ├── features.py       # GTF feature extraction
│   ├── counts.py         # Feature counting (DESeq2-compatible)
│   ├── plot.py           # metaplot, tornadoplot, heatmap, correlate
│   ├── pca.py            # PCA via dask-ml
│   └── ranges.py         # Range operations
├── peak_calling/
│   ├── call_quantile_peaks.py   # Quantile threshold method
│   ├── call_seacr_peaks.py      # SEACR-style AUC island method
│   └── call_lanceotron_peaks.py # LanceOtron ML method (PyTorch)
├── cli.py                # Typer CLI — create-dataset, call-peaks
├── api.py                # QuantNado facade (unified entry point)
└── utils.py              # Logging, region parsing, chunk estimation
```

---

## Metadata table

The metadata CSV/TSV is the single source of truth for store creation.

| column | required | description |
|---|---|---|
| `assay` | yes | ATAC, ChIP, CUT&TAG, RNA, METH, SNP, MCC |
| `sample_name` | yes | unique identifier → zarr filename |
| `ip` | ChIP/CUT&TAG only | IP target e.g. H3K27ac, MLLN, Menin |
| `bam_path` | all except SNP | path to aligned BAM |
| `stranded` | RNA only | R (reverse), F (forward), U (unstranded) |
| `methylation_path` | METH only | bedGraph from MethylDackel |
| `variant_path` | SNP only | annotated VCF (.vcf.gz) |

Example:

| assay | sample_name | ip | bam_path | stranded | methylation_path | variant_path |
|---|---|---|---|---|---|---|
| ATAC | ATAC-SEM-1 | | aligned/ATAC-SEM-1.bam | | | |
| ATAC | ATAC-SEM-2 | | aligned/ATAC-SEM-2.bam | | | |
| CUT&TAG | CAT-HSC_H3K27ac | H3K27ac | aligned/CAT-HSC_H3K27ac.bam | | | |
| CUT&TAG | CAT-HSC_MLLN | MLLN | aligned/CAT-HSC_MLLN.bam | | | |
| ChIP | CM-RCHACV-1_MLLN | MLLN | aligned/CM-RCHACV-1_MLLN.bam | | | |
| ChIP | CM-RCHACV-1_Menin | Menin | aligned/CM-RCHACV-1_Menin.bam | | | |
| RNA | RNA-RCHACV-1 | | aligned/RNA-RCHACV-1.bam | R | | |
| RNA | RNA-RCHACV-2 | | aligned/RNA-RCHACV-2.bam | R | | |
| METH | TAPS-RCHACV | | aligned/TAPS-RCHACV.bam | | methylation/TAPS-RCHACV_CpG.bedGraph | |
| METH | TAPS-RS411 | | aligned/TAPS-RS411.bam | | methylation/TAPS-RS411_CpG.bedGraph | |
| SNP | gDNA-RCHACV | | | | | variant/gDNA-RCHACV.anno.vcf.gz |
| SNP | gDNA-SEM | | | | | variant/gDNA-SEM.anno.vcf.gz |
| MCC | MCC-OCI-AML3 | | aligned/OCI-AML3.bam | | | |

---

## Zarr store layout

### Per-sample store

Each sample gets its own `.zarr`. All stores share the same layout:

```
{SampleName}.zarr/
├── {chrom}/                          # one zarr group per chromosome
│   └── {array_key}  (1, chrom_len)  # shape (1, chrom_len), chunks (1, 65536)
├── metadata/
│   ├── completed        bool    (1,)
│   ├── total_reads      int64   (1,)
│   ├── mean_read_length float32 (1,)
│   └── sparsity         float32 (1,)
└── root.attrs: assay, sample, ip, chromsizes, chunk_len, stranded
```

**Array keys per assay** (written under each `{chrom}/` group):

| Assay | Array keys | dtype | Builder | Notes |
|---|---|---|---|---|
| ATAC | `coverage` | uint32 | BamStore | bamnado coverage |
| ChIP | `coverage` | uint32 | BamStore | bamnado coverage + ip column |
| CUT&TAG | `coverage` | uint32 | BamStore | bamnado coverage + ip column |
| RNA | `rna_fwd`, `rna_rev` | uint32 | BamStore | stranded; bamnado |
| METH | `coverage`, `methyl_pct`, `n_methylated`, `n_total` | uint32/float32/uint32/uint32 | MethylStore | bamnado coverage + bedGraph |
| SNP | `coverage`, `GT`, `DP`, `AF`(VCF FORMAT fields) | varies | VariantStore | bamnado coverage + VCF only |
| MCC | `coverage` per viewpoint | uint32 | BamStore | VP tag filter via bamnado |

**Key naming rules:**
- ChIP/CUT&TAG: `f"{assay}_{ip}".lower().replace("&", "")` → `chip_h3k27ac`, `cat_mlln`
- RNA: always two arrays `rna_fwd` + `rna_rev` (stranded stores only)
- MCC: one array per viewpoint `mcc_{viewpoint}`; sample names = `{bam_name}_{viewpoint}`; `root.attrs["viewpoints"]` lists all VP names
- SNP keys are derived from VCF FORMAT header fields

### Combined store

After combining, same-assay samples are stacked along axis 0 (`(1, L) × N → (N, L)`):

```
combined.zarr/
├── {chrom}/
│   ├── rna_fwd       (n_rna,  chrom_len)  uint32
│   ├── rna_rev       (n_rna,  chrom_len)  uint32
│   ├── coverage      (n_samples, chrom_len)  uint32
│   ├── methyl_pct    (n_meth, chrom_len)  float32
│   └── GT            (n_snp,  chrom_len)  int8
├── metadata/
│   ├── sample_names     str array (all samples, ordered)
│   ├── assay            str array (per sample)
│   ├── completed        bool
│   ├── total_reads      int64
│   ├── mean_read_length float32
│   └── sparsity         float32
└── root.attrs: assay_types=[...], array_keys=[...], chromsizes, chunk_len
```

---

## QuantNadoDataset (analysis API)

**Always use this for reading — never access zarr directly.**

```python
from quantnado.analysis.core import QuantNadoDataset

# Open a directory of per-sample zarrs or a combined zarr — auto-detected
qn = QuantNadoDataset("dataset/")
qn = QuantNadoDataset("dataset/combined.zarr")

# Properties
qn.sample_names   # list[str]
qn.assays         # list[str]  — biological assay types e.g. ['atac', 'chip', 'meth', 'rna', 'snp']
qn.array_keys     # list[str]  — all zarr data variable names e.g. ['atac', 'coverage', 'rna_fwd', 'AF', ...]
qn.chromosomes    # list[str]  — excludes 'metadata' group
qn.chromsizes     # dict[str, int]

# Region slice → xr.Dataset  (1-based coords)
region = qn.sel(chrom="chr1", start=1_000_000, end=1_001_000)
# xr.Dataset
#   dims:      sample × position
#   coords:    position=[1000000..1001000]  (1-based), sample=[...]
#   data_vars: atac, chip_h3k27ac, rna_fwd, rna_rev, coverage, methyl_pct, GT, ...

# Standard xarray slicing
region["atac"].sel(sample="ATAC-SEM-1")
region.sel(position=slice(1_000_100, 1_000_200))
region["atac"].plot()   # x-axis = genomic coords

# Full genome → xr.DataTree  (one node per chromosome)
tree = qn.to_datatree()
tree["chr1"].ds.sel(position=slice(1_000_000, 2_000_000))

# Combine per-sample zarrs (only completed samples included)
QuantNadoDataset.combine("dataset/", output="dataset/combined.zarr")
```

---

## Workflow

### Stage 1 — Create (per-sample, parallelisable)

Via Python:

```python
from quantnado.dataset.store_bam import BamStore
from quantnado.dataset.store_methyl import MethylStore
from quantnado.dataset.store_variants import VariantStore

BamStore.from_bam_files(
    bam_path="aligned/ATAC-SEM-1.bam",
    store_path="dataset/ATAC-SEM-1.zarr",
    assay="atac",
    sample="ATAC-SEM-1",
    chromsizes=chromsizes,
)

MethylStore.from_files(
    bam_path="aligned/TAPS-RCHACV.bam",
    methyl_path="methylation/TAPS-RCHACV_CpG.bedGraph",
    store_path="dataset/TAPS-RCHACV.zarr",
    sample="TAPS-RCHACV",
    chromsizes=chromsizes,
)

VariantStore.from_vcf(
    vcf_path="variant/gDNA-RCHACV.anno.vcf.gz",
    store_path="dataset/gDNA-RCHACV.zarr",
    sample="gDNA-RCHACV",
    chromsizes=chromsizes,
)
```

Or via CLI with a metadata CSV:

```bash
quantnado create-dataset \
  --metadata samples.csv \
  --output-dir dataset/ \
  --chromsizes hg38.chrom.sizes
```

### Stage 2 — Combine (optional)

```python
QuantNadoDataset.combine("dataset/", output="dataset/combined.zarr")
# stacks (1, chrom_len) × N → (N, chrom_len) per assay
# only completed samples included
```

`QuantNadoDataset` reads both formats with the same API.

---

## Peak calling

All callers: `QuantNadoDataset` in, one BED file per sample out.

| Method | File | Use case |
|---|---|---|
| `quantile` | `call_quantile_peaks.py` | Fast, simple threshold |
| `seacr` | `call_seacr_peaks.py` | AUC island calling (CUT&RUN/ATAC) |
| `lanceotron` | `call_lanceotron_peaks.py` | ML classifier (ChIP-seq) |

```bash
quantnado call-peaks \
  --zarr <path> \
  --method [quantile|seacr|lanceotron] \
  --assay atac \
  --output-dir <path>
```

### Adding a new peak caller

1. Create `quantnado/peak_calling/call_{name}_peaks.py`
2. Implement `call_{name}_peaks_from_zarr(zarr_path, output_dir, assay, ...) -> list[str]`:
   - Open with `QuantNadoDataset(zarr_path)`
   - Iterate valid samples (where `completed` is True)
   - Write one BED per sample to `output_dir`
3. Add `elif method == "{name}":` in `cli.py`
4. Export from `peak_calling/__init__.py`

---

## Dependencies

Core: `zarr>=3`, `numpy`, `pandas`, `xarray`, `dask`, `scipy`, `pyranges1`, `loguru`, `typer`, `bamnado`, `pysam`

Optional:
- `pip install quantnado[lanceotron]` → adds `torch>=2.0`
- Dev: `pytest`, `ruff`, `mkdocs-material`

---

## Tests

```
tests/
├── unit/          # Pure-numpy / no I/O  (pytest -m unit)
├── integration/   # Requires real zarr stores  (pytest -m integration)
└── cli/           # CLI smoke tests  (pytest -m cli)
```

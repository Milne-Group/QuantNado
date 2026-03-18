# QuantNado

Zarr-backed genomic quantification platform for multi-modal NGS data (BAM coverage, methylation, VCF variants). Stores signals as chunked Zarr v3 arrays and exposes lazy xarray/dask DataArrays for downstream analysis.

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

# Docs
uv run mkdocs serve
```

## Architecture

### Data stores

| Class | File | Purpose |
|---|---|---|
| `MultiomicsStore` | `dataset/store_multiomics.py` | Orchestrates all sub-stores in one directory |
| `BamStore` | `dataset/store_bam.py` | BAM → per-chromosome Zarr arrays |
| `MethylStore` | `dataset/store_methyl.py` | bedGraph/CXreport → Zarr |
| `VariantStore` | `dataset/store_variants.py` | VCF → Zarr |
| `BaseStore` | `dataset/core.py` | Abstract base: chromsizes, metadata, xarray conversion |

### High-level facade

`QuantNado` (`api.py`) wraps a `MultiomicsStore` and exposes analysis methods: `reduce()`, `count_features()`, `pca()`, `heatmap()`, `metaplot()`, `tornadoplot()`, `locus_plot()`, `correlate()`.

### Key enums (`dataset/enums.py`)

- `CoverageType`: `unstranded` | `stranded` | `mcc` (Micro-Capture C)
- `FeatureType`: `gene` | `transcript` | `exon` | `promoter`
- `ReductionMethod`: `mean` | `sum` | `max` | `min` | `median`
- `AnchorPoint`: `midpoint` | `start` | `end`

### Data flow

```
BAM / bedGraph / VCF
        ↓
MultiomicsStore.from_files()
        ↓
Zarr v3 arrays  (coverage.zarr / methylation.zarr / variants.zarr)
        ↓
BaseStore.to_xarray()  →  lazy xarray.DataArray backed by dask
        ↓
analysis/  (reduce, counts, pca, normalise, plot)
```

## Directory structure

```
quantnado/
  api.py              # QuantNado facade
  cli.py              # Typer CLI (create-dataset, call-peaks)
  dataset/            # Store implementations + enums
  analysis/           # reduce, counts, pca, normalise, plot, features, ranges
  peak_calling/       # Quantile-based peak caller
  utils.py
tests/
  unit/               # No file I/O, fast
  integration/        # Requires BamStore / filesystem
  cli/
  data/               # Small BAM files for integration tests
docs/                 # MkDocs source
```

## Key dependencies

- `bamnado >=0.5.6` — BAM file processing (primary data ingestion)
- `zarr >=3,<4` — chunked array storage backend
- `xarray / dask` — lazy N-D arrays with labeled dims
- `pyranges1` — genomic range operations
- `typer / loguru` — CLI and structured logging

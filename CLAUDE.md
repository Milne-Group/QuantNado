# QuantNado — Codebase Guide

High-performance genomic signal quantification and peak calling. Stores per-bp coverage in Zarr v3 format and provides Python / CLI APIs for signal analysis.

---

## Project layout

```
quantnado/
├── dataset/              # Zarr store construction (write path)
│   ├── store_bam.py      # BamStore — per-chromosome uint32 coverage arrays
│   ├── store_methyl.py   # Methylation store
│   ├── store_variants.py # VCF/variant store
│   ├── store_multiomics.py
│   └── metadata.py       # Zarr metadata extraction helpers
├── analysis/             # Read/analysis API (use this, not dataset/ directly)
│   ├── core.py           # QuantNadoDataset — lightweight read-only Zarr wrapper
│   ├── normalise.py      # get_library_sizes(), normalise() (CPM/RPKM/TPM)
│   ├── reduce.py         # Signal aggregation over BED/GTF ranges
│   ├── features.py       # Feature-level extraction
│   ├── counts.py         # Feature counting (DESeq2-compatible)
│   ├── plot.py           # metaplot, tornadoplot, heatmap, correlate
│   ├── pca.py            # PCA via dask-ml
│   └── ranges.py         # Range operations
├── peak_calling/
│   ├── call_quantile_peaks.py   # Quantile threshold method
│   ├── call_seacr_peaks.py      # SEACR-style AUC island method (pure Python)
│   ├── call_lanceotron_peaks.py # LanceOtron ML method (PyTorch)
│   └── static/lanceotron/       # Pre-converted model weights + scaler .npy files
├── cli.py                # Typer CLI — `quantnado call-peaks`, `create-dataset`
├── api.py                # QuantNado facade (unified entry point)
└── utils.py              # Logging setup, region parsing, chunk estimation
```

---

## Key data structures

### Zarr store layout (BamStore)

```
root/
├── {chrom}              # zarr array, shape (n_samples, chrom_len), dtype uint32
│                        # chunks (1, 65536) — one sample per chunk
├── {chrom}_fwd / _rev   # stranded arrays (optional)
└── metadata/
    ├── completed        # bool (n_samples,) — marks finished samples
    ├── total_reads      # int64 (n_samples,) — mapped reads per sample
    ├── mean_read_length # float32 (n_samples,)
    ├── sparsity         # float (n_samples,)
    └── sample_names     # string array
root.attrs: sample_names, chromsizes, chunk_len, stranded
```

### QuantNadoDataset (analysis API)

Read-only wrapper. **Always use this rather than accessing Zarr directly.**

```python
from quantnado.analysis.core import QuantNadoDataset
from quantnado.analysis.normalise import get_library_sizes

ds = QuantNadoDataset("/path/to/store.zarr")
ds.sample_names        # list[str]
ds.completed_mask      # np.ndarray[bool]
ds.chromosomes         # list[str]  (excludes 'metadata' group)
ds.chromsizes          # dict[str, int]

# Extract whole chromosome as numpy (shape: n_samples × chrom_len)
arr = ds.extract_region(chrom="chr1", as_xarray=False)  # uint32

# Extract per sample
arr = ds.extract_region(chrom="chr1", samples=["s1"], as_xarray=False)  # (1, chrom_len)

# Lazy xarray (all chroms)
xr_dict = ds.to_xarray()  # dict[chrom -> xr.DataArray(sample, position)]

# Library sizes for RPKM etc.
lib_sizes = get_library_sizes(ds)  # pd.Series indexed by sample name
```

---

## Peak calling methods

All methods share the same CLI entry point and zarr-in / BED-out contract.

| Method | File | Use case |
|--------|------|----------|
| `quantile` | `call_quantile_peaks.py` | Fast, simple threshold on tiled signal |
| `seacr` | `call_seacr_peaks.py` | AUC island calling (CUT&RUN/ATAC) |
| `lanceotron` | `call_lanceotron_peaks.py` | ML classifier (ChIP-seq / broad marks) |

### CLI

```bash
quantnado call-peaks \
  --zarr <path>                 # QuantNado zarr store
  --method [quantile|seacr|lanceotron]
  --output-dir <path>

# lanceotron-specific
  --score-threshold 0.5         # overall_classification cutoff
  --smooth-window 400           # rolling mean window for candidates (bp)
  --batch-size 512              # inference batch size
```

---

## Adding a new peak caller

1. Create `quantnado/peak_calling/call_{name}_peaks.py`
2. Implement a `call_{name}_peaks_from_zarr(zarr_path, output_dir, ...) -> list[str]` entry point that:
   - Opens the store with `QuantNadoDataset`
   - Iterates `valid_samples = [s for s, c in zip(ds.sample_names, ds.completed_mask) if c]`
   - Writes one BED file per sample to `output_dir`
   - Returns list of output paths
3. Add `elif method == "{name}":` dispatch in `cli.py`
4. Expose from `peak_calling/__init__.py`

---

## Dependencies

Core: `zarr>=3`, `numpy`, `pandas`, `xarray`, `dask`, `scipy`, `pyranges1`, `loguru`, `typer`, `bamnado`, `pysam`

Optional extras:
- `pip install quantnado[lanceotron]` → adds `torch>=2.0`, `scipy`
- Dev: `pytest`, `ruff`, `mkdocs-material`

---

## Tests

```
tests/
├── unit/          # Pure-numpy / no I/O  (pytest -m unit)
├── integration/   # Requires BamStore    (pytest -m integration)
└── cli/           # CLI smoke tests      (pytest -m cli)
```

Run: `pytest` from project root (respects `testpaths = ["tests"]` in `pyproject.toml`).

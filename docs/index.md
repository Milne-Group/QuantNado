# QuantNado

**Fast, efficient processing of multi-modal NGS data into genomic datasets**

QuantNado streamlines the processing of ChIP-seq, ATAC-seq, CUT&Tag, and other sequencing experiments. Convert aligned BAM files into indexed Zarr datasets for downstream analysis, or call peaks directly from alignment tracks.


![logo](assets/images/logo.png){: style="display: block; margin: 0 auto; width: 200px;"}

## What QuantNado Does

- **BAM → Zarr conversion**: Transform BAM files into efficient, queryable genomic datasets
- **Quantile-based peak calling**: Detect enriched regions from bigWig tracks
- **Parallel processing**: Process multiple samples efficiently with multi-threading
- **Zarr-backed storage**: Space-efficient storage with lazy loading capabilities

## Key Features

✓ **Fast**: Multi-threaded BAM processing  
✓ **Efficient**: Zarr storage compresses and streams data efficiently  
✓ **Flexible**: Works with any organism and any genome build  
✓ **Reproducible**: Full metadata tracking and resume support  
✓ **Accessible**: Simple CLI or full Python API  

## Common Workflows

### Process Aligned BAM Files

```bash
quantnado create-dataset *.bam \
  --output my_dataset.zarr \
  --chromsizes hg38.chrom.sizes \
  --max-workers 8
```

### Call Peaks from Signal Tracks

```bash
quantnado call-peaks \
  --bigwig-dir ./signal_tracks/ \
  --output-dir ./peaks/ \
  --chromsizes hg38.chrom.sizes \
  --quantile 0.98
```

## Documentation

- **[Installation](installation.md)** - Setup and installation
- **[Quick Start](quick_start.md)** - First analysis in 10 minutes
- **[Basic Usage](basic_usage.md)** - Common analysis patterns
- **[CLI Reference](cli.md)** - All command-line options
- **[API Reference](api/quantnado.md)** - Python API documentation
- **[FAQ](faq.md)** - Common questions
- **[Troubleshooting](troubleshooting.md)** - Solving common issues

## Citation

If you use QuantNado, please cite:

```
QuantNado: Efficient genomic signal quantification and peak calling
Milne Group, University of Oxford
```

# Call Peaks Command

Call quantile-based peaks from bigWig files.

## Usage

```bash
quantnado call-peaks [OPTIONS]
```

## Description

The `call-peaks` command identifies peaks in bigWig files using quantile-based thresholding. It's useful for determining enriched regions from ChIP-seq, ATAC-seq, and similar assays.

## Options

### BigWig Directory (`--bigwig-dir`) {: #bigwig-dir}

**Required.** Directory containing bigWig (.bw) files:

```bash
quantnado call-peaks --bigwig-dir ./bigwigs/ --output-dir ./peaks/ --chromsizes hg38.chrom.sizes
```

All .bw files in the directory will be processed.

### Output Directory (`--output-dir`)

**Required.** Directory to save peak files in BED format:

```bash
quantnado call-peaks \
  --bigwig-dir ./bigwigs/ \
  --output-dir ./peaks/ \
  --chromsizes hg38.chrom.sizes
```

Output BED files will be named: `<bigwig_name>.peaks.bed`

### Chromsizes (`--chromsizes`)

**Required.** Path to chromsizes file:

```bash
quantnado call-peaks \
  --bigwig-dir ./bigwigs/ \
  --output-dir ./peaks/ \
  --chromsizes /reference/hg38.chrom.sizes
```

### Quantile (`--quantile`)

Quantile threshold for peak calling (default: 0.98):

```bash
# Stricter threshold - fewer, higher-confidence peaks
quantnado call-peaks --bigwig-dir ./bigwigs/ --output-dir ./peaks/ \
  --chromsizes hg38.chrom.sizes --quantile 0.99

# More relaxed threshold - more peaks
quantnado call-peaks --bigwig-dir ./bigwigs/ --output-dir ./peaks/ \
  --chromsizes hg38.chrom.sizes --quantile 0.95
```

Higher quantile = fewer peaks (more stringent)

| Quantile | Interpretation |
|----------|----------------|
| 0.90 | 90th percentile - relaxed |
| 0.95 | 95th percentile - standard |
| 0.98 | 98th percentile - stringent |
| 0.99 | 99th percentile - very stringent |

### Tile Size (`--tilesize`)

Size of genomic tiles for peak calling (default: 128 bp):

```bash
quantnado call-peaks --bigwig-dir ./bigwigs/ --output-dir ./peaks/ \
  --chromsizes hg38.chrom.sizes --tilesize 50
```

Smaller tiles = more sensitive to local changes
Larger tiles = more robust to noise

### Blacklist (`--blacklist`)

Path to BED file with regions to exclude:

```bash
quantnado call-peaks \
  --bigwig-dir ./bigwigs/ \
  --output-dir ./peaks/ \
  --chromsizes hg38.chrom.sizes \
  --blacklist /reference/hg38-blacklist.bed
```

Blacklist format (tab-separated):
```
chr1    1000    2000
chr2    5000    6000
```

### Merge (`--merge` / `--no-merge`)

Merge overlapping peaks (default: disabled with `--no-merge`):

```bash
# Enable merging
quantnado call-peaks \
  --bigwig-dir ./bigwigs/ \
  --output-dir ./peaks/ \
  --chromsizes hg38.chrom.sizes \
  --merge

# Disable merging (default)
quantnado call-peaks \
  --bigwig-dir ./bigwigs/ \
  --output-dir ./peaks/ \
  --chromsizes hg38.chrom.sizes \
  --no-merge
```

### Temporary Directory (`--tmp-dir`)

Temporary directory for intermediate files (default: `tmp`):

```bash
quantnado call-peaks \
  --bigwig-dir ./bigwigs/ \
  --output-dir ./peaks/ \
  --chromsizes hg38.chrom.sizes \
  --tmp-dir /scratch/tmp
```

### Log File (`--log-file`)

Save logs (default: `quantnado_peaks.log`):

```bash
quantnado call-peaks \
  --bigwig-dir ./bigwigs/ \
  --output-dir ./peaks/ \
  --chromsizes hg38.chrom.sizes \
  --log-file peak_calling.log
```

### Verbose (`--verbose`, `-v`)

Enable debug logging:

```bash
quantnado call-peaks \
  --bigwig-dir ./bigwigs/ \
  --output-dir ./peaks/ \
  --chromsizes hg38.chrom.sizes \
  --verbose
```

## Examples

### Basic Peak Calling

Call peaks from bigWig files:

```bash
quantnado call-peaks \
  --bigwig-dir ./bigwigs/ \
  --output-dir ./peaks/ \
  --chromsizes hg38.chrom.sizes
```

### Stringent Thresholding

Call only high-confidence peaks:

```bash
quantnado call-peaks \
  --bigwig-dir ./bigwigs/ \
  --output-dir ./peaks/ \
  --chromsizes hg38.chrom.sizes \
  --quantile 0.99 \
  --merge
```

### With Blacklist

Exclude problematic regions:

```bash
quantnado call-peaks \
  --bigwig-dir ./bigwigs/ \
  --output-dir ./peaks/ \
  --chromsizes hg38.chrom.sizes \
  --blacklist /reference/hg38-blacklist.bed \
  --quantile 0.98
```

### Fine-Tuned Tile Size

Use smaller tiles for better resolution:

```bash
quantnado call-peaks \
  --bigwig-dir ./bigwigs/ \
  --output-dir ./peaks/ \
  --chromsizes hg38.chrom.sizes \
  --tilesize 50 \
  --quantile 0.98
```

### Full Pipeline

Comprehensive peak calling with all options:

```bash
quantnado call-peaks \
  --bigwig-dir ./bigwigs/ \
  --output-dir ./peaks/ \
  --chromsizes /reference/hg38.chrom.sizes \
  --blacklist /reference/hg38-blacklist.bed \
  --quantile 0.98 \
  --tilesize 128 \
  --merge \
  --tmp-dir /scratch/tmp \
  --log-file peak_calling.log \
  --verbose
```

## Output

Output BED files contain called peaks:

```
chr1    1000    2000    peak_1    100    .
chr1    5000    6500    peak_2    150    .
chr2    10000   11000   peak_3    120    .
```

Columns:
1. Chromosome
2. Start position
3. End position
4. Peak name
5. Score (signal intensity)
6. Strand (always `.` for this implementation)

## Performance

### Typical Run Times

Peak calling time depends on bigWig file size:

| Genome Coverage | Time |
|-----------------|------|
| Low coverage | 1-5 minutes |
| Medium coverage | 5-15 minutes |
| High coverage | 15-30 minutes |

### Resource Requirements

- **Memory**: 2-8 GB (depends on tile size and coverage)
- **Disk**: ~2-3× bigWig file size (for temporary files)

## Troubleshooting

### No bigWig files found

**Problem**: Directory is empty or wrong format

**Solution**: Check directory and file extensions:
```bash
ls -l ./bigwigs/*.bw
```

### Missing chromsizes

**Problem**: `FileNotFoundError: chromsizes file not found`

**Solution**: Download or provide chromsizes:
```bash
# Download from UCSC
wget http://hgdownload.cse.ucsc.edu/goldenpath/hg38/bigZips/hg38.chrom.sizes
```

### Out of memory

**Problem**: Memory error during peak calling

**Solution**: Use larger tile size (fewer tiles):
```bash
quantnado call-peaks \
  --bigwig-dir ./bigwigs/ \
  --output-dir ./peaks/ \
  --chromsizes hg38.chrom.sizes \
  --tilesize 256
```

### No peaks detected

**Problem**: Output peak files are empty

**Likely causes**:
- Quantile threshold too stringent
- BigWig files have low signal
- Incorrect chromsizes

**Solution**: Lower quantile threshold:
```bash
quantnado call-peaks \
  --bigwig-dir ./bigwigs/ \
  --output-dir ./peaks/ \
  --chromsizes hg38.chrom.sizes \
  --quantile 0.90
```

## Preparing BigWig Files

Convert BAM to bigWig if needed:

```bash
# Create bedGraph
bedtools genomecov -ibam sample.bam -bg -o sample.bedgraph

# Convert to bigWig
bedGraphToBigWig sample.bedgraph hg38.chrom.sizes sample.bw
```

## Python Equivalent

There's currently no direct Python API for peak calling. Use the CLI command.

## See Also

- [Create Dataset Command](create_dataset.md)
- [CLI Overview](index.md)
- [Examples](../examples.md)
- [Basic Usage](../basic_usage.md)

# Frequently Asked Questions

## Dataset Creation

**Q: How long does BAM to Zarr conversion take?**

A: Processing time depends mainly on BAM file size:
- **Small samples** (< 10 million reads): 5-10 minutes
- **Medium samples** (10-100M reads): 20-60 minutes
- **Large whole-genome samples** (> 100M reads): 1-4 hours

Use `--max-workers` to parallelize across multiple threads.

**Q: Which BAM files are compatible?**

A: Any BAM file can be processed as long as it:
- Contains aligned reads
- Has a corresponding `.bai` index file
- Uses standard CIGAR string notation

If your BAM lacks an index:
```bash
samtools index sample.bam
```

**Q: Can I process paired-end and single-end reads together?**

A: Yes. QuantNado counts aligned positions, not fragments, so both read types are handled uniformly.

**Q: What if I have samples with different sequencing depths?**

A: This is fine. QuantNado stores raw counts. For comparison, normalize using standard methods (RPKM, CPM, quantile normalization) in downstream analysis.

## Peak Calling

**Q: What quantile should I use?**

A: The quantile controls peak stringency:
- **0.95** - Lenient, more peaks
- **0.98** - Standard (recommended)
- **0.99** - Stringent, fewer peaks

Start with 0.98 and adjust based on your experiment and goals.

**Q: What do I do if no peaks are called?**

A: Possible causes:
1. **Quantile too high** - Try lower values (0.90-0.95)
2. **Sample quality** - Check bigWig files for signal
3. **Blacklist too aggressive** - Temporarily disable with `--blacklist`

## Storage and Performance

**Q: How much disk space does a Zarr dataset use?**

A: Roughly **0.5-2x the original BAM size**, depending on compression:
- For 100 samples × 100M reads each: ~50-200 GB

**Q: Can I access data remotely (cloud storage)?**

A: Currently requires local filesystem. Consider copying data locally first:
```bash
cp -r s3://bucket/data ./local_path
```

**Q: Can I delete intermediate files to save space?**

A: Yes, after dataset creation completes. The only essential file is the `.zarr` directory.

## Python API

**Q: How do I load a dataset in Python?**

A:
```python
from quantnado import QuantNado
qn = QuantNado.open("my_dataset.zarr")
```

**Q: How do I subset samples?**

A:
```python
data = qn["chr1"].sel(sample=["sample1", "sample2"])
```

**Q: Can I write modified data back to the dataset?**

A: No, datasets are read-only. Export to new formats for storage:
```python
df = qn["chr1"].to_dataframe()
df.to_parquet("chr1.parquet")
```

## Troubleshooting

**Q: I get "No such option: --max-workers" error**

A: Upgrade QuantNado:
```bash
pip install --upgrade quantnado
```

**Q: BAM indexing fails with permission errors**

A: Ensure you have write permissions:
```bash
chmod u+w *.bam*
samtools index sample.bam
```

**Q: Metadata CSV not recognized**

A: Check CSV format:
- First row must be headers
- Must include a `sample_id` column matching BAM file stems
- Use UTF-8 encoding

See [Basic Usage](basic_usage.md) for more examples.

Ensure BAM files are indexed:

```bash
samtools index file.bam
```

### What chromsizes file should I use?

QuantNado supports standard `.chrom.sizes` files:

- **NCBI** - Available from NCBI for each genome
- **UCSC** - Standard format from UCSC

Example format:

```
chr1    248956422
chr2    242193529
chr3    198295559
...
```

### Can I use custom chromosomes?

Yes! Provide a custom dict:

```python
chromsizes = {
    "chr1": 248956422,
    "chr2": 242193529,
    "custom_region": 100000
}

qn = QuantNado.from_bam_files(
    bam_files=[...],
    store_path="dataset.zarr",
    chromsizes=chromsizes
)
```

### What metadata formats are supported?

Metadata should be a CSV file with a sample_id column matching BAM file stems:

```csv
sample_id,condition,replicate,quality
sample1,treated,1,high
sample2,treated,2,high
sample3,control,1,medium
```

## Usage Questions

### How do I select specific samples?

Samples are identified by their BAM file stems. Select them by name:

```python
qn = QuantNado.open("dataset.zarr")
print(qn.samples)  # See all available samples

# Select in analysis
signal = qn.reduce(intervals_path="regions.bed")
chip_samples = signal.sel(sample=['sample1', 'sample2'])
```

### What if my sample name has spaces or special characters?

The sample name is derived from the BAM filename stem. For clean names:

- Use underscores instead of spaces: `H3K4me3_R1.bam`
- Avoid special characters: `chip-seq_sample1.bam` → `chip-seq_sample1`

Rename BAM files before processing if needed.

### Can I process partial datasets?

Yes! QuantNado supports resuming interrupted processing:

```python
# Resume will skip completed samples
qn = QuantNado.from_bam_files(
    bam_files=[...],
    store_path="dataset.zarr",
    resume=True
)
```

### What's the difference between reduce() and extract()?

- **`reduce()`**: Returns aggregated (summed/mean/etc.) signal over ranges
- **`extract()`**: Returns per-position signal over ranges
- **`feature_counts()`**: Returns integer counts for each feature

```python
# reduce - returns (n_regions, n_samples)
signal = qn.reduce(intervals_path="regions.bed", reduction="mean")

# extract - returns (n_regions, max_position, n_samples)
signal = qn.extract(intervals_path="regions.bed")

# feature_counts - returns (n_features, n_samples) integer matrix
counts, features = qn.feature_counts(gtf_file="genes.gtf")
```

### How do I handle NaN values?

QuantNado provides several strategies:

```python
pca, transformed = qn.pca(
    data,
    nan_handling_strategy="drop"  # Remove features with NaN
)

# Other options:
# - "set_to_zero": Replace NaN with 0
# - "mean_value_imputation": Replace with feature mean
```

## Performance & Resources

### How much disk space does a Zarr dataset need?

Zarr datasets are stored efficiently. Approximate sizes:

- Full genome (human), 1 sample: 500 MB - 2 GB (depending on sequencing depth)
- Full genome, 10 samples: 5 - 20 GB
- Specific chromosomes: Proportional to size

Calculate before processing:

```python
# Depends on read depth
# BAM file size ≈ final Zarr size (usually slightly smaller)
```

### How long does dataset creation take?

**Time scales with**:

- Number of BAM files
- Read depth (sequencing coverage)
- Chromosome complexity
- Hardware (CPU Speed, I/O)

Typical times (human genome, single sample):
- Low coverage (5M reads): ~1 minute
- Medium coverage (50M reads): ~5-10 minutes
- High coverage (200M reads): ~20-40 minutes

### Can I parallelize processing?

Use the `max_workers` parameter:

```bash
quantnado create-dataset *.bam \
  --output dataset.zarr \
  --max-workers 8
```

### How much RAM do I need?

Minimum: 4 GB for testing
Recommended: 16+ GB for production datasets

For very large datasets, use Dask's out-of-core capabilities.

## Troubleshooting

### Why is my dataset incomplete?

Check completion status:

```python
qn = QuantNado.open("dataset.zarr")
print(qn.store.completed_mask)
print(qn.store.n_completed, "/", qn.store.n_samples)
```

Resume processing:

```python
qn = QuantNado.from_bam_files(..., resume=True)
```

### My BAM file isn't being recognized

Ensure:
- BAM file is valid: `samtools view -H file.bam`
- BAM file is indexed: `samtools index file.bam`
- BAM has .bam extension
- Path is correct

### How do I debug issues?

Enable verbose logging:

```bash
quantnado create-dataset *.bam \
  --output dataset.zarr \
  --verbose \
  --log-file debug.log
```

## Advanced Questions

### Can I stream data directly from S3?

Currently, QuantNado works best with local BAM files, but Zarr supports S3 via:

```python
store = zarr.open_consolidated(
    "s3://bucket/dataset.zarr",
    mode="r"
)
```

### Can I use QuantNado with cloud storage?

Zarr datasets can be stored on S3, but BAM input processing currently requires local files. Zarr stores can be read from S3.

### How do I export results to HDF5?

Use xarray's built-in support:

```python
xr_dataset = qn.to_xarray()
xr_dataset.to_netcdf("output.h5")
```

Or export to commonly used formats:

```python
signal = qn.reduce(intervals_path="regions.bed")
signal["mean"].to_pandas().to_csv("signal.csv")
signal["mean"].to_pandas().to_hdf("signal.h5", key="signal")
```

## Getting Help

- **Documentation**: [Full docs](index.md)
- **GitHub Issues**: [Report bugs](https://github.com/Milne-Group/QuantNado/issues)
- **GitHub Discussions**: [Ask questions](https://github.com/Milne-Group/QuantNado/discussions)
- **Troubleshooting**: [Common issues](troubleshooting.md)

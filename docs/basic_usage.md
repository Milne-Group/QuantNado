# Basic Usage

Common workflows and analysis patterns with QuantNado.

## Creating Datasets

### From Single BAM File

```bash
quantnado create-dataset sample.bam \
  --output sample.zarr \
  --chromsizes hg38.chrom.sizes
```

### From Multiple BAM Files

Process all BAM files in a directory:

```bash
quantnado create-dataset results/aligned/*.bam \
  --output my_cohort.zarr \
  --chromsizes hg38.chrom.sizes \
  --max-workers 8
```

### With Sample Metadata

Create a CSV with sample information (`metadata.csv`):

```
sample_id,condition,replicate
sample1,control,1
sample2,control,2
sample3,treatment,1
sample4,treatment,2
```

Then:

```bash
quantnado create-dataset *.bam \
  --output dataset.zarr \
  --chromsizes hg38.chrom.sizes \
  --metadata metadata.csv
```

## Peak Calling

### Basic Peak Calling

```bash
quantnado call-peaks \
  --bigwig-dir ./bigwigs/ \
  --output-dir ./peaks/ \
  --chromsizes hg38.chrom.sizes \
  --quantile 0.98
```

Output: One BED file per sample in `./peaks/`

### Filtering and Merging

```bash
quantnado call-peaks \
  --bigwig-dir ./bigwigs/ \
  --output-dir ./peaks/ \
  --chromsizes hg38.chrom.sizes \
  --quantile 0.95 \
  --merge  # Merge overlapping peaks
```

## Python API

### Loading a Dataset

```python
from quantnado import QuantNado

# Open existing dataset
qn = QuantNado.open("my_dataset.zarr")

# Check samples
print(qn.samples)
print(qn.metadata)
```

### Accessing Data

```python
# Get data for a specific region (returns xarray DataArray)
region_data = qn.extract_region("chr1:1000000-2000000")

# Get all chromosomes as a dict of lazy xarray DataArrays
all_chroms = qn.to_xarray()
chr1_data = all_chroms["chr1"]

# Compute and convert to pandas DataFrame
df = chr1_data.to_dataframe(name="signal")
```

### Creating Datasets Programmatically

```python
from quantnado import BamStore
import pandas as pd

# Create from BAM files
store = BamStore.from_bam_files(
    bam_files=["sample1.bam", "sample2.bam"],
    chromsizes="hg38.chrom.sizes",
    store_path="dataset.zarr",
    max_workers=8
)

# Add metadata
metadata = pd.DataFrame({
    "sample_id": ["sample1", "sample2"],
    "condition": ["control", "treatment"]
})
store.set_metadata(metadata)
```

## Common Issues

**Q: How long does dataset creation take?**  
A: Depends on BAM file size (typically 20 min - 2 hours for whole-genome). Use `--max-workers` to parallelize.

**Q: Can I resume interrupted dataset creation?**
A: Yes. Pass `--resume` to the CLI (or `resume=True` in Python) and QuantNado will skip already-completed samples. Without this flag the default behaviour is `--overwrite`, which deletes any existing store at that path.

**Q: What genomic coordinates are supported?**  
A: Any coordinates in your chromsizes file. Supports human, mouse, yeast, etc.

See [Troubleshooting](troubleshooting.md) for more help.

# Troubleshooting

Solutions for common issues when using QuantNado.

## Installation Issues

### Python 3.14 Compatibility Error

**Problem:** `the configured Python interpreter version (3.14) is newer than PyO3's maximum supported version (3.13)`

**Solution:** Downgrade to Python 3.13:
```bash
conda install python=3.13
```

QuantNado requires Python 3.12-3.13 due to PyO3 compatibility. Python 3.14 support coming soon.

### Permission Denied When Creating Dataset

**Problem:** `PermissionError: /path/to/dataset.zarr`

**Solution:** 
1. Check write permissions in output directory:
```bash
ls -ld `dirname dataset.zarr`  # Parent directory
```

2. Fix permissions if needed:
```bash
chmod u+w /path/to/output/
```

3. Or write to a different location (e.g., temp directory):
```bash
quantnado create-dataset *.bam --output /tmp/dataset.zarr
```

## BAM File Issues

### Error: "No such file or directory" for BAM

**Problem:** BAM file not found or glob pattern doesn't match

**Solution:**
1. Check files exist:
```bash
ls -la sample*.bam
```

2. Verify correct directory:
```bash
pwd
cd /path/to/bam/files/
quantnado create-dataset *.bam --output dataset.zarr
```

### Error: "Missing index file"

**Problem:** BAM file lacks `.bai` index

**Solution:**
```bash
# Create index
samtools index sample.bam

# Verify index exists
ls -la sample.bam*
```

## Dataset Creation Issues

### Very Slow Dataset Creation

**Problem:** Processing takes hours for moderate-sized files

**Solution:** Enable parallelization with `--max-workers`:
```bash
quantnado create-dataset *.bam \
  --output dataset.zarr \
  --chromsizes hg38.chrom.sizes \
  --max-workers 8  # Adjust based on CPU cores
```

Check available cores:
```bash
nproc  # Linux/Mac
```

### "Invalid chromsizes format" Error

**Problem:** Chromsizes file not recognized

**Solution:** Verify format (tab-separated, 2 columns):
```bash
head hg38.chrom.sizes
# chr1	248956422
# chr2	242193529
# ...
```

Should be:
- Column 1: Chromosome name (chr1, chr2, etc.)
- Column 2: Chromosome length (integer)
- Tab-separated (no spaces)

Download correct file:
```bash
# For hg38
curl -O https://hgdownload.cse.ucsc.edu/goldenPath/hg38/bigZips/hg38.chrom.sizes

# For other genomes, check UCSC Genome Browser
```

## Peak Calling Issues

### No Peaks Called

**Problem:** Peak calling produces empty output

**Solutions:**
1. **Reduce quantile threshold:**
```bash
quantnado call-peaks \
  --bigwig-dir ./signals/ \
  --output-dir ./peaks/ \
  --quantile 0.95  # Less stringent
```

2. **Check bigWig signal:**
```bash
# Verify bigWig files exist and have content
ls -lh *.bw
bigWigInfo sample.bw  # If available
```

3. **Disable blacklist temporarily:**
```bash
# Run without blacklist to test
quantnado call-peaks \
  --bigwig-dir ./signals/ \
  --output-dir ./peaks/
```

4. **Check for all-zero regions:**
```bash
# Examine raw signal
python3 << EOF
import pyBigWig
bw = pyBigWig.open("sample.bw")
print(f"Chromosomes: {bw.chroms()}")
bw.close()
EOF
```

## Python API Issues

### Cannot Import QuantNado

**Problem:** `ImportError: No module named 'quantnado'`

**Solution:**
```bash
# Verify installation
python -c "import quantnado; print(quantnado.__version__)"

# If not found, install
pip install quantnado

# For development:
pip install -e .  # From source directory
```

### Dataset Open Fails

**Problem:** `Error: Cannot open Zarr store`

**Solution:**
1. Check dataset exists:
```bash
ls -la my_dataset.zarr
```

2. Verify it's a valid Zarr store:
```bash
python3 << EOF
import zarr
root = zarr.open("my_dataset.zarr", "r")
print(f"Store groups: {list(root.keys())}")
EOF
```

### Memory Issues With Large Datasets

**Problem:** Out of memory when accessing data

**Solution:** Load data in chunks:
```python
from quantnado import QuantNado
qn = QuantNado.open("large_dataset.zarr")

# Process chromosome by chromosome
for chrom in qn.sample_names:
    data = qn[chrom]  # Lazy load
    result = data.compute()  # Compute only what you need
    del result  # Free memory
```

## HPC/Cluster Issues

### SLURM Job Fails With "Module Not Found"

**Problem:** Script runs locally but fails on cluster

**Solution:** Activate conda environment in job script:
```bash
#!/bin/bash
#SBATCH --job-name=quantnado
#SBATCH --cpus-per-task=16

# Activate environment
source /path/to/conda/etc/profile.d/conda.sh
conda activate quantnado

# Run command
quantnado create-dataset *.bam --output dataset.zarr
```

### Out of Memory in SLURM Job

**Problem:** Job killed with `MemoryLimit exceeded`

**Solution:** Request more memory:
```bash
#SBATCH --mem=64G  # Increase memory allocation
```

Or reduce parallelization:
```bash
quantnado create-dataset *.bam \
  --output dataset.zarr \
  --max-workers 4  # Fewer threads = less memory
```

## Getting Help

If issues persist:
1. Check [Basic Usage](basic_usage.md) for examples
2. Review [CLI Reference](cli.md) for all options
3. Open an issue on [GitHub](https://github.com/Milne-Group/QuantNado/issues)
4. Include relevant error messages and command used

**Problem**: Dependency conflict between anndata and zarr.

**Root Cause**: anndata doesn't support zarr 3.x but QuantNado requires it.

**Solution**:
```bash
# Update to compatible versions
pip install --upgrade "zarr>=3.0.0" "anndata>=0.10"
```

Or reinstall QuantNado fresh:
```bash
pip uninstall quantnado && pip install quantnado
```

### ImportError: cannot import name 'BaseCompressedSparseDataset'

**Problem**: Incompatible anndata version.

**Solution**:
```bash
pip install "anndata>=0.10.0"
```

### Module 'pkg_resources' has been deprecated

**Problem**: Warning about deprecated pkg_resources (non-fatal).

**Solution**: This is a warning only and doesn't affect functionality. Future versions of dependencies will fix this automatically.

## Dataset Creation Issues

### KeyError: "not all values found in index 'sample'"

**Problem**: Trying to select samples that don't exist in dataset.

**Example Error**:
```python
X = chrom['chr1'].sel(sample=['CAT-RCH-ACV_H3K27ac']).compute()
# KeyError: "not all values found in index 'sample'"
```

**Solution**:

1. Check available samples:
```python
qn = QuantNado.open("dataset.zarr")
print(qn.samples)
```

2. Sample names are BAM file stems (without extension):
```
# If BAM file is: CAT_patient_1_H3K27ac.bam
# Sample name is: CAT_patient_1_H3K27ac

# Rename BAM files if needed before processing
```

3. Select correct sample:
```python
X = chrom['chr1'].sel(sample=['CAT_patient_1_H3K27ac']).compute()
```

### FileNotFoundError: BAM file not found

**Problem**: Specified BAM file path doesn't exist.

**Solution**:
```python
from pathlib import Path

# Verify files exist
bam_files = list(Path("/path/to/bams").glob("*.bam"))
print(f"Found {len(bam_files)} BAM files")
print(bam_files)

# Check for indexed files
for bam in bam_files:
    bai = Path(str(bam) + ".bai")
    if not bai.exists():
        print(f"Missing index for {bam}")
```

### ValueError: chromsizes_dict appears empty

**Problem**: No valid chromosomes extracted from BAM file.

**Causes**:
- BAM file is corrupted
- BAM file header is incomplete
- Invalid chromsizes file

**Solutions**:
1. Verify BAM file integrity:
```bash
samtools view -H file.bam
samtools index file.bam
```

2. Provide explicit chromsizes file:
```python
qn = QuantNado.from_bam_files(
    bam_files=[...],
    store_path="dataset.zarr",
    chromsizes="hg38.chrom.sizes"
)
```

### ValueError: sample_names must not be empty

**Problem**: No BAM files provided.

**Solution**:
```python
# Check bam_files list
bam_files = [...]
print(f"BAM files: {bam_files}")

# Verify it's not empty
assert len(bam_files) > 0
```

### Dataset creation is slow

**Problem**: Processing takes longer than expected.

**Typical durations**:
- 10M reads: 1-2 minutes per sample
- 50M reads: 5-10 minutes per sample  
- 100M+ reads: 15-30+ minutes per sample

**Optimization**:
```bash
# Use fewer threads for faster startup (less memory)
quantnado create-dataset *.bam --output dataset.zarr --max-workers 1

# Use more threads for larger datasets (more memory, in-parallel)
quantnado create-dataset *.bam --output dataset.zarr --max-workers 8
```

### MemoryError during dataset creation

**Problem**: Running out of RAM during processing.

**Solutions**:

1. Use fewer workers:
```bash
quantnado create-dataset *.bam --output dataset.zarr --max-workers 1
```

2. Process in batches:
```bash
# Create dataset for subset of files
quantnado create-dataset file1.bam file2.bam --output dataset.zarr

# Resume with more files
quantnado create-dataset file1.bam file2.bam file3.bam \
  --output dataset.zarr --resume
```

3. Increase available memory (system dependent)

## Analysis Issues

### ValueError: not all values found in index

**Problem**: Selecting ranges that don't exist.

**Solution**:
```python
# Check the actual data
signal = qn.reduce(intervals_path="regions.bed", reduction="mean")
print(signal.dims)
print(signal.coords)

# Use valid sample names
valid_samples = signal.sample.values
signal_subset = signal.sel(sample=valid_samples[:2])
```

### ValueError: fixed_width is not divisible by bin_size

**Problem**: Incompatible parameters.

```python
qn.extract(
    feature_type="promoter",
    gtf_path="genes.gtf",
    fixed_width=2000,
    bin_size=50,  # 2000 / 50 = 40 ✓ OK
    bin_agg="mean"
)
```

**Solution**: Use divisible values:
```python
# 2000 is divisible by: 50, 100, 200, 400, 500, 1000
# 2000 is NOT divisible by: 30, 60, 75, 150
```

### Shape mismatch errors

**Problem**: Dimension mismatch in operations.

**Debug**:
```python
signal = qn.reduce(intervals_path="regions.bed")
print("Signal shape:", signal["mean"].shape)  # (ranges, sample)

# PCA expects (feature, sample)
pca, transformed = qn.pca(signal["mean"], n_components=10)

# extract expects (regions, position, sample)
extracted = qn.extract(intervals_path="regions.bed")
print("Extracted shape:", extracted.shape)
```

## Data Issues

### NaN values in output

**Problem**: Reduced signal contains NaN values.

**Causes**:
- Regions outside chromosome boundaries
- Missing data in original BAM
- Integer overflow

**Solutions**:
```python
signal = qn.reduce(intervals_path="regions.bed")

# Check for NaN
print(f"NaN count: {np.isnan(signal['mean']).sum().values}")

# Remove NaN-containing regions
signal_clean = signal.dropna('ranges', how='any')

# Or use for PCA with explicit NaN handling
pca, transformed = qn.pca(
    signal["mean"],
    nan_handling_strategy="drop"
)
```

### Zero signal everywhere

**Problem**: All signal values are zero.

**Likely causes**:
- BAM file has no reads
- Chromosome names don't match
- Region definition is incorrect

**Debug**:
```bash
# Check BAM file
samtools flagstat file.bam
samtools view -c file.bam

# Check chromosome names
samtools view -H file.bam | grep SQ
```

### Unexpected sparsity

**Problem**: Dataset is sparser than expected.

**Info**: Check sparsity metrics:
```python
qn = QuantNado.open("dataset.zarr")
print(qn.store.meta["sparsity"][:])
```

This is normal for sparse genomic data (usually 90%+ sparse).

## File & Storage Issues

### Zarr store is corrupted

**Problem**: Can't read Zarr dataset.

**Symptoms**:
- Permission errors
- "zarr.errors.ArrayNotFoundError"
- Missing data

**Solutions**:
1. Verify store integrity:
```bash
zarr list -v dataset.zarr
```

2. Recreate from BAM files if store is damaged
3. Check disk space and permissions:
```bash
ls -lh dataset.zarr
df -h  # Check available space
```

### Permission denied errors

**Problem**: Cannot access Zarr store or BAM files.

**Solution**:
```bash
# Check permissions
ls -l file.bam
chmod 644 file.bam

# Or check directory permissions
chmod 755 /path/to/dataset.zarr
```

### Out of disk space

**Problem**: Zarr creation fails due to insufficient space.

**Solution**:
1. Free up space
2. Calculate required space:
```python
# Approximate: Zarr size ≈ largest BAM file × number of chromosomes / read density
```
3. Use remote storage if available

## Performance Issues

### Dask operations are slow

**Problem**: Extract or reduce operations take too long.

**Solutions**:
1. Reduce dataset size:
```python
# Use fewer chromosomes
signal = qn.reduce(
    intervals_path="regions_chr1_only.bed",
    reduction="mean"
)

# Use fewer samples
signal = signal.sel(sample=['sample1', 'sample2'])
```

2. Increase chunk size (trade-off with memory):
```bash
quantnado create-dataset *.bam \
  --output dataset.zarr \
  --max-workers 4
```

3. Compute in batches rather than all at once

### High memory usage

**Problem**: Dask operations consume too much memory.

**Solutions**:
1. Set Dask memory limit:
```python
from dask.distributed import Client
client = Client(memory_limit='2GB')
```

2. Use smaller chunks
3. Process regions in batches

## Metadata Issues

### Metadata not being read

**Problem**: Metadata is empty or not loaded.

**Debug**:
```python
qn = QuantNado.open("dataset.zarr")
print(qn.metadata)  # Check if empty

# Check if metadata was set
print(qn.store.list_metadata_columns())
```

**Solution**: Ensure metadata was provided during creation:
```python
qn = QuantNado.from_bam_files(
    bam_files=[...],
    store_path="dataset.zarr",
    metadata="samples.csv",  # Provide CSV path
    sample_column="sample_id"
)
```

### Metadata column name mismatch

**Problem**: Sample column not found.

**Solution**:
```python
import pandas as pd

# Check CSV column names
metadata = pd.read_csv("samples.csv")
print(metadata.columns)

# Use correct column name
qn = QuantNado.from_bam_files(
    bam_files=[...],
    store_path="dataset.zarr",
    metadata="samples.csv",
    sample_column="sample_id"  # Must match CSV column
)
```

## Getting Help

If issues persist:

1. **Check Documentation**: [Full docs](index.md)
2. **Search GitHub Issues**: [QuantNado Issues](https://github.com/Milne-Group/QuantNado/issues)
3. **Enable Debug Logging**:
```bash
quantnado create-dataset *.bam --output dataset.zarr --verbose --log-file debug.log
```
4. **Report Issue** with:
   - Error message and traceback
   - Python version: `python --version`
   - QuantNado version: `python -c "import quantnado; print(quantnado.__version__)"`
   - Operating system
   - Minimal reproducible example

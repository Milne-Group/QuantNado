# Examples and Workflows

For a complete, runnable end-to-end walkthrough using a real multiomics dataset, see the
**[Example Notebook](example_dataset.ipynb)** — it covers dataset creation, signal reduction,
PCA, feature counts, and coverage extraction over a chr21 test dataset.

You can also open or download the notebook directly from GitHub:
[example/example_dataset.ipynb](https://github.com/Milne-Group/QuantNado/blob/main/example/example_dataset.ipynb)

---

The code snippets below show focused workflows for common QuantNado tasks.

## Workflow 1: ChIP-seq Analysis

Process ChIP-seq BAM files, quantify signal over promoters, and identify enriched regions.

```python
from quantnado import QuantNado
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Create dataset from BAM files
qn = QuantNado.from_bam_files(
    bam_files=[
        "chip_rep1.bam",
        "chip_rep2.bam",
        "input_rep1.bam",
        "input_rep2.bam"
    ],
    store_path="chip_dataset.zarr",
    metadata=pd.DataFrame({
        "sample_id": ["chip_rep1", "chip_rep2", "input_rep1", "input_rep2"],
        "type": ["chip", "chip", "input", "input"],
        "replicate": [1, 2, 1, 2]
    }),
    sample_column="sample_id"
)

# Step 2: Extract signal over promoters
promoter_signal = qn.reduce(
    feature_type="promoter",
    gtf_path="genes.gtf",
    reduction="mean"
)

# Step 3: Calculate ChIP/Input ratio
chip_signal = promoter_signal["mean"].sel(sample=["chip_rep1", "chip_rep2"]).mean("sample")
input_signal = promoter_signal["mean"].sel(sample=["input_rep1", "input_rep2"]).mean("sample")
enrichment = chip_signal / (input_signal + 1)  # Avoid division by zero

# Step 4: Identify enriched promoters
top_enriched = enrichment.argsort()[-100:]
print(f"Top 100 enriched promoters: {top_enriched.values}")

# Step 5: Visualize enrichment
plt.figure(figsize=(10, 6))
plt.hist(enrichment.values, bins=50, edgecolor='black')
plt.xlabel("Enrichment (ChIP/Input)")
plt.ylabel("Number of Promoters")
plt.title("ChIP-seq Signal Enrichment Distribution")
plt.savefig("enrichment_distribution.png", dpi=300, bbox_inches='tight')
plt.show()
```

## Workflow 2: Multi-Sample Comparison with PCA

Compare multiple samples using dimensionality reduction.

```python
from quantnado import QuantNado
import matplotlib.pyplot as plt

# Load dataset
qn = QuantNado.open("chip_dataset.zarr")

# Extract signal over enhancers
enhancer_signal = qn.reduce(
    intervals_path="enhancers.bed",
    reduction="mean"
)

# Run PCA
pca, transformed = qn.pca(
    enhancer_signal["mean"],
    n_components=3,
    nan_handling_strategy="drop"
)

# Visualize
import numpy as np
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Color by sample type
colors = {'chip': 'red', 'input': 'blue'}
metadata = qn.metadata
transformed_arr = np.asarray(transformed)  # (n_samples, n_components)
for idx, sample in enumerate(qn.samples):
    sample_type = metadata.loc[sample, 'type']
    color = colors[sample_type]
    ax.scatter(
        transformed_arr[idx, 0],
        transformed_arr[idx, 1],
        transformed_arr[idx, 2],
        c=color,
        s=100,
        label=sample_type if sample_type not in ax.get_legend_handles_labels()[1] else ""
    )

ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]:.1%})")
plt.legend()
plt.savefig("pca_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

# Variance explained
plt.figure(figsize=(8, 6))
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1),
        np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Variance Explained")
plt.title("PCA Variance Explained")
plt.savefig("variance_explained.png", dpi=300, bbox_inches='tight')
plt.show()
```

## Workflow 3: Feature Counting for DESeq2

Generate count matrices for differential expression analysis.

```python
from quantnado import QuantNado
import pandas as pd

# Load dataset
qn = QuantNado.open("chip_dataset.zarr")

# Generate gene counts
counts, features = qn.count_features(
    gtf_file="genes.gtf",
    feature_type="gene",
    integerize=True
)

# Create metadata for DESeq2
metadata = pd.DataFrame({
    "sample": counts.columns,
    "condition": ["chip", "chip", "input", "input"],
    "replicate": [1, 2, 1, 2]
})

# Filter low-count genes
min_counts = counts.sum(axis=1) >= 10
counts_filtered = counts[min_counts]
features_filtered = features[min_counts]

print(f"Genes before filter: {len(counts)}")
print(f"Genes after filter: {len(counts_filtered)}")

# Export for DESeq2
counts_filtered.to_csv("counts.csv")
features_filtered.to_csv("features.csv")
metadata.to_csv("metadata.csv", index=False)

# Optionally, create R script for DESeq2
r_script = """
library(DESeq2)

counts <- read.csv("counts.csv", row.names=1)
metadata <- read.csv("metadata.csv", row.names=1)
features <- read.csv("features.csv", row.names=1)

# Create DESeqDataSet
dds <- DESeqDataSetFromMatrix(
    countData=counts,
    colData=metadata,
    design=~condition
)

# Run DESeq2
dds <- DESeq(dds)
res <- results(dds)

# Export results
write.csv(res, "deseq_results.csv")
"""

with open("run_deseq2.R", "w") as f:
    f.write(r_script)

print("Run: Rscript run_deseq2.R")
```

## Workflow 4: Signal Heatmap over Genomic Features

Create heatmaps of signal over peaks or other regions.

```python
from quantnado import QuantNado
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
qn = QuantNado.open("chip_dataset.zarr")

# Extract signal with fixed width
peak_signal = qn.extract(
    intervals_path="peaks.bed",
    fixed_width=2000,
    bin_size=50,
    bin_agg="mean"
)

# Compute for visualization
data = peak_signal.compute()
data_array = data.values

# Sort peaks by max signal
peak_order = np.argsort(data_array[:, :, 0].max(axis=1))[::-1]  # Descending

# Create heatmap
fig, axes = plt.subplots(1, len(qn.samples), figsize=(15, 8))

for i, sample in enumerate(qn.samples):
    ax = axes[i] if len(qn.samples) > 1 else axes
    im = ax.imshow(
        data_array[peak_order, :, i],
        aspect='auto',
        cmap='RdYlBu_r',
        origin='lower'
    )
    ax.set_title(sample)
    ax.set_xlabel("Position (bp)")
    ax.set_ylabel("Peak")
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig("peak_heatmap.png", dpi=300, bbox_inches='tight')
plt.show()
```

## Workflow 5: Comparative Analysis Across Conditions

Compare enrichment patterns between experimental conditions.

```python
from quantnado import QuantNado
import pandas as pd
import numpy as np

# Load dataset with condition metadata
qn = QuantNado.open("chip_dataset.zarr")

# Get metadata
metadata = qn.metadata

# Extract signal over genomic features
signal = qn.reduce(
    feature_type="promoter",
    gtf_path="genes.gtf",
    reduction="mean"
)

# Group samples by condition
conditions = metadata['condition'].unique()
condition_signals = {}

for condition in conditions:
    samples = metadata[metadata['condition'] == condition].index.tolist()
    condition_signals[condition] = signal["mean"].sel(sample=samples).mean("sample")

# Identify features with differential signal
avg_chip = condition_signals['chip']
avg_input = condition_signals['input']
log2_fc = np.log2((avg_chip + 1) / (avg_input + 1))

# Find significantly enriched features
enriched = np.where(log2_fc > 2)[0]
depleted = np.where(log2_fc < -2)[0]

print(f"Enriched regions: {len(enriched)}")
print(f"Depleted regions: {len(depleted)}")

# Visualize log2FC distribution
plt.figure(figsize=(10, 6))
plt.hist(log2_fc, bins=50, edgecolor='black', alpha=0.7)
plt.axvline(x=2, color='red', linestyle='--', label='log2FC > 2 (enriched)')
plt.axvline(x=-2, color='blue', linestyle='--', label='log2FC < -2 (depleted)')
plt.xlabel("log2(Fold Change)")
plt.ylabel("Number of Features")
plt.title("Distribution of Fold Changes")
plt.legend()
plt.savefig("log2fc_distribution.png", dpi=300, bbox_inches='tight')
plt.show()
```

## Workflow 6: Downstream API Helpers for Visualization

Use the higher-level plotting and extraction helpers exposed on `QuantNado` for
post-processing and exploratory analysis.

```python
from quantnado import QuantNado
import matplotlib.pyplot as plt

qn = QuantNado.open("chip_dataset.zarr")

# 1) Load selected chromosomes as lazy xarray objects
signal_by_chrom = qn.to_xarray(chromosomes=["chr1", "chr2"])
print(signal_by_chrom["chr1"])  # DataArray: (sample, position)

# 2) Extract a specific locus for browser-like inspection
region = qn.extract_region("chr1:100000-102000")
print(region.shape)  # (n_samples, 2000)

# 3) Build binned windows around TSS for profile/heatmap plots
binned = qn.extract(
    feature_type="transcript",
    gtf_path="genes.gtf",
    upstream=1000,
    downstream=1000,
    anchor="start",
    bin_size=50,
)

# Optional grouping from metadata
metadata = qn.metadata.copy()
metadata["assay"] = metadata.index.to_series().str.split("_").str[0]
groups = metadata.groupby("assay").groups

# 4) Metaplot helper
ax = qn.metaplot(
    binned,
    modality="coverage",
    groups=groups,
    flip_minus_strand=True,
    reference_point=0,
    reference_label="TSS",
    xlabel="Distance from TSS (bp)",
    title="Coverage metaplot",
)
plt.show()

# 5) Tornado helper with display aliases
axes = qn.tornadoplot(
    binned,
    modality="coverage",
    samples=["chip_rep1", "chip_rep2", "input_rep1", "input_rep2"],
    sample_names=["ChIP rep1", "ChIP rep2", "Input rep1", "Input rep2"],
    flip_minus_strand=True,
    sort_by="mean",
    ylabel="Promoters",
    title="Coverage tornado",
)
plt.show()
```

## Tips and Best Practices

### 1. Memory Efficiency

Use Dask's lazy evaluation for large datasets:

```python
# Don't compute immediately - keep lazy
signal = qn.reduce(intervals_path="large_region_set.bed")

# Only compute what you need
top_100 = signal["mean"].nlargest(100, 'ranges')
```

### 2. Sample Naming

Keep sample names informative but use underscores for spaces:

```
# Good names (used as BAM stems):
chip_h3k4me3_rep1.bam
chip_h3k4me3_rep2.bam
input_rep1.bam

# Result in sample names:
# ['chip_h3k4me3_rep1', 'chip_h3k4me3_rep2', 'input_rep1']
```

### 3. Parallel Processing

Use the Python API to control worker count when building datasets:

```python
from quantnado import MultiomicsStore

bam_files = ["sample1.bam", "sample2.bam", "sample3.bam"]

# Single-core (least memory)
ms = MultiomicsStore.from_files(
    store_dir="dataset",
    bam_files=bam_files,
    max_workers=1,
)

# Multi-core (more memory, faster)
ms = MultiomicsStore.from_files(
    store_dir="dataset",
    bam_files=bam_files,
    max_workers=8,
)
```

If you prefer the convenience wrapper for BAM-only workflows, use
`QuantNado.from_bam_files(...)`.

### 4. Metadata Organization

Structure metadata for easy filtering and grouping:

```python
metadata = pd.DataFrame({
    "sample_id": [...],
    "condition": [...],
    "replicate": [...],
    "batch": [...],
    "quality_score": [...]
})
```

## See Also

- [Quick Start](quick_start.md)
- [Basic Usage](basic_usage.md)
- [API Reference](api/quantnado.md)
- [Troubleshooting](troubleshooting.md)

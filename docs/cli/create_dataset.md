# `create-dataset`

Create one QuantNado `.zarr` store per sample from a metadata CSV or TSV.

## Usage

```bash
quantnado create-dataset \
  --metadata samples.csv \
  --output-dir dataset
```

## Required Options

- `--metadata`, `-m`: metadata CSV/TSV
- `--output-dir`, `-o`: directory for per-sample stores

## Metadata Format

Required columns:

- `sample`
- `assay`

Optional columns:

- `bam`
- `methyl`
- `vcf`
- `ip`
- `chromsizes`

Supported assay values:

- `ATAC`
- `ChIP`
- `RNA`
- `CUT&TAG`
- `METH`
- `SNP`
- `MCC`

Example:

```csv
sample,assay,bam,methyl,vcf,ip
ATAC_1,ATAC,/data/ATAC_1.bam,,,
H3K27ac_1,ChIP,/data/H3K27ac_1.bam,,,H3K27ac
RNA_1,RNA,/data/RNA_1.bam,,,
METH_1,METH,/data/METH_1.bam,/data/METH_1.bedGraph,,
SNP_1,SNP,,,/data/SNP_1.vcf.gz,
```

## Options

- `--chromsizes PATH`: fallback `.chrom.sizes` file for rows that do not provide one
- `--filter-chromosomes / --no-filter-chromosomes`: keep only canonical chromosomes
- `--overwrite / --no-overwrite`: replace existing stores
- `--chunk-len INTEGER`: override position-axis chunk length
- `--construction-compression TEXT`: one of `default`, `fast`, or `none`
- `--test`: use the default test chromosomes (`chr9`, `chr13`, `chr21`)
- `--test-chrom TEXT`: chromosome to keep in test mode; repeat to pass multiple chromosomes
- `--log-file PATH`: log destination
- `--verbose`, `-v`: debug logging

## Examples

Basic run:

```bash
quantnado create-dataset \
  --metadata samples.csv \
  --output-dir dataset
```

Use explicit chromsizes and overwrite existing stores:

```bash
quantnado create-dataset \
  --metadata samples.csv \
  --output-dir dataset \
  --chromsizes hg38.chrom.sizes \
  --overwrite
```

Use a different build-time compression profile:

```bash
quantnado create-dataset \
  --metadata samples.csv \
  --output-dir dataset \
  --construction-compression fast
```

Use an explicit test chromosome list:

```bash
quantnado create-dataset \
  --metadata samples.csv \
  --output-dir dataset \
  --test-chrom chr21 \
  --test-chrom chr9
```

## Output

For a metadata row with `sample=ATAC_1`, QuantNado writes:

```text
dataset/ATAC_1.zarr
```

Only completed stores are later included when you open a directory or combine stores.

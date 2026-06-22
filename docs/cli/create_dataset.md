# `dataset create`

Create one QuantNado `.zarr` store per sample from direct assay inputs.

## Usage

```bash
quantnado dataset create \
  --sample RNA_1 \
  --assay RNA \
  --bamfile /data/RNA_1.bam \
  --stranded R \
  --output-dir dataset
```

## Required Options

- `--sample`: sample name
- `--assay`: assay type
- `--output-dir`, `-o`: directory for per-sample stores

## Input Options by Assay

- BAM-based assays (`ATAC`, `ChIP`, `RNA`, `CUT&TAG`, `MCC`) use `--bamfile`
- `METH` uses `--bamfile` and `--methylation_file`
- `SNP` uses `--vcf_file`
- `ChIP` and `CUT&TAG` can also use `--ip`
- `RNA` can also use `--stranded`
- Single-end BAMs should use `--single-end`

## Supported Assays

- `ATAC`
- `ChIP`
- `RNA`
- `CUT&TAG`
- `METH`
- `SNP`
- `MCC`

## Options

- `--bamfile`: BAM file for BAM-based assays and `METH`
- `--vcf_file`: VCF file for `SNP`
- `--methylation_file`: methylation bedGraph/TSV for `METH`
- `--ip`: target label for `ChIP` / `CUT&TAG`
- `--stranded`: RNA strandedness (`R`, `F`, `1`, `2`, `U`)
- `--paired / --single-end`: BAM read layout; defaults to paired-end
- `--chromsizes PATH`: fallback `.chrom.sizes` file
- `--filter-chromosomes / --no-filter-chromosomes`: keep only canonical chromosomes
- `--overwrite / --no-overwrite`: replace an existing store
- `--chunk-len INTEGER`: override position-axis chunk length
- `--construction-compression TEXT`: one of `default`, `fast`, or `none`
- `--test`: use the default test chromosomes (`chr9`, `chr13`, `chr21`)
- `--test-chrom TEXT`: chromosome to keep in test mode; repeat to pass multiple chromosomes
- `--log-file PATH`: log destination
- `--verbose`, `-v`: debug logging

## Examples

ATAC:

```bash
quantnado dataset create \
  --sample ATAC_1 \
  --assay ATAC \
  --bamfile /data/ATAC_1.bam \
  --output-dir dataset
```

Single-end RNA:

```bash
quantnado dataset create \
  --sample RNA_SE_1 \
  --assay RNA \
  --bamfile /data/RNA_SE_1.bam \
  --stranded R \
  --single-end \
  --output-dir dataset
```

ChIP:

```bash
quantnado dataset create \
  --sample H3K27ac_1 \
  --assay ChIP \
  --bamfile /data/H3K27ac_1.bam \
  --ip H3K27ac \
  --output-dir dataset
```

METH:

```bash
quantnado dataset create \
  --sample METH_1 \
  --assay METH \
  --bamfile /data/METH_1.bam \
  --methylation_file /data/METH_1.bedGraph \
  --output-dir dataset
```

SNP:

```bash
quantnado dataset create \
  --sample SNP_1 \
  --assay SNP \
  --vcf_file /data/SNP_1.vcf.gz \
  --output-dir dataset
```

## Output

For `--sample ATAC_1`, QuantNado writes:

```text
dataset/ATAC_1.zarr
```

Only completed stores are later included when you open a directory or combine stores.

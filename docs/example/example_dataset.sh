#!/usr/bin/env bash
set -euo pipefail

# alternative example dataset creation script for cli usage demonstration
# Run this script from the repository root:
#   bash example/example_dataset.sh

if [[ ! -f "pyproject.toml" ]]; then
  echo "Run this script from the repository root." >&2
  exit 1
fi

RUN_DIR="docs/example/multiomics_run"
DATA_DIR="${RUN_DIR}/seqnado_output"
REF_DIR="${RUN_DIR}/reference"
OUT_DIR="${RUN_DIR}/output"
SAMPLES_DIR="${OUT_DIR}/samples"
LOG_DIR="${OUT_DIR}/logs"
DATASET="${OUT_DIR}/dataset.zarr"


mkdir -p "${OUT_DIR}" "${SAMPLES_DIR}" "${LOG_DIR}" "${DATA_DIR}" "${REF_DIR}"

DATA_URL="https://userweb.molbiol.ox.ac.uk/public/project/milne_group/cchahrou/quantnado/seqnado_output.tar.gz"
REF_URL="https://userweb.molbiol.ox.ac.uk/public/project/milne_group/cchahrou/quantnado/reference.tar.gz"

wget "${DATA_URL}" -O "${RUN_DIR}/seqnado_output.tar.gz"
wget "${REF_URL}" -O "${RUN_DIR}/reference.tar.gz"

tar -xzf "${RUN_DIR}/seqnado_output.tar.gz" -C "${RUN_DIR}"
tar -xzf "${RUN_DIR}/reference.tar.gz" -C "${RUN_DIR}"

create_store() {
  quantnado dataset create "$@" --output-dir "${SAMPLES_DIR}" --overwrite
}

create_store \
  --sample atac \
  --assay ATAC \
  --bamfile "${DATA_DIR}/atac/aligned/atac.bam" \
  --log-file "${LOG_DIR}/store_creation_atac.log"

create_store \
  --sample chip-rx_MLL \
  --assay ChIP \
  --bamfile "${DATA_DIR}/chip/aligned/chip-rx_MLL.bam" \
  --ip MLLN \
  --log-file "${LOG_DIR}/store_creation_chip.log"

create_store \
  --sample meth-rep1 \
  --assay METH \
  --bamfile "${DATA_DIR}/meth/aligned/meth-rep1.bam" \
  --methylation_file "${DATA_DIR}/meth/methylation/methyldackel/meth-rep1_hg38_CpG_inverted.bedGraph" \
  --log-file "${LOG_DIR}/store_creation_meth_rep1.log"


create_store \
  --sample meth-rep2 \
  --assay METH \
  --bamfile "${DATA_DIR}/meth/aligned/meth-rep2.bam" \
  --methylation_file "${DATA_DIR}/meth/methylation/methyldackel/meth-rep2_hg38_CpG_inverted.bedGraph" \
  --log-file "${LOG_DIR}/store_creation_meth_rep2.log"

create_store \
  --sample snp \
  --assay SNP \
  --vcf_file "${DATA_DIR}/snp/variant/snp.vcf.gz" \
  --log-file "${LOG_DIR}/store_creation_snp.log"

create_store \
  --sample rna-spikein-control-rep1 \
  --assay RNA \
  --bamfile "${DATA_DIR}/rna/aligned/rna-spikein-control-rep1.bam" \
  --stranded R \
  --log-file "${LOG_DIR}/store_creation_rna_spikein_control_rep1.log"

create_store \
  --sample rna-spikein-control-rep2 \
  --assay RNA \
  --bamfile "${DATA_DIR}/rna/aligned/rna-spikein-control-rep2.bam" \
  --stranded R \
  --log-file "${LOG_DIR}/store_creation_rna_spikein_control_rep2.log"

create_store \
  --sample rna-spikein-treated-rep1 \
  --assay RNA \
  --bamfile "${DATA_DIR}/rna/aligned/rna-spikein-treated-rep1.bam" \
  --stranded R \
  --log-file "${LOG_DIR}/store_creation_rna_spikein_treated_rep1.log"

create_store \
  --sample rna-spikein-treated-rep2 \
  --assay RNA \
  --bamfile "${DATA_DIR}/rna/aligned/rna-spikein-treated-rep2.bam" \
  --stranded R \
  --log-file "${LOG_DIR}/store_creation_rna_spikein_treated_rep2.log"

quantnado dataset combine \
  --stores \
    "${SAMPLES_DIR}/atac.zarr" \
    "${SAMPLES_DIR}/chip-rx_MLL.zarr" \
    "${SAMPLES_DIR}/meth-rep1.zarr" \
    "${SAMPLES_DIR}/meth-rep2.zarr" \
    "${SAMPLES_DIR}/snp.zarr" \
    "${SAMPLES_DIR}/rna-spikein-control-rep1.zarr" \
    "${SAMPLES_DIR}/rna-spikein-control-rep2.zarr" \
    "${SAMPLES_DIR}/rna-spikein-treated-rep1.zarr" \
    "${SAMPLES_DIR}/rna-spikein-treated-rep2.zarr" \
  --output "${DATASET}" \
  --overwrite \
  --log-file "${LOG_DIR}/dataset_combination.log"

echo
echo "Example dataset ready:"
echo "  sample stores: ${SAMPLES_DIR}"
echo "  combined zarr: ${DATASET}"

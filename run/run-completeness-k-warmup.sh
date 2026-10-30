#!/usr/bin/env bash
# Run the k-warmup study (where the unknown-unknowns estimator stabilizes per ordering)
# across datasets.
#
# The DATASETS list below is the single place to toggle what runs: comment lines out
# to run only a subset. Each entry is "<dataset-name>|<dataset-dir>". Runs use the
# default config; each invocation writes a timestamped run dir.
#
# Usage: run-completeness-k-warmup.sh [<dataset-name>/<dataset-dir> ...]   (no argument = all)
set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELECTED=("$@")
source "${RUN_DIR}/setup.sh"

DATASETS=(
  "wikidata-federated-overlap-dataset|dataset"
  "wikidata-federated-overlap-dataset|dataset-50"
  # web-data-commons-product|dataset  has no _meta/input_tables, so k-warmup has nothing to run
  # "crowdwork-query-estimation|dataset_vldbsigmod"
  # "crowdwork-query-estimation|dataset_siliconvalley_employee"
  # "crowdwork-query-estimation|dataset_GDP2025"
  # "crowdwork-query-estimation|dataset_siliconvalley_revenue"
)

RUN_NAME="evaluate-k-warmup"

for entry in "${DATASETS[@]}"; do
  IFS="|" read -r DATASET_NAME DATASET_DIR <<< "${entry}"
  if (( ${#SELECTED[@]} > 0 )) && [[ ! " ${SELECTED[*]} " == *" ${DATASET_NAME}/${DATASET_DIR} "* ]]; then
    continue
  fi
  LOG_DIR="${BASE_LOG_DIR}/${DATASET_NAME}/${DATASET_DIR}/${RUN_NAME}-$(date +%Y-%m-%d-%H-%M-%S)"
  mkdir -p "${LOG_DIR}"
  echo "================== '${RUN_NAME}' with dataset '${DATASET_NAME}/${DATASET_DIR}' -> ${LOG_DIR}"
  python3 -m scripts.evaluation k-warmup \
    --dataset_dir "${BASE_DATASET_DIR}/${DATASET_NAME}/${DATASET_DIR}" \
    --output_dir "${LOG_DIR}" \
    --cache_dir "${CACHE_DIR}" \
    --config "${PROJECT_DIR}/config.yaml" \
    --parallel \
    --max_workers 4
done

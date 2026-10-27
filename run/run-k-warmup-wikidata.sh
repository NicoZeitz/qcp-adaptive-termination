#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${RUN_DIR}/setup.sh"

RUN_NAME="evaluate-k-warmup"
DATASET_NAME="wikidata-federated-overlap-dataset"
DATASET_DIR="dataset"

echo "================== Running '${RUN_NAME}' with dataset '${DATASET_NAME}/${DATASET_DIR}'"
LOG_DIR="${BASE_LOG_DIR}/${DATASET_NAME}/${RUN_NAME}-$(date +%Y-%m-%d-%H-%M)"
mkdir -p "${LOG_DIR}"

python3 -m scripts.evaluation k-warmup \
  --dataset_dir "${BASE_DATASET_DIR}/${DATASET_NAME}/${DATASET_DIR}" \
  --output_dir "${LOG_DIR}" \
  --cache_dir "${CACHE_DIR}" \
  --config "${PROJECT_DIR}/config.yaml" \
  --generate_inputs \
  --workers 3 \
  --parallel \
  --max_workers 6

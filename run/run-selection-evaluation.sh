#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${RUN_DIR}/setup.sh"

RUN_NAME="evaluate-selection-strategies"
DATASET_NAME="wikidata-federated-overlap-dataset"

echo "================== Running '${RUN_NAME}' with dataset '${DATASET_NAME}'"
LOG_DIR="${BASE_LOG_DIR}/${DATASET_NAME}/${RUN_NAME}-$(date +%Y-%m-%d-%H-%M)"
mkdir -p "${LOG_DIR}"

python3 -m scripts.evaluation selection-strategies \
  --data_dir "${BASE_DATASET_DIR}/${DATASET_NAME}" \
  --input_dir "${BASE_DATASET_DIR}/${DATASET_NAME}/dataset/_meta/input" \
  --output_dir "${LOG_DIR}" \
  --cache_dir "${CACHE_DIR}" \
  --config "${PROJECT_DIR}/config.yaml" \
  --parallel

#!/usr/bin/env bash
# Run one completeness-evaluation experiment: one dataset, one agent ordering, one stopping variant.
# Usage: run-completeness-variant.sh <dataset-name> <dataset-dir> <ordering> <stopping> [extra evaluation args...]
#   ordering: random | posterior | count             (run/variants/order-<ordering>.yaml)
#   stopping: none | estimator | llm-prevent | estimator-llm-prevent | estimator-llm-estimator-reasoning
#             (run/variants/stop-<stopping>.yaml)
set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${RUN_DIR}/setup.sh"

DATASET_NAME="$1"
DATASET_DIR="$2"
ORDERING="$3"
STOPPING="$4"
shift 4

RUN_NAME="evaluate-completeness-${ORDERING}-${STOPPING}"
LOG_DIR="${BASE_LOG_DIR}/${DATASET_NAME}/${DATASET_DIR}/${RUN_NAME}-$(date +%Y-%m-%d-%H-%M-%S)"
mkdir -p "${LOG_DIR}"

echo "================== Running '${RUN_NAME}' with dataset '${DATASET_NAME}/${DATASET_DIR}' -> ${LOG_DIR}"

python3 -m scripts.evaluation completeness \
  --dataset_dir "${BASE_DATASET_DIR}/${DATASET_NAME}/${DATASET_DIR}" \
  --output_dir "${LOG_DIR}" \
  --cache_dir "${CACHE_DIR}" \
  --config "${PROJECT_DIR}/config.yaml" \
  --config_override "${RUN_DIR}/variants/order-${ORDERING}.yaml" \
  --config_override "${RUN_DIR}/variants/stop-${STOPPING}.yaml" \
  --parallel \
  --max_workers 6 \
  "$@"

#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${RUN_DIR}/setup.sh"

RUN_NAME="main"
DATASET_NAME="wikidata-federated-overlap-dataset"
QUERY_FILE="What_is_the_average_attendance_of_NBA_matches.json"

echo "================== Running '${RUN_NAME}' with dataset '${DATASET_NAME}'"
INPUT_FILE="${BASE_DATASET_DIR}/${DATASET_NAME}/dataset/_meta/input/${QUERY_FILE}"
LOG_DIR="${BASE_LOG_DIR}/${DATASET_NAME}/${RUN_NAME}-$(date +%Y-%m-%d-%H-%M)/$(basename "${QUERY_FILE}" .json)"
echo "  Input: ${INPUT_FILE}"
mkdir -p "${LOG_DIR}"

python3 -m scripts.main \
  --name "[${RUN_USER:0:2}] ${RUN_NAME} (${DATASET_NAME}): ${QUERY_FILE}" \
  --input "${INPUT_FILE}" \
  --output_dir "${LOG_DIR}" \
  --cache_dir "${CACHE_DIR}" \
  --data_dir "${BASE_DATASET_DIR}/${DATASET_NAME}" \
  --config "${PROJECT_DIR}/config.yaml" \
  --start-stage discovery --end-stage execution

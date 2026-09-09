#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

uv sync
source .venv/bin/activate

USER=$(whoami)
ASSET_USER="nzeitz"

PROJECT_DIR="/mnt/labstore/${USER}/qcp-adaptive-termination"
CACHE_DIR="/mnt/labstore/${ASSET_USER}/.cache"
BASE_DATASET_DIR="/mnt/labstore/${ASSET_USER}/dataset"
BASE_LOG_DIR="/mnt/labstore/${USER}/.logs"

DATASET_NAME="wikidata-federated-overlap-dataset"
DATASET_DIR="${BASE_DATASET_DIR}/${DATASET_NAME}"

# TODO: set input file
INPUT_FILE="${DATASET_DIR}/input/How_many_airlines_are_headquartered_in_UK.json"
OUTPUT_FILE="${BASE_LOG_DIR}/cluster_visualization.png"

mkdir -p "${BASE_LOG_DIR}"

python3 -m scripts.cluster_visualization \
  --input "${INPUT_FILE}" \
  --data-dir "${DATASET_DIR}" \
  --cache_dir "${CACHE_DIR}" \
  --config "${PROJECT_DIR}/config.yaml" \
  --output "${OUTPUT_FILE}"

echo "Saved to ${OUTPUT_FILE}"

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
INPUT_DIR="${DATASET_DIR}/input"

RUN_NAME="embedding_viz_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${BASE_LOG_DIR}/${DATASET_NAME}/${RUN_NAME}"

mkdir -p "${OUT_DIR}"

# Use first input file just to get the agent list — all files share the same 1000 agents
INPUT_FILE="${INPUT_DIR}/How_many_airlines_are_headquartered_in_UK.json"

echo "================== Running '${RUN_NAME}' — output: ${OUT_DIR}"

python3 -m scripts.embedding_visualization \
  --input "${INPUT_FILE}" \
  --output_dir "${OUT_DIR}" \
  --cache_dir "${CACHE_DIR}" \
  --data_dir "${DATASET_DIR}" \
  --config "${PROJECT_DIR}/config.yaml" \
  --start-stage 1 --end-stage 1 \
  > "${OUT_DIR}/stdout.log" 2>&1

echo "================== Done. Results in ${OUT_DIR}"

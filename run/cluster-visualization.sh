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

RUN_NAME="clustering_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${BASE_LOG_DIR}/${DATASET_NAME}/${RUN_NAME}"

mkdir -p "${OUT_DIR}"

echo "================== Running '${RUN_NAME}' — output: ${OUT_DIR}"

shopt -s nullglob
FILES=("${INPUT_DIR}"/*.json)
shopt -u nullglob

PIDS=()
for INPUT_FILE in "${FILES[@]}"; do
    SLUG=$(basename "${INPUT_FILE}" .json)
    OUTPUT_FILE="${OUT_DIR}/${SLUG}.png"
    LOG_FILE="${OUT_DIR}/${SLUG}.log"

    echo "  Queuing: ${SLUG}"

    python3 -m scripts.cluster_visualization \
      --input "${INPUT_FILE}" \
      --data-dir "${DATASET_DIR}" \
      --cache_dir "${CACHE_DIR}" \
      --config "${PROJECT_DIR}/config.yaml" \
      --output "${OUTPUT_FILE}" \
      --log "${LOG_FILE}" \
      > "${OUT_DIR}/${SLUG}.stdout.log" 2>&1 &

    PIDS+=($!)
done

echo "Waiting for ${#PIDS[@]} jobs..."
wait "${PIDS[@]}"
echo "================== Done. Results in ${OUT_DIR}"

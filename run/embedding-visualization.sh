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

# TODO: set registry type: technical | natural_language | generated_nl
REGISTRY_TYPE="generated_nl"

RUN_NAME="embedding_viz_${REGISTRY_TYPE}_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${BASE_LOG_DIR}/${DATASET_NAME}/${RUN_NAME}"

mkdir -p "${OUT_DIR}"

# write a temporary config with the desired registry_type
TMP_CONFIG=$(mktemp /tmp/config_embviz_XXXXXX.yaml)
sed "s/registry_type:.*/registry_type: ${REGISTRY_TYPE}/" "${PROJECT_DIR}/config.yaml" > "${TMP_CONFIG}"

# Use first input file just to get the agent list — all files share the same 1000 agents
INPUT_FILE="${INPUT_DIR}/How_many_airlines_are_headquartered_in_UK.json"

echo "================== Running '${RUN_NAME}' (registry_type=${REGISTRY_TYPE}) — output: ${OUT_DIR}"

python3 -m scripts.embedding_visualization \
  --input "${INPUT_FILE}" \
  --output_dir "${OUT_DIR}" \
  --cache_dir "${CACHE_DIR}" \
  --data_dir "${DATASET_DIR}" \
  --config "${TMP_CONFIG}" \
  --start-stage 1 --end-stage 1 \
  > "${OUT_DIR}/stdout.log" 2>&1

rm -f "${TMP_CONFIG}"
echo "================== Done. Results in ${OUT_DIR}"

#!/usr/bin/env bash
############################################################ GENERAL SETUP ############################################################
set -euo pipefail

# set current working directory to the script's parent directory (project dir)
cd "$(dirname "$0")/.."

# sync project dependencies with uv
uv sync

# activate the virtual environment
source .venv/bin/activate

USER=$(whoami)
ASSET_USER="nzeitz"

PROJECT_DIR="/mnt/labstore/${USER}/qcp-adaptive-termination"
CACHE_DIR="/mnt/labstore/${ASSET_USER}/.cache"

BASE_DATASET_DIR="/mnt/labstore/${ASSET_USER}/dataset"
BASE_LOG_DIR="/mnt/labstore/${USER}/.logs"

############################################################ DATASET SPECIFIC SETUP ############################################################
DATASET_NAME="bird-train-dev"

DATASET_DIR="${BASE_DATASET_DIR}/${DATASET_NAME}"
DATA_DIR="${DATASET_DIR}"
INPUT_DIR="${DATASET_DIR}/input"
############################################################ RUN SETUP ############################################################
RUN_NAME="eval_stage1_recall_test_with_more_database_information"

# TODO: Set to true to run files in parallel, false for sequential execution
PARALLEL=true
# TODO: set to true, then only print out the commands that would be executed, without actually running them
TEST=false

# write stdout and stderr to log files (in addition to printing to console)
mkdir -p "${BASE_LOG_DIR}/${DATASET_NAME}/${RUN_NAME}"
exec > >(tee -i "${BASE_LOG_DIR}/${DATASET_NAME}/${RUN_NAME}/stdout.log") 2> >(tee -i "${BASE_LOG_DIR}/${DATASET_NAME}/${RUN_NAME}/stderr.log" >&2)

# Load all input files from directory
shopt -s nullglob
FILES=("${INPUT_DIR}"/*.json)
FILES=("${FILES[@]#${INPUT_DIR}/}")
shopt -u nullglob

############################################################ RUN EVERYTHING ############################################################

echo "================== Running '${RUN_NAME}' with dataset '${DATASET_NAME}'"
COMPLETED=0
TOTAL="${#FILES[@]}"

for FILE in "${FILES[@]}"; do

    INPUT_FILE="${INPUT_DIR}/${FILE}"
    QUERY=$(jq -r '.value.query' "${INPUT_FILE}")
    QUERY_SLUG=$(basename "${INPUT_FILE}" .json)
    INPUT_FILE="${INPUT_DIR}/${FILE}"
    QUERY=$(jq -r '.value.query' "${INPUT_FILE}")
    QUERY_SLUG=$(basename "${INPUT_FILE}" .json)

    USR=$(echo "${USER:0:2}" | tr '[:lower:]' '[:upper:]')
    LOG_DIR="${BASE_LOG_DIR}/${DATASET_NAME}/${RUN_NAME}/${QUERY_SLUG}"
    NAME="[$USR] ${RUN_NAME} (${DATASET_NAME}): ${QUERY}"

    echo "  ================== Running file '${FILE}'"
    mkdir -p "${LOG_DIR}"

    if [[ "${TEST}" == "true" ]]; then
        echo "      [TEST] Would run: python3 -m scripts.main --name \"${NAME}\" --input \"${INPUT_FILE}\" --output_dir \"${LOG_DIR}\" --cache_dir \"${CACHE_DIR}\" --data_dir \"${DATA_DIR}\" --start-stage 1 --end-stage 1"
    else
        python3 -m scripts.main \
          --name "${NAME}" \
          --input "${INPUT_FILE}" \
          --output_dir "${LOG_DIR}" \
          --cache_dir "${CACHE_DIR}" \
          --data_dir "${DATA_DIR}" \
          --config "${PROJECT_DIR}/config.yaml" \
          --start-stage 1 --end-stage 1 \
          > >(tee "${LOG_DIR}/stdout.log") \
          2> >(tee "${LOG_DIR}/stderr.log" >&2)
    fi

    COMPLETED=$((COMPLETED + 1))

    echo "  ================== RAM usage after query ${COMPLETED}/${TOTAL}"
    free -h

    ############################################################ EVALUATE ############################################################

    if [[ "${TEST}" != "true" ]]; then
        echo "  ================== Computing precision / recall / F1 (${COMPLETED}/${TOTAL})"
        python3 -m scripts.eval_stage1_recall \
          --log_dir "${BASE_LOG_DIR}/${DATASET_NAME}/${RUN_NAME}" \
          --input_dir "${INPUT_DIR}" \
          --chart_output "${BASE_LOG_DIR}/${DATASET_NAME}/${RUN_NAME}/chart_data"
    fi

    echo "  ================== Query done: ${COMPLETED}/${TOTAL} — ${QUERY_SLUG}"

done

# echo "================== All ${TOTAL} queries completed"

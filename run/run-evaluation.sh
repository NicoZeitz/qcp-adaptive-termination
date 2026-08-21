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
DATASET_NAME="wikidata-federated-overlap-dataset"

DATASET_DIR="${BASE_DATASET_DIR}/${DATASET_NAME}"
DATA_DIR="${DATASET_DIR}/dataset"
INPUT_DIR="${DATASET_DIR}/input"
############################################################ RUN SETUP ############################################################
RUN_NAME="evaluate-selection-strategies"

# TODO: Set to true to run files in parallel, false for sequential execution
PARALLEL=false
# TODO: set to true, then only print out the commands that would be executed, without actually running them
TEST=true

# write stdout and stderr to log files (in addition to printing to console)
mkdir -p "${BASE_LOG_DIR}/${DATASET_NAME}/${RUN_NAME}"
exec > >(tee -i "${BASE_LOG_DIR}/${DATASET_NAME}/${RUN_NAME}/stdout.log") 2> >(tee -i "${BASE_LOG_DIR}/${DATASET_NAME}/${RUN_NAME}/stderr.log" >&2)

# Load all input files from directory
shopt -s nullglob
FILES=("${INPUT_DIR}"/*.json)
FILES=("${FILES[@]#${INPUT_DIR}/}")
shopt -u nullglob

# Or use a specific list of files
# FILES=(
#     "What_is_the_average_population_of_German_states.json"
# )

############################################################ RUN EVERYTHING ############################################################

echo "================== Running '${RUN_NAME}' with dataset '${DATASET_NAME}'"
for FILE in "${FILES[@]}"; do

    INPUT_FILE="${INPUT_DIR}/${FILE}"
    QUERY=$(jq -r '.value.query' "${INPUT_FILE}")
    QUERY_SLUG=$(basename "${INPUT_FILE}" .json)

    # take first 2 letter from $USER and uppercase them
    USR=$(echo "${USER:0:2}" | tr '[:lower:]' '[:upper:]')
    LOG_DIR="${BASE_LOG_DIR}/${DATASET_NAME}/${RUN_NAME}/${QUERY_SLUG}"
    NAME="[$USR] ${RUN_NAME} (${DATASET_NAME}): ${QUERY}"

    echo "  ================== Running file '${FILE}'"
    mkdir -p "${LOG_DIR}"

    if [[ "${TEST}" == "true" ]]; then
        # echo "      [TEST] Would run: python3 -m scripts.evaluate_selection_strategies --name \"${NAME}\" --input \"${INPUT_FILE}\" --output_dir \"${LOG_DIR}\" --data_dir \"${DATA_DIR}\" --start-stage 3 --end-stage 3"
        continue
    fi

    python3 -m scripts.evaluate_selection_strategies \
      --name "${NAME}" \
      --input "${INPUT_FILE}" \
      --output_dir "${LOG_DIR}" \
      --data_dir "${DATA_DIR}" \
      --start-stage 3 --end-stage 3 \
      > >(tee "${LOG_DIR}/stdout.log") \
      2> >(tee "${LOG_DIR}/stderr.log" >&2) &

    if [[ "${PARALLEL}" != "true" ]]; then
        wait
    fi
done

if [[ "${PARALLEL}" == "true" ]]; then
    wait
fi

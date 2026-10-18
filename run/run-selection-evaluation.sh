#!/usr/bin/env bash
############################################################ GENERAL SETUP ############################################################
set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${RUN_DIR}/paths.sh"
cd "$PROJECT_DIR"

# sync project dependencies with uv
uv sync

# activate the virtual environment
source .venv/bin/activate

############################################################ DATASET SPECIFIC SETUP ############################################################
DATASET_NAME="wikidata-federated-overlap-dataset"

DATASET_DIR="${BASE_DATASET_DIR}/${DATASET_NAME}"
DATA_DIR="${DATASET_DIR}"
INPUT_DIR="${DATASET_DIR}/dataset/_meta/input"
############################################################ RUN SETUP ############################################################
RUN_NAME="evaluate-selection-strategies"

# TODO: Set to true to run files in parallel, false for sequential execution
PARALLEL=true
# TODO: set to true, then only print out the commands that would be executed, without actually running them
TEST=false

# write stdout and stderr to log files (in addition to printing to console)
mkdir -p "${BASE_LOG_DIR}/${DATASET_NAME}/${RUN_NAME}"
# exec > >(tee -i "${BASE_LOG_DIR}/${DATASET_NAME}/${RUN_NAME}/stdout.log") 2> >(tee -i "${BASE_LOG_DIR}/${DATASET_NAME}/${RUN_NAME}/stderr.log" >&2)

# Load all input files from directory
shopt -s nullglob
FILES=("${INPUT_DIR}"/*.json)
FILES=("${FILES[@]#${INPUT_DIR}/}")
shopt -u nullglob

############################################################ RUN EVERYTHING ############################################################

CONFIG=$(python3 - <<PYEOF
import yaml, sys
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

print(yaml.dump(cfg, default_flow_style=False))
PYEOF
)

echo "================== Running '${RUN_NAME}' with dataset '${DATASET_NAME}'"
for FILE in "${FILES[@]}"; do

    INPUT_FILE="${INPUT_DIR}/${FILE}"
    QUERY=$(jq -r '.value.query' "${INPUT_FILE}")
    QUERY_SLUG=$(basename "${INPUT_FILE}" .json)

    # take first 2 letter from $USER and uppercase them
    USR=$(echo "${RUN_USER:0:2}" | tr '[:lower:]' '[:upper:]')
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
      --cache_dir "${CACHE_DIR}" \
      --config "${CONFIG}" \
      --start-stage 3 --end-stage 3 &
    #   > >(tee "${LOG_DIR}/stdout.log") \
    #   2> >(tee "${LOG_DIR}/stderr.log" >&2) &

    if [[ "${PARALLEL}" != "true" ]]; then
        wait
    fi
done

if [[ "${PARALLEL}" == "true" ]]; then
    wait
fi

if [[ "${TEST}" != "true" && ${#FILES[@]} -gt 0 ]]; then
    python3 -c 'from pathlib import Path; import sys; from scripts.evaluate_selection_strategies import aggregate_selection_results; aggregate_selection_results(Path(sys.argv[1]), Path(sys.argv[2]))' \
      "${BASE_LOG_DIR}/${DATASET_NAME}/${RUN_NAME}" \
      "${INPUT_DIR}"
fi
# python3 -m scripts.evaluate_selection_strategies \

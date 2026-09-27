USER=$(whoami)
ASSET_USER="nzeitz"

PROJECT_DIR="/mnt/labstore/${USER}/qcp-adaptive-termination"
CACHE_DIR="/mnt/labstore/${ASSET_USER}/.cache"

BASE_DATASET_DIR="/mnt/labstore/${ASSET_USER}/dataset"
BASE_LOG_DIR="/mnt/labstore/${USER}/.logs"

DATASET_NAME="wikidata-federated-overlap-dataset"

DATASET_DIR="${BASE_DATASET_DIR}/${DATASET_NAME}"
DATA_DIR="${DATASET_DIR}"
INPUT_DIR="${DATASET_DIR}/input_simple"

RUN_NAME="GLM_eval_stage1_recall_wiki_simple_recursive_llm_filter_embedding_and_prompt_change"

python3 -m scripts.eval_stage1_recall \
    --log_dir "${BASE_LOG_DIR}/${DATASET_NAME}/${RUN_NAME}" \
    --input_dir "${INPUT_DIR}" \
    --chart_output "${BASE_LOG_DIR}/${DATASET_NAME}/${RUN_NAME}/chart_data"
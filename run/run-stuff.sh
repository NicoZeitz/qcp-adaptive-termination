USER=$(whoami)
ASSET_USER="nzeitz"

PROJECT_DIR="/mnt/labstore/${USER}/qcp-adaptive-termination"
CACHE_DIR="/mnt/labstore/${ASSET_USER}/.cache"

BASE_DATASET_DIR="/mnt/labstore/${ASSET_USER}/dataset"
BASE_LOG_DIR="/mnt/labstore/${USER}/.logs"

DATASET_NAME="bird-train-dev"

DATASET_DIR="${BASE_DATASET_DIR}/${DATASET_NAME}"
DATA_DIR="${DATASET_DIR}"
INPUT_DIR="${DATASET_DIR}/input"

RUN_NAME="eval_stage1_recall_test_with_more_database_information"

python3 -m scripts.eval_stage1_recall \
    --log_dir "${BASE_LOG_DIR}/${DATASET_NAME}/${RUN_NAME}" \
    --input_dir "${INPUT_DIR}" \
    --chart_output "${BASE_LOG_DIR}/${DATASET_NAME}/${RUN_NAME}/chart_data"
#!/usr/bin/env bash
############################################################ GENERAL SETUP ############################################################
set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${RUN_DIR}/.." && pwd)"
RUN_USER="$(whoami)"

case "$RUN_USER" in
    I550817)
        DATASET_ROOT_DEFAULT="${HOME}/TUDevelop/dataset"
        CACHE_DIR_DEFAULT="${PROJECT_DIR}/.cache"
        LOG_ROOT_DEFAULT="${PROJECT_DIR}/.logs"
        ;;
    nzeitz)
        DATASET_ROOT_DEFAULT="/mnt/labstore/nzeitz/dataset"
        CACHE_DIR_DEFAULT="/mnt/labstore/nzeitz/.cache"
        LOG_ROOT_DEFAULT="/mnt/labstore/nzeitz/.logs"
        ;;
    *)
        printf 'Unsupported account: %s\n' "$RUN_USER" >&2
        return 1
        ;;
esac

BASE_DATASET_DIR="${QCP_DATASET_ROOT:-$DATASET_ROOT_DEFAULT}"
CACHE_DIR="${QCP_CACHE_DIR:-$CACHE_DIR_DEFAULT}"
BASE_LOG_DIR="${QCP_LOG_ROOT:-$LOG_ROOT_DEFAULT}"

cd "$PROJECT_DIR"

# sync project dependencies with uv
uv sync

# activate the virtual environment
source .venv/bin/activate

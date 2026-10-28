#!/usr/bin/env bash
############################################################ GENERAL SETUP ############################################################
set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${RUN_DIR}/.." && pwd)"
RUN_USER="$(whoami)"

case "$RUN_USER" in
    I550817)
        BASE_DATASET_DIR="${HOME}/TUDevelop/dataset"
        CACHE_DIR="${PROJECT_DIR}/.cache"
        BASE_LOG_DIR="${PROJECT_DIR}/.logs"
        ;;
    nzeitz)
        BASE_DATASET_DIR="/mnt/labstore/nzeitz/dataset"
        CACHE_DIR="/mnt/labstore/nzeitz/.cache"
        BASE_LOG_DIR="/mnt/labstore/nzeitz/.logs"
        ;;
    *)
        printf 'Unsupported account: %s\n' "$RUN_USER" >&2
        return 1
        ;;
esac

cd "$PROJECT_DIR"

# sync project dependencies with uv
uv sync

# activate the virtual environment
source .venv/bin/activate

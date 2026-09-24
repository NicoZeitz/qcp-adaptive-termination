#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

uv sync
source .venv/bin/activate

ASSET_USER="nzeitz"
CACHE_DIR="/mnt/labstore/${ASSET_USER}/.cache"

# Pass --delete to remove old .json files after successful migration
DELETE_FLAG="${1:-}"

python3 -m scripts.migrate_cache "${CACHE_DIR}" ${DELETE_FLAG}

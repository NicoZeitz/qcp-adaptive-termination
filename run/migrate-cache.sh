#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

ASSET_USER="nzeitz"
CACHE_DIR="/mnt/labstore/${ASSET_USER}/.cache"

# Pass --delete to remove old .json files after successful migration
DELETE_FLAG="${1:-}"

uv run python src/scripts/migrate_cache.py "${CACHE_DIR}" ${DELETE_FLAG}

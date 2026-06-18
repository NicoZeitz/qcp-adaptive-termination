#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-../data/unknown-synthetic}"
OUT_DIR="${2:-../data/runs/unknown-synthetic}"
PROVIDER="${3:-OpenAI}"

mapfile -d '' INPUT_FILES < <(find "$DATA_DIR" -maxdepth 1 -name "*.json" -print0 | sort -z)

if [ ${#INPUT_FILES[@]} -eq 0 ]; then
    echo "No JSON files found in $DATA_DIR" >&2
    exit 1
fi

echo "Running ${#INPUT_FILES[@]} file(s) from $DATA_DIR"

for INPUT_FILE in "${INPUT_FILES[@]}"; do
    NAME=$(basename "$INPUT_FILE" .json)
    echo "→ $NAME"
    uv run main.py \
        --name "$NAME" \
        --input "$INPUT_FILE" \
        --log_dir "$OUT_DIR" \
        --start_stage 1 \
        --end_stage 3 \
        --provider "$PROVIDER"
done

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$SCRIPT_DIR/src"

DATA_DIR="${1:-$SCRIPT_DIR/data/unknown-synthetic}"
OUT_DIR="${2:-$SCRIPT_DIR/data/runs/unknown-synthetic}"
PROVIDER="${3:-OpenAI}"

INPUT_FILES=()
while IFS= read -r -d '' f; do
    INPUT_FILES+=("$f")
done < <(find "$DATA_DIR" -maxdepth 1 -name "*.json" -print0 | sort -z)

if [ ${#INPUT_FILES[@]} -eq 0 ]; then
    echo "No JSON files found in $DATA_DIR" >&2
    exit 1
fi

echo "Running ${#INPUT_FILES[@]} file(s) from $DATA_DIR"

cd "$SRC_DIR"

# only first file
INPUT_FILES=("${INPUT_FILES[0]}")

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

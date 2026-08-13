#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

INPUT="${1:-dataset/wikidata/input/fifa_england_worldcup_game_count.json}"
DATA_DIR="${2:-dataset/wikidata/dataset}"
BASE_RUNS=".logs/wikidata/selection-path-visualization-all-strategies"

DATASET_NAME=$(basename "${INPUT}" .json)

echo "=== Selection path visualization (all strategies): ${DATASET_NAME} ==="

# for STRATEGY in sequential embedding count distinct_count hll_sketch theta_sketch min_hash posterior; do
for STRATEGY in posterior; do
    CONFIG=$(python3 - <<PYEOF
import yaml
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)
cfg.setdefault("selection_kwargs", {})["registry_type"] = "technical"
cfg["selection_type"] = "${STRATEGY}"
print(yaml.dump(cfg, default_flow_style=False))
PYEOF
)

    echo "--- Strategy: ${STRATEGY} ---"

    uv run python -m scripts.selection_path_visualization \
        --input "${INPUT}" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${BASE_RUNS}/${DATASET_NAME}/${STRATEGY}" \
        --cache_dir ".cache" \
        --config "${CONFIG}" \
        --start-stage 3 --end-stage 3
done

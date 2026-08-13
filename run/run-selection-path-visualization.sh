#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

INPUT="${1:-dataset/wikidata/input/airports_germany_average_elevation.json}"
INPUT="${1:-dataset/wikidata/input/award_winners_asian.json}"
INPUT="${1:-dataset/wikidata/input/companies_mining_usa_canada_mexico.json}"
INPUT="${1:-dataset/wikidata/input/fifa_europe_kurt_referee.json}"
INPUT="${1:-dataset/wikidata/input/fifa_england_worldcup_game_count.json}"
DATA_DIR="${2:-dataset/wikidata/dataset}"
BASE_RUNS=".logs/wikidata/selection-path-visualization"

CONFIG=$(python3 - <<PYEOF
import yaml
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)
cfg.setdefault("selection_kwargs", {})["registry_type"] = "technical"
print(yaml.dump(cfg, default_flow_style=False))
PYEOF
)

DATASET_NAME=$(basename "${INPUT}" .json)

echo "=== Selection path visualization: ${DATASET_NAME} ==="

uv run python -m scripts.selection_path_visualization \
    --input "${INPUT}" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${BASE_RUNS}/${DATASET_NAME}" \
    --cache_dir ".cache" \
    --config "${CONFIG}" \
    --start-stage 3 --end-stage 3

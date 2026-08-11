#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

DATA_DIR="dataset/wikidata/dataset"
BASE_RUNS=".logs/wikidata/embedding-visualization"

DATASETS=(
    "airports_overlap_sampled"
    "fifa_overlap_sampled"
    "companies_overlap_sampled"
    "award_winners_overlap_sampled"
    "all_datasets_overlap_sampled"
)
# DATASETS=(
#     "airports_overlap_sampled"
# )

REGISTRY_TYPES=("technical" "natural_language")

for DATASET in "${DATASETS[@]}"; do
    for REGISTRY_TYPE in "${REGISTRY_TYPES[@]}"; do
        echo "=== ${DATASET} / ${REGISTRY_TYPE} ==="

        CONFIG=$(python3 - <<PYEOF
import yaml
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)
cfg.setdefault("selection_kwargs", {})["registry_type"] = "${REGISTRY_TYPE}"
print(yaml.dump(cfg, default_flow_style=False))
PYEOF
)

        uv run python -m scripts.embedding_visualization \
            --input "${DATA_DIR}/${DATASET}.json" \
            --data_dir "${DATA_DIR}" \
            --output_dir "${BASE_RUNS}/${DATASET}/${REGISTRY_TYPE}" \
            --cache_dir ".cache" \
            --config "${CONFIG}"
    done
done

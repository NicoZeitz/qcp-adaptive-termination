#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

BASE_DATA="/mnt/labstore/fwolf/data/.queries"
BASE_RUNS="/mnt/labstore/fwolf/.logs"

CONFIG=$(python3 - <<'PYEOF'
import yaml, sys
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

cfg["estimator_mode"] = "full"
cfg["selection_type"] = "embedding"
cfg["continue_after_final_result"] = True

print(yaml.dump(cfg, default_flow_style=False))
PYEOF
)

uv run python -m scripts.main \
  --name "bird aware one table per agent: q53" \
  --input "${BASE_DATA}/bird_aware_one_table_per_agent/q53.json" \
  --output_dir "${BASE_RUNS}/bird_aware_one_table_per_agent" \
  --cache_dir ".cache" \
  --config "${CONFIG}" \
  --start-stage 1 --end-stage 1
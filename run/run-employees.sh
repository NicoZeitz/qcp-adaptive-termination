#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

BASE_DATA="data/wikidata/companies/split_overlap_employees"
BASE_RUNS=".logs/wikidata/companies/split_overlap_employees"

CONFIG=$(python3 - <<'PYEOF'
import yaml, sys
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

cfg["estimator_mode"] = "full"
cfg["selection_type"] = "theta_sketch_ucb"
cfg["continue_after_final_result"] = True

print(yaml.dump(cfg, default_flow_style=False))
PYEOF
)

# uv run qcp-adaptive-termination \
uv run python -m scripts.main \
  --name "employees" \
  --input "${BASE_DATA}/employees.json" \
  --output_dir "${BASE_RUNS}/employees" \
  --cache_dir ".cache" \
  --data_dir "${BASE_DATA}" \
  --config "${CONFIG}" \
  --start-stage 1 --end-stage 3
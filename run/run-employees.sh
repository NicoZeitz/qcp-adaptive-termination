#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source ../.venv/bin/activate

BASE_DATA="../data/wikidata/companies/split_overlap_employees"
BASE_RUNS="../logs/wikidata/companies/split_overlap_employees"

CONFIG=$(cat <<EOF
{
  "estimator_mode": "full",
  "include_confidence_intervals": true,
  "use_bucket": true,
  "use_monte_carlo": false,
  "continue_after_final_result": true,
  "selection_type": "theta_sketch_ucb",
  "data_path_prefix": "${BASE_DATA}/"
}
EOF
)

(cd ../src && uv run main.py \
  --name "employees" \
  --input "${BASE_DATA}/employees.json" \
  --log_dir "${BASE_RUNS}/employees" \
  --config "${CONFIG}" \
  --start_stage 1 --end_stage 3)

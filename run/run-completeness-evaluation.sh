#!/usr/bin/env bash
# Run every completeness-evaluation variant across datasets.
#
# The DATASET_VARIANTS list below is the single place to toggle what runs:
# comment out lines (or leave them commented) to run only a subset. Each entry is
#   "<dataset-name>|<dataset-dir>|<ordering> <ordering> ..."
# expanded per ordering over every STOPPING variant.
#
# Each variant uses a stable run directory. A fresh pass runs every query; then any
# query that failed (timeout/error) is re-run alone with --continue (folder cleared,
# then re-aggregated) up to MAX_RETRIES passes or until nothing is left failing.
# Per-dataset cross-variant analysis runs after that dataset's variants.
#
# Usage: run-completeness-evaluation.sh [<dataset-name>/<dataset-dir> ...]   (no argument = all)
# Invocations are queued: a second call waits until the running one has finished.
# Env: MAX_RETRIES (default 2) auto re-run passes per variant after the first failed pass.
set -uo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELECTED=("$@")
source "${RUN_DIR}/setup.sh"

# Active datasets (leave commented the ones you don't currently want).
DATASET_VARIANTS=(
  "wikidata-federated-overlap-dataset|dataset|posterior random"
  "wikidata-federated-overlap-dataset|dataset-50|posterior random"
  "web-data-commons-product|dataset|count"
  # "crowdwork-query-estimation|dataset_vldbsigmod|posterior random"
  # "crowdwork-query-estimation|dataset_siliconvalley_employee|posterior random"
  # "crowdwork-query-estimation|dataset_GDP2025|posterior random"
  # "crowdwork-query-estimation|dataset_siliconvalley_revenue|posterior random"
)

STOPPINGS=(none llm-prevent estimator estimator-llm-prevent estimator-llm-estimator-reasoning)
MAX_RETRIES="${MAX_RETRIES:-2}"

SUMMARY="${BASE_LOG_DIR}/completeness-$(date +%Y-%m-%d-%H-%M-%S).log"
mkdir -p "${BASE_LOG_DIR}"

exec 9>"${BASE_LOG_DIR}/.completeness-evaluation.lock"
if ! flock -n 9; then
  echo "Another run-completeness-evaluation.sh is running, waiting for it to finish..."
  flock 9
fi
echo "Writing run summary to ${SUMMARY}"
echo "Max re-run passes per variant: ${MAX_RETRIES}"

# Print the number of non-complete queries in a run's status CSV (-1 when never aggregated).
failed_status() {
  python3 - "$1" <<'PY'
import csv, sys
try:
    with open(sys.argv[1], encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(sum(1 for r in rows if r.get("status") != "complete"))
except FileNotFoundError:
    print(-1)
PY
}

run_variant() {
  local name="$1" sub="$2" ordering="$3" stopping="$4" mode="$5"
  local run_dir="${BASE_LOG_DIR}/${name}/${sub}/evaluate-completeness-${ordering}-${stopping}"
  local continue_args=()
  if [[ "$mode" == "continue" ]]; then
    continue_args=(--continue)
  else
    rm -rf "${run_dir}"
  fi
  mkdir -p "${run_dir}"
  echo "================== [${mode}] ${name}/${sub} ${ordering}/${stopping} -> ${run_dir}"
  python3 -m scripts.evaluation completeness \
    --dataset_dir "${BASE_DATASET_DIR}/${name}/${sub}" \
    --output_dir "${run_dir}" \
    --cache_dir "${CACHE_DIR}" \
    --config "${PROJECT_DIR}/config.yaml" \
    --config_override "${RUN_DIR}/variants/order-${ordering}.yaml" \
    --config_override "${RUN_DIR}/variants/stop-${stopping}.yaml" \
    --parallel \
    --max_workers 4 \
    "${continue_args[@]}"
}

for entry in "${DATASET_VARIANTS[@]}"; do
  IFS="|" read -r DATASET_NAME DATASET_DIR ORDERINGS <<< "${entry}"
  if (( ${#SELECTED[@]} > 0 )) && [[ ! " ${SELECTED[*]} " == *" ${DATASET_NAME}/${DATASET_DIR} "* ]]; then
    continue
  fi
  for ORDERING in ${ORDERINGS}; do
    for STOPPING in "${STOPPINGS[@]}"; do
      START="$(date +%s)"
      STATUS="ok"
      run_variant "${DATASET_NAME}" "${DATASET_DIR}" "${ORDERING}" "${STOPPING}" fresh || STATUS="FAILED"
      RUN_DIR_STABLE="${BASE_LOG_DIR}/${DATASET_NAME}/${DATASET_DIR}/evaluate-completeness-${ORDERING}-${STOPPING}"
      failed="$(failed_status "${RUN_DIR_STABLE}/evaluator_run_status.csv")"
      attempts=1
      while (( failed > 0 && attempts <= MAX_RETRIES )); do
        run_variant "${DATASET_NAME}" "${DATASET_DIR}" "${ORDERING}" "${STOPPING}" continue || STATUS="FAILED"
        failed="$(failed_status "${RUN_DIR_STABLE}/evaluator_run_status.csv")"
        attempts=$((attempts + 1))
      done
      echo "$(date -Iseconds) ${STATUS} ${DATASET_NAME}/${DATASET_DIR} ${ORDERING} ${STOPPING} (passes=$attempts remaining_failed=${failed} $(( $(date +%s) - START ))s)" | tee -a "${SUMMARY}"
    done
  done
  python3 -m scripts.completeness_evaluation.analysis \
    --results_dir "${BASE_LOG_DIR}/${DATASET_NAME}/${DATASET_DIR}" \
    || echo "$(date -Iseconds) FAILED analysis ${DATASET_NAME}/${DATASET_DIR}" | tee -a "${SUMMARY}"
done

#!/usr/bin/env bash
# Run every completeness-evaluation experiment sequentially, ordered for maximal cache reuse:
#   per dataset and ordering: none -> llm-prevent -> estimator -> estimator-llm-prevent
#   -> estimator-llm-estimator-reasoning
# The no-stop baseline caches every data-agent call; LLM-only caches the LLM checks that the combined
# variants replay until the estimator stops; estimator_reasoning only differs on long lists.
# A failing experiment is logged and the chain continues. Afterwards the cross-variant analysis runs per dataset.
# Usage: run-completeness-all.sh [<dataset-name>/<dataset-dir> ...]   (no arguments = all datasets)
# Invocations are queued: a second call waits until the running one has finished.
set -uo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELECTED=("$@")
source "${RUN_DIR}/setup.sh"

STOPPINGS=(none llm-prevent estimator estimator-llm-prevent estimator-llm-estimator-reasoning)

# dataset-name|dataset-dir|orderings
EXPERIMENTS=(
  "crowdwork-query-estimation|dataset_GDP2025|posterior random"
  "crowdwork-query-estimation|dataset_siliconvalley_employee|posterior random"
  "crowdwork-query-estimation|dataset_siliconvalley_revenue|posterior random"
  "crowdwork-query-estimation|dataset_vldbsigmod|posterior random"
  "wikidata-federated-overlap-dataset|dataset|posterior random"
  "wikidata-federated-overlap-dataset|dataset-50|posterior random"
  "web-data-commons-product|dataset|count"
)

SUMMARY="${BASE_LOG_DIR}/completeness-all-$(date +%Y-%m-%d-%H-%M-%S).log"
mkdir -p "${BASE_LOG_DIR}"

exec 9>"${BASE_LOG_DIR}/.completeness-all.lock"
if ! flock -n 9; then
  echo "Another run-completeness-all.sh is running, waiting for it to finish..."
  flock 9
fi
echo "Writing run summary to ${SUMMARY}"

for experiment in "${EXPERIMENTS[@]}"; do
  IFS="|" read -r DATASET_NAME DATASET_DIR ORDERINGS <<< "${experiment}"
  if (( ${#SELECTED[@]} > 0 )) && [[ ! " ${SELECTED[*]} " == *" ${DATASET_NAME}/${DATASET_DIR} "* ]]; then
    continue
  fi
  for ORDERING in ${ORDERINGS}; do
    for STOPPING in "${STOPPINGS[@]}"; do
      START="$(date +%s)"
      if bash "${RUN_DIR}/run-completeness-variant.sh" "${DATASET_NAME}" "${DATASET_DIR}" "${ORDERING}" "${STOPPING}"; then
        STATUS="ok"
      else
        STATUS="FAILED"
      fi
      echo "$(date -Iseconds) ${STATUS} ${DATASET_NAME}/${DATASET_DIR} ${ORDERING} ${STOPPING} ($(( $(date +%s) - START ))s)" | tee -a "${SUMMARY}"
    done
  done
  python3 -m scripts.completeness_evaluation.analysis \
    --results_dir "${BASE_LOG_DIR}/${DATASET_NAME}/${DATASET_DIR}" \
    || echo "$(date -Iseconds) FAILED analysis ${DATASET_NAME}/${DATASET_DIR}" | tee -a "${SUMMARY}"
done

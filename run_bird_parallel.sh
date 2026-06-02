#!/usr/bin/env bash

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage:
  ./run_bird_parallel.sh [--test] [--sample] <dataset|dataset_vertical|dataset_own|data/bird/dataset|data/bird/dataset_vertical|data/bird/dataset_own|/data/bird/dataset|/data/bird/dataset_vertical|/data/bird/dataset_own>

Examples:
  ./run_bird_parallel.sh dataset
  ./run_bird_parallel.sh --test dataset
  ./run_bird_parallel.sh --sample dataset_own
  ./run_bird_parallel.sh dataset_vertical
  ./run_bird_parallel.sh dataset_own
  ./run_bird_parallel.sh /data/bird/dataset
EOF
}

test_mode=false
sample_mode=false
input_arg=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --test|-t)
      test_mode=true
      shift
      ;;
    --sample|-s)
      sample_mode=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      if [[ -n "$input_arg" ]]; then
        echo "Only one dataset argument is supported." >&2
        usage
        exit 1
      fi
      input_arg="$1"
      shift
      ;;
  esac
done

if [[ -z "$input_arg" ]]; then
  usage
  exit 1
fi

case "$input_arg" in
  dataset|data/bird/dataset|/data/bird/dataset)
    dataset_name="dataset"
    run_name="Bird"
    ;;
  dataset_vertical|data/bird/dataset_vertical|/data/bird/dataset_vertical)
    dataset_name="dataset_vertical"
    run_name="Bird Vertical"
    ;;
  dataset_own|data/bird/dataset_own|/data/bird/dataset_own)
    dataset_name="dataset_own"
    run_name="Bird Own"
    ;;
  *)
    echo "Unsupported dataset argument: $input_arg" >&2
    usage
    exit 1
    ;;
esac

input_dir="$ROOT_DIR/data/bird/$dataset_name"
run_base_dir="$ROOT_DIR/data/runs/bird/$dataset_name"

if [[ ! -d "$input_dir" ]]; then
  echo "Input directory not found: $input_dir" >&2
  exit 1
fi

shopt -s nullglob
json_files=("$input_dir"/*.json)
shopt -u nullglob

if [[ ${#json_files[@]} -eq 0 ]]; then
  echo "No JSON files found in: $input_dir" >&2
  exit 1
fi

if [[ "$test_mode" == true ]]; then
  json_files=("${json_files[0]}")
  echo "Test mode enabled: only running ${json_files[0]}"
fi

if [[ "$sample_mode" == true ]]; then
  # list of 10 questions to sample here
  questions=(
    "q992.json"
    "q752.json"
    "q694.json"
    "q1413.json"
    "q1447.json"
    "q1533.json"
    "q1220.json"
    "q1221.json"
    "q1375.json"
    "q1237.json"
  )
  json_files=()
  for q in "${questions[@]}"; do
    if [[ -f "$input_dir/$q" ]]; then
      json_files+=("$input_dir/$q")
    else
      echo "Warning: Sample question file not found: $input_dir/$q" >&2
    fi
  done
fi

echo "Found ${#json_files[@]} JSON files to process in: $input_dir"

mkdir -p "$run_base_dir"

# shellcheck disable=SC1091
source "$ROOT_DIR/.venv/bin/activate"

declare -a pids
declare -a names

for json_file in "${json_files[@]}"; do
  file_name="$(basename "$json_file")"
  file_stem="${file_name%.json}"

  run_dir="$run_base_dir/$file_stem"
  mkdir -p "$run_dir"

  cp "$json_file" "$run_dir/$file_name"

  (
    cd "$ROOT_DIR/src" || exit 1
    uv run main.py \
      --name "$run_name" \
      --input "../data/bird/$dataset_name/$file_name" \
      --log_dir "../data/runs/bird/$dataset_name/$file_stem" \
      >"$run_dir/stdout.txt" \
      2>"$run_dir/stderr.txt"
  ) &

  pids+=("$!")
  names+=("$file_stem")
done

echo "Started ${#pids[@]} parallel runs for $dataset_name"

failures=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "Run failed for ${names[$i]} (see $run_base_dir/${names[$i]}/stderr.txt)" >&2
    failures=$((failures + 1))
  fi
done

if [[ $failures -gt 0 ]]; then
  echo "Completed with $failures failing run(s)." >&2
  exit 1
fi

echo "All runs completed successfully. Outputs are in: $run_base_dir"
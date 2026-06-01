#!/usr/bin/env bash

# q53 ❌ [6, 114, 2216] missing from result
# q63 ✅ result includes [null null Little Red Schoolhouse null] but not clear from query that that is needed
# q97 ✅ exact match
# q129 ❌ complete failure
# q216 ❌ only 2 entries, 12 expected
# q280 ⚠️ includes additional [TR447_1_3, unknown] with unknown carcinogenicity, rest ist correct
# q338 ❌ only 2 entries, 12 expected
# q344 ❌ complete failure
# q347 ❌ empty result
# q389 ❌ empty result
# q391 ❌ only 1 entry, 8 expected
# q404 ❌ empty result
# q410 ❌ 16 entries, 5 expected, only a partial overlap of the entries
# q427 ✅ exact match
# q448 ❌ complete failure
# q469 ❌ entries missing (Yes: 8ED, DDC, No: JMP, MB1, P02, S99)
# q481 ❌ complete failure
# q500 ❌ 1 entry missing (Magic: the Gathering Commander)
# q575 ❌ empty result
# q584 ❌ complete failure
# q611 ✅❓ same, but expected result does not deduplicate the reputation values for same user id with multiple badges on the same date
# q613 ✅❓ query asks for date, answer from agent correct, expected result is date + time instead so mismatch
# q622 ❌ empty result
# q652 ❌ only some badge names, post_ids missing
# q694 ❌ only 1 entry, 10 expected
# q752 ✅ exact match
# q753 ❌ empty result
# q840 ❌ empty result
# q848 ✅ exact match
# q856 ✅ answer differentiates by year, which is not needed by expected result
# q866 ✅ matches, but different columns
# q924 ✅ exact match
# q932 ❌ only 8 entries, 11 expected (e.g. 2007 missing)
# q973 ❌ empty result
# q992 ✅ matches, but different columns
# q1188 ✅ exact match
# q1220 ❌ only 10 entries, 20 expected (e.g. 912147 missing)
# q1221 ❌ only 26 entries, 38 expected (e.g. 71417 missing)
# q1237 ❌ empty result
# q1259 ❌ empty result
# q1375 ⚠️ has only last names, first names missing (otherwise correct)
# q1399 ❌ complete failure
# q1411 ❌ empty result
# q1413 ✅❓ same, but expected result does not deduplicate the zip codes
# q1417 ✅❓ same, but expected result does not deduplicate the event type
# q1436 ✅ exact match
# q1443 ✅ exact match
# q1447 ✅ exact match
# q1456 ✅❓ expected returns top 5 but non unique person, answer returns only top 3 deduplicated (because there are not more people matching)
# q1463 ✅ same, but weird column values (from noise)
# q1533 ✅❓ same, but expected result does not deduplicate value 126157.70 for customer id 6768


set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
	cat <<'EOF'
Usage:
	./run.sh <qid> [start_stage [end_stage]]

Examples:
	./run.sh q1221
	./run.sh 1221 1 2
EOF
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
	usage
	exit 0
fi

if [[ $# -lt 1 || $# -gt 3 ]]; then
	usage
	exit 1
fi

qid_input="$1"
start_stage="${2:-1}"
end_stage="${3:-3}"

if [[ "$qid_input" =~ ^q?[0-9]+$ ]]; then
	qid="q${qid_input#q}"
else
	echo "Invalid qid: $qid_input" >&2
	usage
	exit 1
fi

timestamp="$(date +%F_%H-%M-%S)"
run_dir="$ROOT_DIR/data/runs/run/${qid}_${timestamp}"
question_file="$ROOT_DIR/data/bird/dataset/${qid}.json"
final_result_glob_prefix="Stage ${start_stage}-${end_stage} - ${timestamp}"

if [[ ! -f "$question_file" ]]; then
	echo "Question file not found: $question_file" >&2
	exit 1
fi

mkdir -p "$run_dir"

(
	cd "$ROOT_DIR/src" || exit 1
	uv run main.py \
		--name "Bird" \
		--input "../data/bird/dataset/${qid}.json" \
		--log_dir "../data/runs/run/${qid}_${timestamp}" \
		--start_stage "$start_stage" \
		--end_stage "$end_stage" \
		>"$run_dir/stdout.txt" \
		2>"$run_dir/stderr.txt"
)

if [[ "$end_stage" -ge 3 ]]; then
    final_result_file="$run_dir/${final_result_glob_prefix} - final_result.json"

    if [[ ! -f "$final_result_file" ]]; then
        final_result_file="$(find "$run_dir" -maxdepth 1 -name '*final_result.json' -type f | sort | tail -n 1)"
    fi
    if [[ -z "$final_result_file" || ! -f "$final_result_file" ]]; then
        echo "Could not locate final result file in: $run_dir" >&2
        exit 1
    fi

	cd "$ROOT_DIR/src" || exit 1
	uv run q_repl.py "$qid" \
		--input "../data/bird/dataset/${qid}.json" \
		--final-result "${final_result_file}" \
		--repo-root "$ROOT_DIR" \
		> >(tee "$run_dir/q_repl_stdout.txt") \
		2> >(tee "$run_dir/q_repl_stderr.txt" >&2)
else
	echo "Skipping q_repl: end_stage is $end_stage (requires 3)"
fi
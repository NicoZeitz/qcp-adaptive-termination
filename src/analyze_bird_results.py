import argparse
import itertools
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ComparisonResult:
    status: str
    failure_reason: str | None
    match_mode: str | None
    expected_rows: int
    produced_rows: int
    expected_width: int
    produced_width: int
    projection_columns: list[str] | None
    exact_match: bool


def _normalize_value(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, float):
        if math.isnan(value):
            return ""
        if value.is_integer():
            return str(int(value))
        return format(value, ".15g")

    if isinstance(value, int):
        return str(value)

    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in {"nan", "none", "null"}:
            return ""
        return stripped

    return str(value).strip()


def _normalize_row(row: list[Any]) -> tuple[str, ...]:
    return tuple(_normalize_value(v) for v in row)


def _counter_from_rows(rows: list[list[Any]]) -> Counter[tuple[str, ...]]:
    return Counter(_normalize_row(r) for r in rows)


def _find_projection_match(
    produced_rows: list[list[Any]],
    produced_columns: list[str],
    expected_counter: Counter[tuple[str, ...]],
    expected_width: int,
) -> tuple[bool, list[str] | None]:
    produced_width = len(produced_columns)

    if expected_width <= 0 or produced_width < expected_width:
        return False, None

    for index_combo in itertools.combinations(range(produced_width), expected_width):
        projected_rows = [[row[i] if i < len(row) else None for i in index_combo] for row in produced_rows]
        projected_counter = _counter_from_rows(projected_rows)

        if projected_counter == expected_counter:
            return True, [produced_columns[i] for i in index_combo]

    return False, None


def _find_packed_list_match(
    expected_rows: list[list[Any]],
    produced_columns: list[str],
    produced_rows: list[list[Any]],
) -> tuple[bool, list[str] | None]:
    # Supports outputs that compress many expected single-column rows
    # into one list-valued cell (often paired with a count column).
    if not expected_rows:
        return False, None

    if any(len(row) != 1 for row in expected_rows):
        return False, None

    expected_counter = Counter(_normalize_value(row[0]) for row in expected_rows)

    for row in produced_rows:
        for col_idx, cell in enumerate(row):
            if isinstance(cell, list):
                list_counter = Counter(_normalize_value(item) for item in cell)
                if list_counter == expected_counter:
                    col_name = produced_columns[col_idx] if col_idx < len(produced_columns) else f"col_{col_idx}"
                    return True, [col_name]

    return False, None


def _detect_complete_failure(produced_columns: list[str], produced_rows: list[list[Any]]) -> str | None:
    if not produced_rows:
        return None

    failure_keywords = [
        "failed",
        "error",
        "exception",
        "could not",
        "cannot",
        "unable",
        "traceback",
        "not executed",
        "not execute",
    ]

    flat_text = " ".join(
        _normalize_value(cell).lower()
        for row in produced_rows
        for cell in row
        if _normalize_value(cell)
    )
    if any(keyword in flat_text for keyword in failure_keywords):
        return flat_text[:300]

    lowered_cols = {c.strip().lower() for c in produced_columns}
    if lowered_cols.intersection({"error", "errors", "exception", "message"}):
        return "error-like result columns present"

    return None


def _parse_expected_answer(question_file: Path) -> list[list[Any]]:
    with question_file.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    extra = payload.get("extra", {})
    answer = extra.get("answer", [])
    if not isinstance(answer, list):
        return []

    rows: list[list[Any]] = []
    for item in answer:
        if isinstance(item, list):
            rows.append(item)
        else:
            rows.append([item])
    return rows


def _parse_produced_result(final_result_file: Path) -> tuple[list[str], list[list[Any]]]:
    with final_result_file.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    result = payload.get("result", {})
    columns = result.get("columns", [])
    data = result.get("data", [])

    if not isinstance(columns, list):
        columns = []
    if not isinstance(data, list):
        data = []

    normalized_rows: list[list[Any]] = []
    for row in data:
        if isinstance(row, list):
            normalized_rows.append(row)
        else:
            normalized_rows.append([row])

    return [str(c) for c in columns], normalized_rows


def _latest_final_result(run_dir: Path) -> Path | None:
    candidates = sorted(run_dir.glob("*final_result.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        return None
    return candidates[-1]


def _compare(expected_rows: list[list[Any]], produced_columns: list[str], produced_rows: list[list[Any]]) -> ComparisonResult:
    expected_width = len(expected_rows[0]) if expected_rows else 0
    produced_width = len(produced_columns)

    failure_reason = _detect_complete_failure(produced_columns, produced_rows)
    if failure_reason:
        return ComparisonResult(
            status="complete_failure",
            failure_reason=failure_reason,
            match_mode=None,
            expected_rows=len(expected_rows),
            produced_rows=len(produced_rows),
            expected_width=expected_width,
            produced_width=produced_width,
            projection_columns=produced_columns if expected_width == produced_width else None,
            exact_match=False,
        )

    packed_match, packed_columns = _find_packed_list_match(
        expected_rows=expected_rows,
        produced_columns=produced_columns,
        produced_rows=produced_rows,
    )
    if packed_match:
        return ComparisonResult(
            status="exact_match",
            failure_reason=None,
            match_mode="packed_list",
            expected_rows=len(expected_rows),
            produced_rows=len(produced_rows),
            expected_width=expected_width,
            produced_width=produced_width,
            projection_columns=packed_columns,
            exact_match=True,
        )

    expected_counter = _counter_from_rows(expected_rows)

    exact = False
    projection_columns: list[str] | None = None

    if len(produced_rows) != len(expected_rows):
        return ComparisonResult(
            status="mismatch",
            failure_reason="row_count_mismatch",
            match_mode=None,
            expected_rows=len(expected_rows),
            produced_rows=len(produced_rows),
            expected_width=expected_width,
            produced_width=produced_width,
            projection_columns=None,
            exact_match=False,
        )

    match_mode: str | None = None
    if expected_width == produced_width and expected_width > 0:
        produced_counter = _counter_from_rows(produced_rows)
        exact = produced_counter == expected_counter
        projection_columns = produced_columns
        if exact:
            match_mode = "direct"
    elif expected_width > 0:
        exact, projection_columns = _find_projection_match(
            produced_rows=produced_rows,
            produced_columns=produced_columns,
            expected_counter=expected_counter,
            expected_width=expected_width,
        )
        if exact:
            match_mode = "projection"

    status = "exact_match" if exact else "mismatch"

    return ComparisonResult(
        status=status,
        failure_reason=None,
        match_mode=match_mode,
        expected_rows=len(expected_rows),
        produced_rows=len(produced_rows),
        expected_width=expected_width,
        produced_width=produced_width,
        projection_columns=projection_columns,
        exact_match=exact,
    )


def run(dataset: str) -> int:
    repo_root = Path(__file__).resolve().parent.parent
    source_dir = repo_root / "data" / "bird" / dataset
    runs_dir = repo_root / "data" / "runs" / "bird" / dataset

    if not source_dir.exists():
        raise FileNotFoundError(f"Source dataset directory not found: {source_dir}")
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")

    report_rows: list[dict[str, Any]] = []

    for question_file in sorted(source_dir.glob("q*.json")):
        qid = question_file.stem
        run_dir = runs_dir / qid

        row: dict[str, Any] = {
            "question": qid,
            "question_file": str(question_file.relative_to(repo_root)),
            "run_dir": str(run_dir.relative_to(repo_root)),
            "status": "",
            "expected_rows": 0,
            "produced_rows": 0,
            "expected_width": 0,
            "produced_width": 0,
            "projection_columns": None,
            "match_mode": None,
            "final_result_file": None,
            "failure_reason": None,
            "error": None,
        }

        if not run_dir.exists():
            row["status"] = "missing_run_dir"
            row["error"] = "Run directory not found"
            report_rows.append(row)
            continue

        final_result_file = _latest_final_result(run_dir)
        if final_result_file is None:
            row["status"] = "missing_final_result"
            row["error"] = "No final_result.json found"
            report_rows.append(row)
            continue

        row["final_result_file"] = str(final_result_file.relative_to(repo_root))

        try:
            expected_rows = _parse_expected_answer(question_file)
            produced_columns, produced_rows = _parse_produced_result(final_result_file)
            cmp = _compare(expected_rows, produced_columns, produced_rows)

            row["status"] = cmp.status
            row["expected_rows"] = cmp.expected_rows
            row["produced_rows"] = cmp.produced_rows
            row["expected_width"] = cmp.expected_width
            row["produced_width"] = cmp.produced_width
            row["projection_columns"] = cmp.projection_columns
            row["match_mode"] = cmp.match_mode
            row["failure_reason"] = cmp.failure_reason
        except Exception as exc:  # pylint: disable=broad-except
            row["status"] = "analysis_error"
            row["error"] = str(exc)

        report_rows.append(row)

    summary: dict[str, int] = {}
    for row in report_rows:
        status = str(row["status"])
        summary[status] = summary.get(status, 0) + 1

    output_path = runs_dir / "analysis_report.json"
    payload = {
        "dataset": dataset,
        "total": len(report_rows),
        "summary": summary,
        "rows": report_rows,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Dataset: {dataset}")
    print(f"Total questions analyzed: {len(report_rows)}")
    print("Summary:")
    for status in sorted(summary):
        print(f"  - {status}: {summary[status]}")
    print(f"Wrote report: {output_path}")

    return 0


def run_all() -> int:
    for dataset in ["dataset", "dataset_vertical"]:
        print("=" * 72)
        try:
            run(dataset)
        except FileNotFoundError as exc:
            print(f"Dataset: {dataset}")
            print(f"Skipped: {exc}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Bird run outputs against expected answers")
    parser.add_argument(
        "--dataset",
        choices=["dataset", "dataset_vertical", "all"],
        default="dataset",
        help="Dataset split to analyze (default: dataset)",
    )
    args = parser.parse_args()

    if args.dataset == "all":
        return run_all()

    return run(dataset=args.dataset)


if __name__ == "__main__":
    raise SystemExit(main())

"""Cross-query aggregation for selection strategy evaluation."""

import json
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

from qcp_adaptive_termination.models import SelectionStrategyType
from qcp_adaptive_termination.models.query_plan import (
    MultiAgentQueryPlanStep,
    QueryPlan,
)
from scripts.selection_strategy_evaluation.metrics import EvaluationCurve
from scripts.selection_strategy_evaluation.summary import StepSummary, write_summary


def aggregate(results_dir: Path, input_dir: Path) -> Path:
    """Aggregate latest per-query curves into all-step and no-find summaries.

    Summaries are written to the top level of ``results_dir`` (next to the
    per-query subfolders), not into a nested ``aggregate/`` directory.
    """
    aggregate_dir = results_dir
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    step_summaries: list[StepSummary] = []
    status_rows: list[dict[str, Any]] = []
    known_strategies = {str(strategy) for strategy in SelectionStrategyType}

    for input_file in sorted(input_dir.glob("*.json")):
        query_slug = input_file.stem
        run_dir = results_dir / query_slug
        if not run_dir.is_dir():
            status_rows.append({"query": query_slug, "status": "missing_output", "details": "no run directory"})
            continue

        summary_path = run_dir / "summary_all_per_step_auc.csv"
        if not summary_path.is_file():
            status_rows.append({"query": query_slug, "status": "missing_summary", "details": str(summary_path)})
            continue

        with input_file.open(encoding="utf-8") as stream:
            raw_input = json.load(stream)
        raw_plan = (raw_input.get("extra") or {}).get("query_plan")
        expected_steps = (
            {step.id for step in QueryPlan.model_validate(raw_plan).steps if isinstance(step, MultiAgentQueryPlanStep)}
            if raw_plan is not None
            else set()
        )
        summary_frame = pd.read_csv(summary_path)
        summary_by_step = {
            str(step_id): rows.to_dict("records") for step_id, rows in summary_frame.groupby("step", sort=False)
        }

        logs = ""
        for log_name in ("stderr.log", "stdout.log"):
            log_path = run_dir / log_name
            if log_path.is_file():
                logs += log_path.read_text(errors="replace") + "\n"

        failures: dict[tuple[str, str], str] = {}
        for match in re.finditer(r"Strategy (s\d+)/(\w+): no agents queried", logs):
            failures.setdefault((match.group(1), match.group(2)), "no agents queried")
        for match in re.finditer(r"Strategy (s\d+)/(\w+) failed: (.+)", logs):
            failures[(match.group(1), match.group(2))] = match.group(3).strip()

        failure_path = run_dir / "summary_all_failures.csv"
        if failure_path.is_file():
            for row in pd.read_csv(failure_path).to_dict("records"):
                failures[(str(row["step"]), str(row["strategy"]))] = str(row["error"])

        missing_steps = expected_steps - set(summary_by_step)
        query_failure_count = 0
        missing_curves: list[str] = []
        for step_id, summary_rows in summary_by_step.items():
            curve_path = run_dir / f"{query_slug}_{step_id}.csv"
            if not curve_path.is_file():
                missing_curves.append(step_id)
                continue

            metadata_by_strategy = {str(row["strategy"]): row for row in summary_rows}
            curve_frame = pd.read_csv(curve_path)
            curves: dict[str, EvaluationCurve] = {}
            oracle_auc: float | None = None
            oracle_agents = 0
            final_row_count = 0

            for strategy, rows in curve_frame.groupby("strategy", sort=False):
                strategy = str(strategy)
                rows = rows.sort_values("agents_fraction")
                x = [float(value) for value in rows["agents_fraction"]]
                y = [float(value) for value in rows["coverage"]]
                auc = sum(y[1:]) / max(1, len(y) - 1)

                if strategy == "oracle":
                    oracle_auc = auc
                    oracle_agents = int(metadata_by_strategy.get("oracle", {}).get("agents_total", len(y) - 1))
                    final_row_count = int(metadata_by_strategy.get("oracle", {}).get("final_row_count", 0))
                    curves[strategy] = EvaluationCurve(x=x, y=y, auc=auc)
                    continue

                error = failures.get((step_id, strategy))
                if error is not None:
                    query_failure_count += 1
                    continue
                curves[strategy] = EvaluationCurve(x=x, y=y, auc=auc)

            for strategy in known_strategies - curves.keys() - {"oracle"}:
                if (step_id, strategy) not in failures:
                    failures[(step_id, strategy)] = "missing strategy curve"
                    query_failure_count += 1

            if oracle_auc is None:
                missing_curves.append(f"{step_id}:oracle")
                continue

            step_summaries.append(
                StepSummary(
                    query_slug=query_slug,
                    step_id=step_id,
                    agents_total=oracle_agents,
                    oracle_auc=oracle_auc,
                    final_row_count=final_row_count,
                    curves=curves,
                    failures=[
                        (strategy, error)
                        for (failed_step, strategy), error in failures.items()
                        if failed_step == step_id
                    ],
                )
            )

        status = "complete_with_failures" if query_failure_count else "complete"
        details = []
        if missing_steps:
            status = "partial"
            details.append(f"missing steps: {','.join(sorted(missing_steps))}")
        if missing_curves:
            status = "partial"
            details.append(f"missing curves: {','.join(missing_curves)}")
        status_rows.append(
            {
                "query": query_slug,
                "status": status,
                "expected_steps": len(expected_steps),
                "completed_steps": len(summary_by_step),
                "failed_strategies": query_failure_count,
                "details": "; ".join(details),
            }
        )

    write_summary(step_summaries, aggregate_dir, "summary_all_queries", " — all queries and steps")
    non_find_steps = [summary for summary in step_summaries if not summary.is_find_step]
    write_summary(non_find_steps, aggregate_dir, "summary_no_find_queries", " — all queries, excluding find steps")
    pd.DataFrame(status_rows).to_csv(aggregate_dir / "aggregate_run_status.csv", index=False)
    logging.info(
        "Aggregated %d/%d input queries to %s", len(status_rows), len(list(input_dir.glob("*.json"))), aggregate_dir
    )
    return aggregate_dir

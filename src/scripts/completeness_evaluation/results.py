"""Result serialization for completeness evaluation."""

import json
import logging
from pathlib import Path

import pandas as pd

from qcp_adaptive_termination.models import AggregationType
from scripts.completeness_evaluation.models import COVERAGE_TARGETS, StepResult
from scripts.completeness_evaluation.plotting import save_step_plot

_TARGET_LABELS: list[str] = [f"{int(t * 100)}" for t in COVERAGE_TARGETS] + ["budget"]

_EARLY_STOP_COLUMNS: list[str] = (
    [f"earliest_stop_agents_{label}" for label in _TARGET_LABELS]
    + [f"agents_saved_{label}" for label in _TARGET_LABELS]
    + ["earliest_scalar_stop_agents"]
)

_OBSERVED_AGGREGATION_COLUMNS: list[str] = [f"observed_agg_{agg.value}" for agg in AggregationType]

# CSV columns: one row per completeness check (per queried agent).
_CSV_COLUMNS = [
    "step_id",
    "step_sql",
    "compounded",
    "compound_source",
    "agents_total",
    "agents_queried",
    "agents_queried_fraction",
    "step_budget_t",
    "decision",
    "stop",
    "stop_reason",
    "estimator_decision",
    "llm_decision",
    "llm_status",
    "long_list",
    "estimator_active",
    "sample_coverage",
    "aggregation",
    "results_count",
    "unique_results_count",
    "true_count",
    "true_value_count",
    "true_value_avg",
    "true_value_sum",
    "true_value_min",
    "true_value_max",
    "true_value_median",
    "true_value",
    "e2e_true_count",
    "e2e_true_value",
    "observed_count",
    "observed_value",
    *_OBSERVED_AGGREGATION_COLUMNS,
    "estimated_count",
    "estimated_count_lower",
    "estimated_count_upper",
    "estimated_value",
    "estimated_value_lower",
    "estimated_value_upper",
    "c_worst",
    "c_middle",
    "c_best",
    "observed_error",
    "estimated_error",
    "e2e_observed_error",
    "predicted_remaining",
    "realized_unseen",
    "coverage_achieved",
    "e2e_coverage_achieved",
    "produced_scalar",
    "produced_value_source",
    "produced_rel_err",
    "e2e_produced_rel_err",
    "produced_precision",
    "produced_recall",
    "produced_used_estimator",
    "result",
    *_EARLY_STOP_COLUMNS,
]


def _step_level(result: StepResult) -> dict:
    """Build the per-step (query-level) row fragment."""
    gt = result.ground_truth
    return {
        "step_id": result.step_id,
        "step_sql": result.step_sql,
        "compounded": result.compounded,
        "compound_source": result.compound_source,
        "agents_total": result.agents_total,
        "step_budget_t": result.step_budget_t,
        "true_value_count": gt.value_count,
        "true_value_avg": gt.value_avg,
        "true_value_sum": gt.value_sum,
        "true_value_min": gt.value_min,
        "true_value_max": gt.value_max,
        "true_value_median": gt.value_median,
        "produced_scalar": result.produced_scalar,
        "produced_value_source": result.produced_value_source,
        "produced_rel_err": result.produced_rel_err,
        "e2e_produced_rel_err": result.e2e_produced_rel_err,
        "produced_precision": result.produced_precision,
        "produced_recall": result.produced_recall,
        "produced_used_estimator": result.produced_used_estimator,
        "result": result.produced_result,
    }


def save_step_results(result: StepResult, query_slug: str, output_dir: Path):
    """Write one step's evaluator CSV, result txt, and plot."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{query_slug}_{result.step_id}"

    rows: list[dict] = []
    for b in result.checks:
        row = {
            **_step_level(result),
            **{
                "agents_queried": b.cumulative_agents,
                "agents_queried_fraction": b.agents_queried_fraction,
                "decision": b.decision,
                "stop": b.stop,
                "stop_reason": b.decision,
                "estimator_decision": b.estimator_decision,
                "llm_decision": b.llm_decision or "not_called",
                "llm_status": b.llm_status,
                "long_list": b.long_list,
                "estimator_active": b.estimator_active,
                "sample_coverage": b.sample_coverage,
                "aggregation": b.aggregation,
                "results_count": b.results_count,
                "unique_results_count": b.unique_results_count,
                "true_count": b.true_count,
                "true_value": b.true_value,
                "e2e_true_count": b.e2e_true_count,
                "e2e_true_value": b.e2e_true_value,
                "observed_count": b.observed_count,
                "observed_value": b.observed_value,
                **{f"observed_agg_{agg.value}": b.observed_by_aggregation.get(agg.value) for agg in AggregationType},
                "estimated_count": b.estimated_count,
                "estimated_count_lower": b.estimated_count_lower,
                "estimated_count_upper": b.estimated_count_upper,
                "estimated_value": b.estimated_value,
                "estimated_value_lower": b.estimated_value_lower,
                "estimated_value_upper": b.estimated_value_upper,
                "c_worst": b.c_worst,
                "c_middle": b.c_middle,
                "c_best": b.c_best,
                "observed_error": b.observed_error,
                "estimated_error": b.estimated_error,
                "e2e_observed_error": b.e2e_observed_error,
                "predicted_remaining": b.predicted_remaining,
                "realized_unseen": b.realized_unseen,
                "coverage_achieved": b.coverage_achieved,
                "e2e_coverage_achieved": b.e2e_coverage_achieved,
                **{f"earliest_stop_agents_{label}": b.earliest_stop.get(label) for label in _TARGET_LABELS},
                **{f"agents_saved_{label}": b.agents_saved.get(label) for label in _TARGET_LABELS},
                "earliest_scalar_stop_agents": b.earliest_scalar_stop_agents,
            },
        }
        rows.append(row)

    pd.DataFrame(rows, columns=_CSV_COLUMNS).to_csv(output_dir / f"{prefix}_evaluator.csv", index=False)

    _save_result_txt(result, query_slug, output_dir)

    save_step_plot(result, prefix, output_dir)
    logging.info("Saved evaluator results to %s", output_dir)


def _save_result_txt(result: StepResult, query_slug: str, output_dir: Path):
    """Write the result synthesis as a human-readable txt."""
    if result.produced_result is not None:
        try:
            synthesized = json.loads(result.produced_result)
            text = json.dumps(synthesized, indent=2, ensure_ascii=False)
        except Exception:
            text = result.produced_result
    else:
        text = "no result"
    (output_dir / f"{query_slug}_{result.step_id}_result.txt").write_text(
        f"{result.step_id}\n{text}\n", encoding="utf-8"
    )

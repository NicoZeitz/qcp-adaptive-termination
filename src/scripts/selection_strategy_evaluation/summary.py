"""Summary aggregation for selection strategy evaluation."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from scripts.selection_strategy_evaluation.metrics import (
    EvaluationCurve,
    compute_relative_auc,
)
from scripts.selection_strategy_evaluation.plotting import plot_summary_curves


@dataclass(slots=True)
class StepSummary:
    """Benchmark outcome for one step, used for summary aggregation."""

    query_slug: str
    step_id: str
    agents_total: int
    oracle_auc: float
    final_row_count: int
    curves: dict[str, EvaluationCurve] = field(default_factory=dict)
    failures: list[tuple[str, str]] = field(default_factory=list)

    @property
    def is_find_step(self) -> bool:
        """Find step: full result set is a single row (plus oracle AUC = 1)."""
        return self.final_row_count == 1


def average_curves(summaries: list[StepSummary]) -> dict[str, list[tuple[float, float]]]:
    """Average coverage over steps at a common fraction grid, per strategy (incl. oracle)."""
    if not summaries:
        return {}
    grid = [i / 100 for i in range(101)]
    all_names: list[str] = ["oracle"]
    for summary in summaries:
        for name in summary.curves:
            if name not in all_names:
                all_names.append(name)

    averaged: dict[str, list[tuple[float, float]]] = {}
    for name in all_names:
        values: list[list[float]] = []
        for summary in summaries:
            curve = summary.curves.get(name)
            if curve is None:
                continue
            series = pd.Series(curve.y, index=curve.x).reindex(grid, method="ffill").fillna(0.0)
            values.append(series.tolist())
        if not values:
            continue
        means = [sum(col) / len(values) for col in zip(*values, strict=True)]
        averaged[name] = list(zip(grid, means, strict=True))
    return averaged


def write_summary(
    summaries: list[StepSummary],
    output_dir: Path,
    filename_prefix: str,
    title_suffix: str,
) -> None:
    """Write summary CSVs and averaged fraction plots for a set of step summaries."""
    if not summaries:
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    auc_rows = []
    failure_rows = []
    for summary in summaries:
        for strategy, curve in summary.curves.items():
            auc_rows.append(
                {
                    "query": summary.query_slug,
                    "step": summary.step_id,
                    "strategy": strategy,
                    "auc": curve.auc,
                    "relative_auc": compute_relative_auc(curve.auc, summary.oracle_auc),
                    "oracle_auc": summary.oracle_auc,
                    "agents_total": summary.agents_total,
                    "final_row_count": summary.final_row_count,
                }
            )
        failure_rows.extend(
            {"query": summary.query_slug, "step": summary.step_id, "strategy": strategy, "error": error}
            for strategy, error in summary.failures
        )
    per_step_auc = pd.DataFrame(auc_rows)
    per_step_auc.to_csv(output_dir / f"{filename_prefix}_per_step_auc.csv", index=False)
    pd.DataFrame(failure_rows, columns=["query", "step", "strategy", "error"]).to_csv(
        output_dir / f"{filename_prefix}_failures.csv", index=False
    )

    mean_auc = (
        per_step_auc.groupby("strategy")["auc"]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    mean_auc.to_csv(output_dir / f"{filename_prefix}_mean_auc.csv", index=False)

    averaged = average_curves(summaries)
    curve_rows = [
        {"strategy": name, "agents_fraction": x, "coverage": y} for name, points in averaged.items() for x, y in points
    ]
    pd.DataFrame(curve_rows).to_csv(output_dir / f"{filename_prefix}.csv", index=False)

    plot_summary_curves(averaged, mean_auc, output_dir, filename_prefix, title_suffix)
    logging.info("Summary saved to %s (%s)", output_dir, filename_prefix)

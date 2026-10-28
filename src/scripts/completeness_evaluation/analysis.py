"""Cross-variant analysis of completeness-evaluation runs (optimal stops, deviations, cost)."""

import logging
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from qcp_adaptive_termination.models import StopReason

_PLOT_EXTS = ("png", "pdf")

# Accuracy thresholds (in percent) at which optimal stops are reported; ``budget`` uses the step threshold.
_REPORTED_TARGETS = ("100", "95", "budget")

# Accuracy metric -> description; every metric is in [0, 1] with 1 meaning exact.
METRICS: dict[str, str] = {
    "rows": "fraction of true rows observed",
    "observed": "1 - relative error of the observed aggregate",
    "estimated_count": "1 - relative error of the estimated count",
    "estimated_value": "1 - relative error of the estimated aggregate",
    "best_estimate": "max of estimated_count and estimated_value accuracy",
}

_STOPPING_LABELS = {
    (False, False, "prevent"): "none",
    (False, False, "estimator_reasoning"): "none",
    (True, False, "prevent"): "estimator",
    (True, False, "estimator_reasoning"): "estimator",
    (False, True, "prevent"): "llm-prevent",
    (False, True, "estimator_reasoning"): "llm-estimator-reasoning",
    (True, True, "prevent"): "estimator-llm-prevent",
    (True, True, "estimator_reasoning"): "estimator-llm-estimator-reasoning",
}


@dataclass(slots=True)
class _Run:
    """One completeness-evaluation run (one ordering and stopping variant)."""

    path: Path
    """Run output directory"""
    ordering: str
    """Agent selection strategy"""
    stopping: str
    """Stopping variant label"""
    checks: pd.DataFrame
    """All per-check rows of the run"""
    usage: pd.DataFrame | None
    """Per-query usage summary"""
    usage_by_role: pd.DataFrame | None
    """Usage summed per call type and role"""


def _parse_arguments() -> Namespace:
    """Parse CLI arguments."""
    parser = ArgumentParser(description="Cross-variant analysis of completeness-evaluation runs")
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing one sub-directory per run (ordering x stopping variant) of one dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to write the analysis (default: <results_dir>/analysis)",
    )
    return parser.parse_args()


def _load_runs(results_dir: Path) -> list[_Run]:
    """Load the latest run per (ordering, stopping) variant."""
    runs: dict[tuple[str, str], _Run] = {}
    for config_path in sorted(results_dir.glob("*/config_resolved.yaml")):
        run_dir = config_path.parent
        checks_path = run_dir / "evaluator_summary_all_queries.csv"
        if not checks_path.exists():
            logging.warning("Skipping %s: no evaluator_summary_all_queries.csv", run_dir)
            continue
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        completeness = config["completeness"]
        lwk = completeness["llm_world_knowledge"]
        stopping = _STOPPING_LABELS[
            (bool(completeness["estimator"]["enabled"]), bool(lwk["enabled"]), lwk["long_list_behavior"])
        ]
        ordering = config["selection"]["strategy"]
        usage_path = run_dir / "usage_summary.csv"
        role_path = run_dir / "usage_by_role.csv"
        runs[(ordering, stopping)] = _Run(
            path=run_dir,
            ordering=ordering,
            stopping=stopping,
            checks=pd.read_csv(checks_path),
            usage=pd.read_csv(usage_path) if usage_path.exists() else None,
            usage_by_role=pd.read_csv(role_path) if role_path.exists() else None,
        )
    return list(runs.values())


def _accuracies(checks: pd.DataFrame) -> pd.DataFrame:
    """Add one accuracy column per metric to the per-check rows."""

    def accuracy(estimate: str, truth: str) -> np.ndarray:
        """1 - relative error between two columns (NaN when undefined)."""
        est = np.asarray(pd.to_numeric(checks[estimate], errors="coerce"), dtype=float)
        true = np.asarray(pd.to_numeric(checks[truth], errors="coerce"), dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(true != 0, 1 - np.abs(est - true) / np.abs(true), np.nan)

    observed_error = np.asarray(pd.to_numeric(checks["observed_error"], errors="coerce"), dtype=float)
    estimated_error = np.asarray(pd.to_numeric(checks["estimated_error"], errors="coerce"), dtype=float)
    count_acc = accuracy("estimated_count", "true_count")
    value_acc = 1 - estimated_error
    return checks.assign(
        acc_rows=np.asarray(pd.to_numeric(checks["coverage_achieved"], errors="coerce"), dtype=float),
        acc_observed=1 - observed_error,
        acc_estimated_count=count_acc,
        acc_estimated_value=value_acc,
        acc_best_estimate=np.fmax(count_acc, value_acc),
    )


def _targets(budget: float) -> dict[str, float]:
    """Accuracy targets (0-100% plus the step budget) keyed by label."""
    targets = {str(k): k / 100 for k in range(0, 101)}
    if np.isfinite(budget):
        targets["budget"] = budget
    return targets


def _first_reaching(trajectory: pd.DataFrame, metric: str, target: float) -> float:
    """First agents_queried at which the metric reaches the target (NaN if never)."""
    values = np.asarray(trajectory[f"acc_{metric}"], dtype=float)
    agents = np.asarray(trajectory["agents_queried"], dtype=float)
    hit = np.flatnonzero(values >= target - 1e-9)
    return float(agents[hit[0]]) if hit.size else np.nan


def _num(frame: pd.DataFrame, column: str) -> np.ndarray:
    """Column as a float array (non-numeric values become NaN)."""
    return np.asarray(pd.to_numeric(frame[column], errors="coerce"), dtype=float)


def _groups(frame: pd.DataFrame, columns: list[str]) -> list[tuple[tuple[str, ...], pd.DataFrame]]:
    """Split a frame into (key, group) pairs in order of appearance."""
    return [(cast(tuple[str, ...], key), group) for key, group in frame.groupby(columns, sort=False)]


def _optimal_stops(baselines: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Optimal stop (earliest agent reaching each accuracy target) per ordering, step and metric."""
    rows: list[dict] = []
    for ordering, checks in baselines.items():
        for (query, step_id), trajectory in _groups(checks, ["query", "step_id"]):
            trajectory = trajectory.sort_values("agents_queried")
            first = trajectory.to_dict("records")[0]
            budget = float(_num(trajectory, "step_budget_t")[0])
            row = {
                "ordering": ordering,
                "query": query,
                "step_id": step_id,
                "step_sql": first["step_sql"],
                "agents_total": int(first["agents_total"]),
                "step_budget_t": budget,
            }
            for metric in METRICS:
                for label, target in _targets(budget).items():
                    row[f"opt_{metric}_{label}"] = _first_reaching(trajectory, metric, target)
            rows.append(row)
    return pd.DataFrame(rows)


def _variant_steps(run: _Run, optimal: pd.DataFrame) -> pd.DataFrame:
    """Per-step stop, accuracy at stop and deviation from the optimal stop for one run."""
    rows: list[dict] = []
    opt_index = {(r["ordering"], r["query"], r["step_id"]): r for r in optimal.to_dict("records")}
    for (query, step_id), trajectory in _groups(run.checks, ["query", "step_id"]):
        trajectory = trajectory.sort_values("agents_queried")
        last = trajectory.to_dict("records")[-1]
        stop = float(last["agents_queried"])
        budget = float(_num(trajectory, "step_budget_t")[0])
        baseline = opt_index.get((run.ordering, query, step_id))
        comparable = baseline is not None and baseline["step_sql"] == last["step_sql"]
        row = {
            "ordering": run.ordering,
            "stopping": run.stopping,
            "query": query,
            "step_id": step_id,
            "aggregation": last["aggregation"],
            "agents_total": int(last["agents_total"]),
            "agents_at_stop": stop,
            "agents_saved": int(last["agents_total"]) - stop,
            "stop_reason": last["decision"],
            "llm_status_at_stop": last.get("llm_status"),
            "long_list_at_stop": last.get("long_list"),
            "compounded": last.get("compounded"),
            "comparable_to_baseline": comparable,
            "produced_rel_err": last.get("produced_rel_err"),
            "e2e_produced_rel_err": last.get("e2e_produced_rel_err"),
            "produced_recall": last.get("produced_recall"),
            "e2e_coverage_at_stop": last.get("e2e_coverage_achieved"),
        }
        for metric in METRICS:
            row[f"acc_{metric}_at_stop"] = float(last[f"acc_{metric}"])
            for label in _REPORTED_TARGETS:
                target = budget if label == "budget" else int(label) / 100
                if label == "budget" and not np.isfinite(budget):
                    continue
                own = _first_reaching(trajectory, metric, target)
                opt = baseline[f"opt_{metric}_{label}"] if comparable and baseline is not None else own
                row[f"opt_{metric}_{label}"] = opt
                row[f"deviation_{metric}_{label}"] = stop - opt if np.isfinite(opt) else np.nan
                # Stopping before the target was reached is an early (error) stop, even without a baseline.
                row[f"category_{metric}_{label}"] = (
                    "early" if not np.isfinite(own) else "optimal" if stop == opt else "late" if stop > opt else "early"
                )
        rows.append(row)
    return pd.DataFrame(rows)


def _variant_summary(steps: pd.DataFrame, runs: list[_Run]) -> pd.DataFrame:
    """Aggregate deviations, accuracy, saved agents and cost per variant."""
    rows: list[dict] = []
    for (ordering, stopping), group in _groups(steps, ["ordering", "stopping"]):
        agents_at_stop = _num(group, "agents_at_stop")
        agents_total = _num(group, "agents_total")
        row: dict = {
            "ordering": ordering,
            "stopping": stopping,
            "steps": len(group),
            "early_stops": int((agents_at_stop < agents_total).sum()),
            "agents_queried": float(agents_at_stop.sum()),
            "agents_total": float(agents_total.sum()),
            "agents_saved": float(_num(group, "agents_saved").sum()),
            "mean_produced_rel_err": _nanmean(_num(group, "produced_rel_err")),
            "mean_e2e_produced_rel_err": _nanmean(_num(group, "e2e_produced_rel_err")),
        }
        for reason, count in group["stop_reason"].value_counts().items():
            row[f"stop_reason_{reason}"] = int(count)
        for metric in METRICS:
            row[f"mean_acc_{metric}_at_stop"] = _nanmean(_num(group, f"acc_{metric}_at_stop"))
            for label in _REPORTED_TARGETS:
                if f"deviation_{metric}_{label}" not in group.columns:
                    continue
                deviation = _num(group, f"deviation_{metric}_{label}")
                category = np.asarray(group[f"category_{metric}_{label}"], dtype=object)
                row[f"mean_deviation_{metric}_{label}"] = _nanmean(deviation)
                row[f"median_deviation_{metric}_{label}"] = (
                    float(np.nanmedian(deviation)) if np.isfinite(deviation).any() else np.nan
                )
                for name in ("early", "optimal", "late"):
                    row[f"{name}_{metric}_{label}"] = int((category == name).sum())
        run = next((r for r in runs if r.ordering == ordering and r.stopping == stopping), None)
        if run is not None and run.usage is not None:
            for column in (
                "cost",
                "cost_uncached",
                "input_tokens",
                "cached_input_tokens",
                "output_tokens",
                "reasoning_tokens",
            ):
                if column in run.usage.columns:
                    row[f"total_{column}"] = float(_num(run.usage, column).sum())
            row["total_llm_runs"] = float(_num(run.usage, "llm_runs").sum())
            row["total_embedding_calls"] = float(_num(run.usage, "embedding_calls").sum())
        if run is not None and run.usage_by_role is not None:
            for record in run.usage_by_role.to_dict("records"):
                role = record.get("role") if isinstance(record.get("role"), str) else record.get("type")
                row[f"calls_{role}"] = record["calls"]
                row[f"cost_{role}"] = record["cost"]
        rows.append(row)
    return pd.DataFrame(rows)


def _nanmean(values: np.ndarray) -> float:
    """Mean over finite values (NaN when there are none)."""
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else np.nan


def _accuracy_curves(steps: pd.DataFrame) -> pd.DataFrame:
    """Fraction of steps whose accuracy at stop reaches k, for k = 0..100%, per variant and metric."""
    rows: list[dict] = []
    for (ordering, stopping), group in _groups(steps, ["ordering", "stopping"]):
        for metric in METRICS:
            values = np.asarray(group[f"acc_{metric}_at_stop"], dtype=float)
            values = values[np.isfinite(values)]
            if not values.size:
                continue
            for k in range(0, 101):
                rows.append(
                    {
                        "ordering": ordering,
                        "stopping": stopping,
                        "metric": metric,
                        "k": k,
                        "fraction_reached": float((values >= k / 100 - 1e-9).mean()),
                    }
                )
    return pd.DataFrame(rows)


def _failures(steps: pd.DataFrame, runs: list[_Run]) -> pd.DataFrame:
    """Early stops (before all true rows were seen) with pointers to the deciding transcript."""
    early = steps.loc[steps["category_rows_100"] == "early"].copy()
    run_paths = {(r.ordering, r.stopping): r.path for r in runs}
    early["transcript"] = [
        str(
            run_paths[(row.ordering, row.stopping)]
            / str(row.query)
            / f"COMPLETENESS_step_{row.step_id}_agents_queried_{int(row.agents_at_stop)}.md"
        )
        if row.stop_reason in (StopReason.LLM.value, StopReason.LLM_WITH_ESTIMATOR.value)
        else ""
        for row in early.itertuples()
    ]
    return early


def _save(fig, output_dir: Path, stem: str):
    """Save a figure in all configured formats."""
    fig.tight_layout()
    for ext in _PLOT_EXTS:
        fig.savefig(output_dir / f"{stem}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot(steps: pd.DataFrame, summary: pd.DataFrame, curves: pd.DataFrame, output_dir: Path):
    """Deviation boxplots, accuracy curves and cost-vs-accuracy scatter across variants."""
    labels = [f"{o}\n{s}" for o, s in zip(summary["ordering"], summary["stopping"], strict=True)]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))
    data = [
        np.asarray(steps.loc[(steps["ordering"] == o) & (steps["stopping"] == s), "deviation_rows_100"], dtype=float)
        for o, s in zip(summary["ordering"], summary["stopping"], strict=True)
    ]
    ax.boxplot([d[np.isfinite(d)] for d in data], tick_labels=labels)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("stop - optimal stop (agents, 100% rows)")
    _save(fig, output_dir, "deviation_rows_100")

    for metric in METRICS:
        sub = curves.loc[curves["metric"] == metric]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 5))
        for (ordering, stopping), group in sub.groupby(["ordering", "stopping"], sort=False):
            ax.plot(group["k"], group["fraction_reached"], label=f"{ordering} / {stopping}")
        ax.set_xlabel("accuracy k (%)")
        ax.set_ylabel("fraction of steps with accuracy >= k at stop")
        ax.set_title(METRICS[metric])
        ax.legend(fontsize=7)
        _save(fig, output_dir, f"accuracy_curve_{metric}")

    if "total_cost" in summary.columns:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(summary["total_cost"], summary["mean_acc_rows_at_stop"])
        for label, x, y in zip(labels, summary["total_cost"], summary["mean_acc_rows_at_stop"], strict=True):
            ax.annotate(label.replace("\n", " / "), (x, y), fontsize=7)
        ax.set_xlabel("total cost (USD)")
        ax.set_ylabel("mean fraction of rows observed at stop")
        _save(fig, output_dir, "cost_vs_rows_accuracy")


def analyze(results_dir: Path, output_dir: Path) -> Path:
    """Run the cross-variant analysis for one dataset's runs.

    Args:
        results_dir: directory with one sub-directory per run
        output_dir: directory to write the analysis artifacts into

    Returns:
        (Path): the output directory
    """
    runs = _load_runs(results_dir)
    if not runs:
        raise RuntimeError(f"No completeness runs found under {results_dir}")
    for run in runs:
        run.checks = _accuracies(run.checks)
    baselines = {r.ordering: r.checks for r in runs if r.stopping == "none"}
    output_dir.mkdir(parents=True, exist_ok=True)

    optimal = _optimal_stops(baselines)
    steps = pd.concat([_variant_steps(r, optimal) for r in runs], ignore_index=True)
    summary = _variant_summary(steps, runs)
    curves = _accuracy_curves(steps)
    failures = _failures(steps, runs)

    optimal.to_csv(output_dir / "optimal_stops.csv", index=False)
    steps.to_csv(output_dir / "variant_steps.csv", index=False)
    summary.to_csv(output_dir / "variant_summary.csv", index=False)
    curves.to_csv(output_dir / "accuracy_curves.csv", index=False)
    failures.to_csv(output_dir / "early_stop_failures.csv", index=False)
    _plot(steps, summary, curves, output_dir)
    print(f"[ANALYSIS] {len(runs)} runs, {len(steps)} step results -> {output_dir}", flush=True)
    return output_dir


def main():
    """Parse CLI args and run the analysis."""
    args = _parse_arguments()
    results_dir = Path(args.results_dir)
    analyze(results_dir, Path(args.output_dir) if args.output_dir else results_dir / "analysis")


if __name__ == "__main__":
    main()

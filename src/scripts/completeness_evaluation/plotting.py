"""Plotting for completeness evaluation (per-step and cross-query plots)."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.completeness_evaluation.models import StepResult

_PLOT_EXTS = ("png", "pdf", "svg")


def _save(fig, output_dir: Path, stem: str):
    """Save a figure in all configured formats."""
    fig.tight_layout()
    for ext in _PLOT_EXTS:
        fig.savefig(output_dir / f"{stem}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _series(rows, key):
    """Return a float numpy array for ``key`` with missing values as NaN."""
    return np.array(
        [float(v) if (v := getattr(b, key, None)) is not None else np.nan for b in rows],
        dtype=float,
    )


def _plot_series(ax, xs, key, rows, **kwargs):
    """Plot ``key`` over ``xs``, skipping all-NaN series."""
    series = _series(rows, key)
    if np.isfinite(series).any():
        ax.plot(xs, series, **kwargs)


def _fill_bounds(ax, xs, lo_key, hi_key, rows, **kwargs):
    """Fill between two bound series, skipping when all values are NaN."""
    lo = _series(rows, lo_key)
    hi = _series(rows, hi_key)
    if np.isfinite(lo).any() and np.isfinite(hi).any():
        ax.fill_between(xs, lo, hi, **kwargs)


def save_step_plot(result: StepResult, prefix: str, output_dir: Path):
    """Rich per-step figure: coverage, count bounds, c_* ratios, and value (if any)."""
    checks = result.checks
    if not checks:
        return
    xs = [b.cumulative_agents for b in checks]
    stop_x = result.agents_queried_at_stop
    gt = result.ground_truth
    last_observed_value = checks[-1].observed_value
    has_value = result.aggregation is not None and last_observed_value is not None

    n_cols = 2 if has_value else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(13 if has_value else 7, 6))
    ax_cov = axes[0] if has_value else axes

    # Coverage
    _plot_series(ax_cov, xs, "coverage_achieved", checks, marker="o", markersize=3, label="coverage")
    _plot_series(ax_cov, xs, "c_worst", checks, marker="s", markersize=3, linestyle="--", label="c_worst")
    _plot_series(ax_cov, xs, "c_middle", checks, marker="^", markersize=3, linestyle="--", label="c_middle")
    _plot_series(ax_cov, xs, "c_best", checks, marker="v", markersize=3, linestyle="--", label="c_best")
    ax_cov.axvline(stop_x, color="red", linestyle="--", alpha=0.7, label=f"stop (agent {stop_x})")
    ax_cov.axhline(1.0, color="green", linestyle=":", alpha=0.5)
    ax_cov.set_xlabel("Agents queried")
    ax_cov.set_ylabel("Coverage / c_*")
    ax_cov.set_ylim(0.0, 1.15)
    ax_cov.set_title(f"Coverage & completeness ratios\n{prefix}")
    ax_cov.set_xticks(range(0, xs[-1] + 1))
    ax_cov.legend(fontsize=8)

    # Counts + estimated bounds
    ax_cnt = ax_cov.twinx()
    _fill_bounds(
        ax_cnt,
        xs,
        "estimated_count_lower",
        "estimated_count_upper",
        checks,
        color="gray",
        alpha=0.15,
        label="N_hat bounds",
    )
    _plot_series(ax_cnt, xs, "observed_count", checks, marker="o", markersize=3, color="navy", label="observed count")
    _plot_series(
        ax_cnt,
        xs,
        "estimated_count",
        checks,
        marker="s",
        markersize=3,
        color="crimson",
        linestyle="--",
        label="est count",
    )
    ax_cnt.axhline(gt.count, color="green", linestyle=":", alpha=0.7, label=f"true count ({gt.count})")
    ax_cnt.set_ylabel("Count")
    ax_cnt.tick_params(axis="y", labelcolor="navy")
    ax_cnt.legend(fontsize=8, loc="lower right")

    if has_value:
        ax_val = axes[1]
        _plot_series(ax_val, xs, "observed_value", checks, marker="o", markersize=3, label="observed value")
        _plot_series(ax_val, xs, "estimated_value", checks, marker="s", markersize=3, linestyle="--", label="est value")
        _fill_bounds(
            ax_val,
            xs,
            "estimated_value_lower",
            "estimated_value_upper",
            checks,
            color="gray",
            alpha=0.15,
            label="est value bounds",
        )
        tv = result.checks[-1].true_value
        if tv is not None:
            ax_val.axhline(tv, color="green", linestyle=":", alpha=0.7, label=f"true value ({tv:.3g})")
        ax_val.axvline(stop_x, color="red", linestyle="--", alpha=0.7)
        agg_label = result.aggregation.value if result.aggregation is not None else "value"
        ax_val.set_xlabel("Agents queried")
        ax_val.set_ylabel(f"Value ({agg_label})")
        ax_val.set_title(f"Value trajectory\n{prefix}")
        ax_val.set_xticks(range(0, xs[-1] + 1))
        ax_val.legend(fontsize=8)

    _save(fig, output_dir, f"{prefix}_evaluator")


def save_aggregate_plots(step_df: pd.DataFrame, output_dir: Path):
    """Cross-query plots from the merged step-level dataframe.

    ``step_df`` carries one row per (query, step) with stop decision, ground
    truth, observed/estimated count at stop, achieved coverage and produced-result
    error. Writes ``aggregate_evaluator_*.{png,pdf,svg}`` into ``output_dir``.
    """
    if step_df.empty:
        logging.warning("No step rows for aggregate plots")
        return

    reason_order = ["all_agents", "estimator", "llm"]
    present_reasons = [r for r in reason_order if (step_df["stop_reason"] == r).any()]
    palette = {"all_agents": "#4c72b0", "estimator": "#dd8452", "llm": "#55a868"}

    # 1. Stop-condition histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = {r: int((step_df["stop_reason"] == r).sum()) for r in present_reasons}
    container = ax.bar(list(counts.keys()), list(counts.values()), color=[palette[r] for r in counts])
    ax.set_title("Stop condition across steps")
    ax.set_xlabel("Stop reason")
    ax.set_ylabel("Number of steps")
    ax.bar_label(container)
    _save(fig, output_dir, "aggregate_evaluator_stop_histogram")

    # 2. Estimate vs truth scatter (count), log-log with identity line
    fig, ax = plt.subplots(figsize=(7, 7))
    est_n = step_df[["count_est_at_stop", "true_count"]].dropna().to_numpy(dtype=float)
    if est_n.size:
        est_arr, true_arr = est_n[:, 0], est_n[:, 1]
        pos = true_arr > 0.0
        true_n = true_arr[pos]
        est_m = est_arr[pos]
        if true_n.size:
            ax.scatter(true_n, est_m, alpha=0.6, s=30)
            lo = float(min(float(true_n.min()), float(est_m.min())) * 0.5)
            hi = float(max(float(true_n.max()), float(est_m.max())) * 2)
            ax.plot([lo, hi], [lo, hi], color="red", linestyle="--", linewidth=1, label="identity")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_xlabel("True distinct count")
            ax.set_ylabel("Estimated count at stop (N_hat)")
            ax.set_title("Count estimate vs truth")
            ax.legend()
    _save(fig, output_dir, "aggregate_evaluator_estimate_vs_truth")

    # 3. Coverage-at-stop vs true distinct count
    fig, ax = plt.subplots(figsize=(7, 5))
    cov = step_df[["coverage_at_stop", "true_count"]].dropna()
    ax.scatter(cov["true_count"], cov["coverage_at_stop"], alpha=0.6, s=30)
    ax.axhline(0.95, color="red", linestyle="--", linewidth=1, label="95% coverage")
    ax.axhline(1.0, color="green", linestyle=":", linewidth=1, label="full")
    ax.set_xscale("log")
    ax.set_xlabel("True distinct count")
    ax.set_ylabel("Observed coverage at stop")
    ax.set_title("Coverage at stop vs true set size")
    ax.legend()
    _save(fig, output_dir, "aggregate_evaluator_coverage_vs_true")

    # 4. Cost vs correctness frontier (agents fraction vs coverage, colored by reason)
    fig, ax = plt.subplots(figsize=(8, 6))
    for reason, color in palette.items():
        sub = step_df[step_df["stop_reason"] == reason]
        if sub.empty:
            continue
        ax.scatter(
            sub["stop_fraction"],
            sub["coverage_at_stop"],
            alpha=0.6,
            s=30,
            color=color,
            label=reason,
        )
    ax.set_xlabel("Agents used (stop_fraction)")
    ax.set_ylabel("Observed coverage at stop")
    ax.set_title("Cost vs correctness at stop")
    ax.legend()
    _save(fig, output_dir, "aggregate_evaluator_cost_vs_correctness")

    # 5. Agents-at-stop distribution by stop reason
    fig, ax = plt.subplots(figsize=(8, 5))
    for reason, color in palette.items():
        sub = step_df[step_df["stop_reason"] == reason]
        if sub.empty:
            continue
        ax.scatter(
            np.full(len(sub), reason),
            sub["agents_queried_at_stop"],
            alpha=0.5,
            s=30,
            color=color,
            label=reason,
        )
    ax.set_title("Agents queried at stop by stop reason")
    ax.set_xlabel("Stop reason")
    ax.set_ylabel("Agents queried at stop")
    ax.legend()
    _save(fig, output_dir, "aggregate_evaluator_agents_at_stop")

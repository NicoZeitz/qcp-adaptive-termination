"""Plotting for selection strategy evaluation (per-step and aggregated)."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from scripts.selection_strategy_evaluation.metrics import EvaluationCurve


def _save_figure(fig: Any, output_dir: Path, prefix: str) -> None:
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"{prefix}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_step_curves(
    results: list[tuple[str, EvaluationCurve]],
    oracle_curve: EvaluationCurve,
    query_slug: str,
    step_id: str,
    output_dir: Path,
) -> None:
    """Plot per-strategy coverage curves for one step (agents-queried x-axis)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        range(len(oracle_curve.x)),
        oracle_curve.y,
        "--",
        color="black",
        label=f"oracle (AUC={oracle_curve.auc:.3f})",
        linewidth=2,
        drawstyle="steps-post",
    )
    for strategy, curve in results:
        ax.plot(
            range(len(curve.x)),
            curve.y,
            label=f"{strategy} (AUC={curve.auc:.3f})",
            drawstyle="steps-post",
        )
    ax.set_xlabel("Agents queried")
    ax.set_ylabel("Fraction of unique rows covered")
    ax.set_title(f"Selection strategy comparison\n{query_slug} / {step_id}")
    ax.legend(fontsize=8)
    x_max = max(len(oracle_curve.x) - 1, max((len(c.x) - 1 for _, c in results), default=0))
    ax.set_xlim(0.0, x_max)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_ylim(0.0, 1.05)
    fig.tight_layout()
    _save_figure(fig, output_dir, f"{query_slug}_{step_id}")


def plot_summary_curves(
    averaged: dict[str, list[tuple[float, float]]],
    mean_auc: pd.DataFrame,
    output_dir: Path,
    filename_prefix: str,
    title_suffix: str,
) -> None:
    """Plot averaged coverage curves across steps (fraction x-axis)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    oracle_points = averaged.get("oracle")
    if oracle_points is not None:
        oracle_mean_auc = mean_auc.loc[mean_auc["strategy"] == "oracle", "mean"].iloc[0]
        ax.plot(
            [x for x, _ in oracle_points],
            [y for _, y in oracle_points],
            "--",
            color="black",
            label=f"oracle (mean AUC={oracle_mean_auc:.3f})",
            linewidth=2,
            drawstyle="steps-post",
        )
    for name, points in averaged.items():
        if name == "oracle":
            continue
        mean_auc_value = mean_auc.loc[mean_auc["strategy"] == name, "mean"].iloc[0]
        ax.plot(
            [x for x, _ in points],
            [y for _, y in points],
            label=f"{name} (mean AUC={mean_auc_value:.3f})",
            drawstyle="steps-post",
        )
    ax.set_xlabel("Fraction of agents queried")
    ax.set_ylabel("Fraction of unique rows covered")
    ax.set_title(f"Selection strategy comparison{title_suffix}")
    ax.legend(fontsize=8)
    ax.set_xlim(0.0, 1.0)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_ylim(0.0, 1.05)
    fig.tight_layout()
    _save_figure(fig, output_dir, filename_prefix)

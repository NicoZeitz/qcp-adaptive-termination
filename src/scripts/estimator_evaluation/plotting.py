"""Plotting for estimator evaluation (per-step coverage/count trajectories)."""

from pathlib import Path

import matplotlib.pyplot as plt

from scripts.estimator_evaluation.models import GroundTruth, StepResult


def save_step_plot(result: StepResult, gt: GroundTruth, prefix: str, output_dir: Path) -> None:
    batches = result.batches
    if not batches:
        return

    xs = [b.cumulative_agents for b in batches]
    stop_x = result.agents_queried_at_stop

    has_value = gt.value_sum is not None

    fig, axes = plt.subplots(1, 2 if has_value else 1, figsize=(13 if has_value else 7, 5))
    ax_cov = axes[0] if has_value else axes

    coverages = [b.coverage_achieved for b in batches]
    ax_cov.plot(xs, coverages, marker="o", markersize=3, label="coverage (observed/true)")
    ax_cov.axvline(stop_x, color="red", linestyle="--", alpha=0.7, label=f"stop (agent {stop_x})")
    ax_cov.axhline(1.0, color="green", linestyle=":", alpha=0.5)
    ax_cov.set_xlabel("Agents queried")
    ax_cov.set_ylabel("Coverage (observed / true count)")
    ax_cov.set_ylim(0.0, 1.1)
    ax_cov.set_title(f"Coverage trajectory\n{prefix}")
    ax_cov.legend(fontsize=8)

    if has_value:
        ax_val = axes[1]
        est_counts = [b.estimated_count for b in batches]
        obs_counts = [b.observed_count for b in batches]
        ax_val.plot(xs, obs_counts, marker="o", markersize=3, label="observed count")
        ax_val.plot(xs, est_counts, marker="s", markersize=3, linestyle="--", label="estimated count")
        ax_val.axhline(gt.count, color="green", linestyle=":", alpha=0.7, label=f"true count ({gt.count})")
        ax_val.axvline(stop_x, color="red", linestyle="--", alpha=0.7, label=f"stop (agent {stop_x})")
        ax_val.set_xlabel("Agents queried")
        ax_val.set_ylabel("Count")
        ax_val.set_title(f"Count estimation trajectory\n{prefix}")
        ax_val.legend(fontsize=8)

    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"{prefix}_evaluator.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)

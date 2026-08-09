#!/usr/bin/env python3
"""Plot observed and estimated accuracy over agents queried for each estimation scenario.

For every scenario directory under the data dir, produces:
  - accuracy.{png,svg,pdf}  — per-scenario plot
  - accuracy_combined_<group>.{png,svg,pdf} — one grid figure per dataset group
    with one subplot per variant, saved into the data dir

Each plot shows:
  - Observed accuracy  max(0, 1 - |observed - gt| / gt) * 100  — blue line
  - Estimated accuracy max(0, 1 - |estimated - gt| / gt) * 100 — orange line
  - 100% reference line (perfect)                              — red dashed
  - Evaluator-decision markers                                 — coloured triangles

Ground truth = last record's observed_value (final result).

Usage:
    python src/scripts/plot_accuracy.py [--data-dir PATH]

Defaults:
    --data-dir  data/runs/unknown-query-estimation
"""

import argparse
import glob
import json
import math
import os

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

COLOR_OBSERVED = "#4C72B0"
COLOR_ESTIMATED = "#DD8452"

DECISION_ORDER = ["batch_query", "query_more_agents", "final_result"]
DECISION_COLORS = ["#4C72B0", "#DD8452", "#55A868"]
DECISION_COLOR_MAP = dict(zip(DECISION_ORDER, DECISION_COLORS, strict=False))

DATASET_GROUPS = [
    "gdp2012",
    "gdp2025",
    "siliconvalley_employees",
    "siliconvalley_revenue",
    "vldbsigmod",
]

VARIANT_ORDER = [
    "none",
    "observed-only",
    "coverage-only",
    "value-mc",
    "value-bucket",
    "value-ci-mc",
    "value-ci-bucket",
    "full-mc",
    "full-bucket",
    "full-ci-mc",
    "full-ci-bucket",
]


def accuracy(value, ground_truth):
    if value is None or ground_truth == 0:
        return None
    return max(0.0, 1.0 - abs(value - ground_truth) / abs(ground_truth)) * 100


def load_records(scenario_dir):
    files = glob.glob(os.path.join(scenario_dir, "*iteration_records*.json"))
    if not files:
        return None
    with open(files[0]) as f:
        records = json.load(f)
    records.sort(key=lambda r: r["estimator_result"].get("agents_queried", 0))
    return records


def extract_series(records):
    """Return (xs, obs_acc, est_acc, decisions, n_agents) or None if no ground truth."""
    last_er = records[-1].get("estimator_result", {})
    ground_truth = last_er.get("observed_value")
    if not ground_truth:
        return None

    xs, obs_acc, est_acc, decisions = [], [], [], []
    for rec in records:
        er = rec.get("estimator_result", {})
        aq = er.get("agents_queried")
        if aq is None:
            continue
        xs.append(aq)
        decisions.append(rec.get("evaluator_decision"))
        obs_acc.append(accuracy(er.get("observed_value"), ground_truth))
        est_v = er.get("estimated_value")
        est_acc.append(accuracy(est_v, ground_truth) if est_v is not None and math.isfinite(est_v) else None)

    return xs, obs_acc, est_acc, decisions, max(xs)


def draw_decision_markers(ax, xs, decisions):
    trans = ax.get_xaxis_transform()
    seen = set()
    for x, dec in zip(xs, decisions, strict=False):
        if dec is None:
            continue
        color = DECISION_COLOR_MAP.get(dec, "#6b7280")
        ax.plot(
            x,
            0.02,
            marker="^",
            markersize=7,
            color=color,
            transform=trans,
            clip_on=False,
            zorder=5,
            label=dec if dec not in seen else "_nolegend_",
        )
        seen.add(dec)


def draw_accuracy_axes(ax, xs, obs_acc, est_acc, decisions, n_agents, title):
    ax.axhline(100, color="#ef4444", linewidth=1.5, linestyle="--", zorder=2)

    x_obs = [xs[i] for i, v in enumerate(obs_acc) if v is not None]
    y_obs = [v for v in obs_acc if v is not None]
    if x_obs:
        ax.plot(
            x_obs,
            y_obs,
            color=COLOR_OBSERVED,
            linewidth=1.5,
            marker="o",
            markersize=4,
            label="Observed",
        )

    x_est = [xs[i] for i, v in enumerate(est_acc) if v is not None]
    y_est = [v for v in est_acc if v is not None]
    if x_est:
        ax.plot(
            x_est,
            y_est,
            color=COLOR_ESTIMATED,
            linewidth=1.5,
            marker="s",
            markersize=4,
            label="Estimated",
        )

    draw_decision_markers(ax, xs, decisions)

    ax.set_ylim(0, 110)
    ax.set_xlim(0.5, n_agents + 0.5)
    agent_labels = [f"A{i}" for i in range(1, n_agents + 1)]
    ax.set_xticks(range(1, n_agents + 1))
    ax.set_xticklabels(agent_labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Agent queried", fontsize=9)
    ax.set_ylabel("Accuracy", fontsize=9)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=10, fontweight="bold")


def build_legend_handles():
    gt_handle = mlines.Line2D([], [], color="#ef4444", linewidth=1.5, linestyle="--", label="Perfect (100%)")
    obs_handle = mlines.Line2D(
        [],
        [],
        color=COLOR_OBSERVED,
        linewidth=1.5,
        marker="o",
        markersize=4,
        label="Observed",
    )
    est_handle = mlines.Line2D(
        [],
        [],
        color=COLOR_ESTIMATED,
        linewidth=1.5,
        marker="s",
        markersize=4,
        label="Estimated",
    )
    dec_handles = [
        mpatches.Patch(color=DECISION_COLORS[i], label=DECISION_ORDER[i]) for i in range(len(DECISION_ORDER))
    ]
    return [obs_handle, est_handle, gt_handle] + dec_handles


def plot_scenario(scenario_dir, out_dir):
    records = load_records(scenario_dir)
    if not records:
        return None

    result = extract_series(records)
    if result is None:
        print(f"  Skipping {os.path.basename(scenario_dir)}: no ground truth")
        return None

    xs, obs_acc, est_acc, decisions, n_agents = result

    fig, ax = plt.subplots(figsize=(max(8, n_agents * 0.55), 5))
    scenario_name = os.path.basename(scenario_dir)
    draw_accuracy_axes(ax, xs, obs_acc, est_acc, decisions, n_agents, f"{scenario_name} — accuracy")

    handles = build_legend_handles()
    ax.legend(handles=handles, fontsize=8, loc="lower right")

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, "accuracy")
    for ext in ("png", "svg", "pdf"):
        fig.savefig(f"{base}.{ext}", dpi=150, bbox_inches="tight")
    print(f"  Saved: {base}.{{png,svg,pdf}}")
    plt.close(fig)

    return xs, obs_acc, est_acc, decisions, n_agents


def plot_combined(group_key, variant_data, out_dir):
    """One subplot per variant, laid out in a grid."""
    variants = [v for v in VARIANT_ORDER if v in variant_data]
    if not variants:
        return

    n_cols = 3
    n_rows = math.ceil(len(variants) / n_cols)
    fig_w = n_cols * 7
    fig_h = n_rows * 4.5

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)
    fig.suptitle(f"{group_key} — accuracy per variant", fontsize=14, fontweight="bold", y=1.01)

    for idx, variant in enumerate(variants):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]
        xs, obs_acc, est_acc, decisions, n_agents = variant_data[variant]
        draw_accuracy_axes(ax, xs, obs_acc, est_acc, decisions, n_agents, variant)

    # Hide unused subplots
    for idx in range(len(variants), n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    handles = build_legend_handles()
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(handles),
        fontsize=9,
        frameon=True,
    )

    fig.tight_layout()
    base = os.path.join(out_dir, f"accuracy_combined_{group_key}")
    for ext in ("png", "svg", "pdf"):
        fig.savefig(f"{base}.{ext}", dpi=150, bbox_inches="tight")
    print(f"  Saved combined: {base}.{{png,svg,pdf}}")
    plt.close(fig)


def main():
    repo_root = os.path.join(os.path.dirname(__file__), "..", "..")
    default_data = os.path.join(repo_root, "data", "runs", "unknown-query-estimation")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=default_data)
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)

    # group_key -> variant -> series tuple
    group_series = {g: {} for g in DATASET_GROUPS}

    for name in sorted(os.listdir(data_dir)):
        scenario_dir = os.path.join(data_dir, name)
        if not os.path.isdir(scenario_dir):
            continue
        print(f"Processing: {name}")
        result = plot_scenario(scenario_dir, scenario_dir)
        if result is None:
            continue
        for group_key in DATASET_GROUPS:
            if name.startswith(group_key + "-"):
                variant = name[len(group_key) + 1 :]
                group_series[group_key][variant] = result
                break

    for group_key in DATASET_GROUPS:
        if group_series[group_key]:
            print(f"Generating combined plot for: {group_key}")
            plot_combined(group_key, group_series[group_key], data_dir)


if __name__ == "__main__":
    main()

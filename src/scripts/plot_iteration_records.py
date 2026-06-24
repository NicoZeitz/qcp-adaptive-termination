#!/usr/bin/env python3
"""Plot iteration records from a QCP estimation run.

Produces two side-by-side panels from a single iteration records file:
  Left  — count (observed, estimated + bounds) vs. optional ground-truth count
  Right — aggregation-type value (observed, estimated + bounds) vs. optional ground truth

Ground truth values default to the last record's observed value when not supplied.
The plot is saved as a PNG next to the iteration file (same name, .png extension).

Usage:
    python plot_iteration_records.py <iteration_file> \\
        [--count-gt N] [--value-gt V] [--output plot.png]
"""

import argparse
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


DECISION_COLORS = {
    "batch_query": "#f59e0b",
    "query_more_agents": "#3b82f6",
    "final_result": "#10b981",
}
DECISION_DEFAULT_COLOR = "#6b7280"

Y_MARGIN_FACTOR = 1.5


def load_records(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def extract_series(records: list[dict]) -> dict:
    agents_queried = []
    observed_values, estimated_values, lower_bounds, upper_bounds = [], [], [], []
    observed_counts, estimated_counts, count_lowers, count_uppers = [], [], [], []
    decisions = []

    for rec in records:
        est = rec["estimator_result"]
        agents_queried.append(est["agents_queried"])
        observed_values.append(est.get("observed_value"))
        estimated_values.append(est.get("estimated_value"))
        lower_bounds.append(est.get("estimated_value_lower"))
        upper_bounds.append(est.get("estimated_value_upper"))
        observed_counts.append(est.get("observed_count"))
        estimated_counts.append(est.get("estimated_count"))
        count_lowers.append(est.get("estimated_count_lower"))
        count_uppers.append(est.get("estimated_count_upper"))
        decisions.append(rec.get("evaluator_decision"))

    last = records[-1]["estimator_result"]
    return {
        "agents_queried": agents_queried,
        "observed_value": observed_values,
        "estimated_value": estimated_values,
        "lower": lower_bounds,
        "upper": upper_bounds,
        "observed_count": observed_counts,
        "estimated_count": estimated_counts,
        "count_lower": count_lowers,
        "count_upper": count_uppers,
        "decisions": decisions,
        "aggregation_type": last.get("aggregation_type", "value"),
        "last_observed_value": last.get("observed_value"),
        "last_observed_count": last.get("observed_count"),
    }


def draw_decision_strip(ax, x: list, decisions: list) -> None:
    decision_labels = sorted(set(d for d in decisions if d))
    decision_y = {label: i for i, label in enumerate(decision_labels)}

    for xi, dec in zip(x, decisions):
        if dec is None:
            continue
        ax.scatter(xi, decision_y[dec],
                   color=DECISION_COLORS.get(dec, DECISION_DEFAULT_COLOR), s=60, zorder=3)

    ax.set_yticks(list(decision_y.values()))
    ax.set_yticklabels(list(decision_y.keys()), fontsize=7)
    ax.set_xlabel("Agents queried")
    ax.set_ylabel("Decision", fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(min(x) - 0.5, max(x) + 0.5)

    patches = [
        mpatches.Patch(color=DECISION_COLORS.get(lbl, DECISION_DEFAULT_COLOR), label=lbl)
        for lbl in decision_labels
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=7)


def draw_estimation_panel(ax_main, ax_dec, x, ground_truth,
                          observed, estimated, lower, upper,
                          obs_label, est_label, y_label, title,
                          decisions) -> None:
    ax_main.axhline(ground_truth, color="#ef4444", linewidth=1.5,
                    linestyle="--", label=f"Ground truth ({ground_truth:,.0f})", zorder=2)

    x_obs = [x[i] for i, v in enumerate(observed) if v is not None]
    y_obs = [v for v in observed if v is not None]
    if x_obs:
        ax_main.plot(x_obs, y_obs, color="#6366f1", linewidth=1.5,
                     marker="o", markersize=4, label=obs_label, zorder=3)

    est_x, est_y, est_lo, est_hi = [], [], [], []
    for i, v in enumerate(estimated):
        lo, hi = lower[i], upper[i]
        if v is not None and lo is not None and hi is not None:
            est_x.append(x[i])
            est_y.append(v)
            est_lo.append(lo)
            est_hi.append(hi)

    if est_x:
        ax_main.plot(est_x, est_y, color="#f97316", linewidth=1.5,
                     marker="s", markersize=4, label=est_label, zorder=3)
        ax_main.fill_between(est_x, est_lo, est_hi,
                              color="#f97316", alpha=0.15, label=f"{est_label} bounds")

    # Compute y-axis limits from observed + estimated values, ignoring the band
    all_finite = [v for v in y_obs + est_y + [ground_truth] if v is not None]
    if all_finite:
        y_min, y_max = min(all_finite), max(all_finite)
        span = max(abs(y_max - y_min), abs(y_max) * 0.01)
        ax_main.set_ylim(y_min - span * (Y_MARGIN_FACTOR - 1),
                         y_max + span * (Y_MARGIN_FACTOR - 1))

    ax_main.set_ylabel(y_label)
    ax_main.legend(loc="upper left", fontsize=8)
    ax_main.grid(True, alpha=0.3)
    ax_main.set_title(title)
    draw_decision_strip(ax_dec, x, decisions)


def plot(series: dict, count_gt: float, value_gt: float, output_path: str, show: bool = False) -> None:
    agg = series["aggregation_type"]
    x = series["agents_queried"]

    fig, axes = plt.subplots(
        2, 2,
        figsize=(18, 8),
        gridspec_kw={"height_ratios": [4, 1]},
    )
    fig.subplots_adjust(hspace=0.08, wspace=0.25)

    draw_estimation_panel(
        axes[0, 0], axes[1, 0], x, count_gt,
        observed=series["observed_count"],
        estimated=series["estimated_count"],
        lower=series["count_lower"],
        upper=series["count_upper"],
        obs_label="Observed count",
        est_label="Estimated count",
        y_label="Count",
        title="Count — unique items seen",
        decisions=series["decisions"],
    )

    draw_estimation_panel(
        axes[0, 1], axes[1, 1], x, value_gt,
        observed=series["observed_value"],
        estimated=series["estimated_value"],
        lower=series["lower"],
        upper=series["upper"],
        obs_label=f"Observed {agg}",
        est_label=f"Estimated {agg}",
        y_label=agg.capitalize(),
        title=f"{agg.capitalize()} estimation",
        decisions=series["decisions"],
    )

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")
    if show:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot QCP estimation iteration records.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("iteration_file", help="Path to the iteration records JSON file")
    parser.add_argument("--count-gt", type=float, default=None, metavar="N",
                        help="Ground truth for count (default: last record's observed_count)")
    parser.add_argument("--value-gt", type=float, default=None, metavar="V",
                        help="Ground truth for the aggregation value (default: last record's observed_value)")
    parser.add_argument("--show", action="store_true",
                        help="Display the plot interactively after saving")
    parser.add_argument("--output", "-o", default=None,
                        help="Override output path (default: same folder/name as iteration file, .png)")
    args = parser.parse_args()

    records = load_records(args.iteration_file)
    if not records:
        print("No records found.", file=sys.stderr)
        sys.exit(1)

    series = extract_series(records)

    count_gt = args.count_gt if args.count_gt is not None else series["last_observed_count"]
    value_gt = args.value_gt if args.value_gt is not None else series["last_observed_value"]

    if count_gt is None:
        print("Warning: no observed_count in records and --count-gt not provided; defaulting to 0.", file=sys.stderr)
        count_gt = 0
    if value_gt is None:
        print("Warning: no observed_value in records and --value-gt not provided; defaulting to 0.", file=sys.stderr)
        value_gt = 0

    output_path = args.output or str(Path(args.iteration_file).with_suffix(".png"))
    plot(series, count_gt, value_gt, output_path, show=args.show)


if __name__ == "__main__":
    main()

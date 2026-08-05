"""
Plot evaluator decisions per agent for each scenario variant.

One heatmap per dataset group (gdp2012, siliconvalley_employees,
siliconvalley_revenue, vldbsigmod). Rows = variants, columns = agents.

With --extra-plots, also produces per-variant line plots (x-axis = agents):
  - sample coverage over agents queried
  - observed + estimated count with confidence interval (bucket variants)
  - observed + estimated value with confidence interval (bucket variants)

Usage:
    python src/scripts/plot_evaluator_decisions.py [--data-dir PATH] [--out-dir PATH]
                                                    [--extra-plots]

Defaults:
    --data-dir  data/runs/unknown-query-estimation
    --out-dir   data/runs/unknown-query-estimation
"""

import argparse
import glob
import json
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

DECISION_ORDER = ["batch_query", "query_more_agents", "final_result"]
DECISION_COLORS = ["#4C72B0", "#DD8452", "#55A868"]

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

DATASET_GROUPS = [
    "gdp2012",
    "gdp2025",
    "siliconvalley_employees",
    "siliconvalley_revenue",
    "vldbsigmod",
]

CMAP = ListedColormap(DECISION_COLORS)


def load_scenario(scenario_dir):
    files = glob.glob(os.path.join(scenario_dir, "*iteration_records*.json"))
    if not files:
        return None
    with open(files[0]) as f:
        records = json.load(f)
    return {r["agent_name"]: r["evaluator_decision"] for r in records}


def load_scenario_records(scenario_dir):
    """Return raw record list sorted by agents_queried, or None if not found."""
    files = glob.glob(os.path.join(scenario_dir, "*iteration_records*.json"))
    if not files:
        return None
    with open(files[0]) as f:
        records = json.load(f)
    records.sort(key=lambda r: r["estimator_result"].get("agents_queried", 0))
    return records


def decision_to_int(d):
    return DECISION_ORDER.index(d)


def plot_group(group_key, scenarios, out_dir, scenario_records=None):
    all_agents = set()
    for data in scenarios.values():
        all_agents.update(data.keys())
    n_agents = max(int(a[1:]) for a in all_agents)
    agent_names = [f"A{i}" for i in range(1, n_agents + 1)]

    variants = [v for v in VARIANT_ORDER if v in scenarios]
    matrix = np.full((len(variants), n_agents), np.nan)
    for row_i, variant in enumerate(variants):
        for col_i, agent in enumerate(agent_names):
            if agent in scenarios[variant]:
                matrix[row_i, col_i] = decision_to_int(scenarios[variant][agent])

    full_bucket = (scenario_records or {}).get("full-bucket")
    n_extra = 3 if full_bucket else 0

    heatmap_h = len(variants) * 0.7 + 1.5
    fig_w = max(8, n_agents * 0.55)
    height_ratios = [heatmap_h] + [3.0] * n_extra
    fig, axes = plt.subplots(
        1 + n_extra,
        1,
        figsize=(fig_w, sum(height_ratios) + 0.5),
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.15},
    )
    ax_heat = axes[0] if n_extra else axes

    ax_heat.imshow(
        np.ma.masked_invalid(matrix),
        cmap=CMAP,
        vmin=0,
        vmax=2,
        aspect="auto",
        interpolation="none",
    )
    ax_heat.set_xticks(range(n_agents))
    ax_heat.set_xticklabels(agent_names, rotation=45, ha="right", fontsize=8)
    ax_heat.tick_params(labelbottom=True)
    ax_heat.set_yticks(range(len(variants)))
    ax_heat.set_yticklabels(variants, fontsize=9)
    ax_heat.set_title(
        f"{group_key} — evaluator decision per agent", fontsize=12, fontweight="bold"
    )
    patches = [
        mpatches.Patch(color=DECISION_COLORS[i], label=DECISION_ORDER[i])
        for i in range(len(DECISION_ORDER))
    ]
    ax_heat.legend(
        handles=patches,
        loc="upper right",
        bbox_to_anchor=(1.13, 1),
        ncol=1,
        fontsize=9,
        frameon=True,
    )

    if full_bucket:
        # x positions: agents_queried - 1  =>  aligns with heatmap column index
        ax_cov, ax_cnt, ax_val = axes[1], axes[2], axes[3]

        # --- coverage (all variants) ---
        for variant in VARIANT_ORDER:
            recs = (scenario_records or {}).get(variant)
            if not recs:
                continue
            xs, ys = [], []
            for r in recs:
                er = r["estimator_result"]
                if er.get("sample_coverage") is not None:
                    xs.append(er["agents_queried"] - 1)
                    ys.append(er["sample_coverage"])
            if xs:
                ax_cov.plot(xs, ys, marker="o", markersize=4, label=variant)
        ax_cov.set_ylabel("Sample coverage", fontsize=9)
        ax_cov.legend(
            fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left", frameon=True
        )
        ax_cov.tick_params(labelbottom=False)

        # --- count and value from full-bucket ---
        for ax, field_obs, field_est, field_lo, field_hi, ylabel in [
            (
                ax_cnt,
                "observed_count",
                "estimated_count",
                "estimated_count_lower",
                "estimated_count_upper",
                "Count",
            ),
            (
                ax_val,
                "observed_value",
                "estimated_value",
                "estimated_value_lower",
                "estimated_value_upper",
                "Value",
            ),
        ]:
            xs_obs, ys_obs = [], []
            xs_est, ys_est, lo, hi = [], [], [], []
            for r in full_bucket:
                er = r["estimator_result"]
                x = er["agents_queried"] - 1
                if er.get(field_obs) is not None:
                    xs_obs.append(x)
                    ys_obs.append(er[field_obs])
                if er.get(field_est) is not None:
                    xs_est.append(x)
                    ys_est.append(er[field_est])
                    lo.append(er.get(field_lo) or er[field_est])
                    hi.append(er.get(field_hi) or er[field_est])
            ax.plot(
                xs_obs,
                ys_obs,
                marker="o",
                markersize=4,
                color="#4C72B0",
                label="observed",
            )
            if xs_est:
                ax.plot(
                    xs_est,
                    ys_est,
                    marker="s",
                    markersize=4,
                    color="#DD8452",
                    label="estimated",
                )
                ax.fill_between(xs_est, lo, hi, color="#DD8452", alpha=0.2, label="CI")
            ax.set_ylabel(ylabel, fontsize=9)
            ax.legend(fontsize=8)

        ax_val.set_xlabel("Agent", fontsize=10)
        ax_val.set_xticks(range(n_agents))
        ax_val.set_xticklabels(agent_names, rotation=45, ha="right", fontsize=8)
    else:
        ax_heat.set_xlabel("Agent", fontsize=10)

    for ext in ("png", "svg", "pdf"):
        out_path = os.path.join(out_dir, f"decisions_{group_key}.{ext}")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")
    plt.close()


def main():
    repo_root = os.path.join(os.path.dirname(__file__), "..", "..")
    default_data = os.path.join(repo_root, "data", "runs", "unknown-query-estimation")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=default_data)
    parser.add_argument(
        "--out-dir", default=None, help="Output directory (defaults to --data-dir)"
    )
    parser.add_argument(
        "--extra-plots",
        action="store_true",
        help="Also produce combined extra_{group} plots (coverage + count CI + value CI)",
    )
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else data_dir
    os.makedirs(out_dir, exist_ok=True)

    for group_key in DATASET_GROUPS:
        prefix = group_key + "-"
        scenarios = {}
        scenario_records = {}
        for name in os.listdir(data_dir):
            if name.startswith(prefix):
                variant = name[len(prefix) :]
                scenario_dir = os.path.join(data_dir, name)
                data = load_scenario(scenario_dir)
                if data:
                    scenarios[variant] = data
                if args.extra_plots:
                    records = load_scenario_records(scenario_dir)
                    if records:
                        scenario_records[variant] = records
        if not scenarios:
            print(f"No data found for {group_key}, skipping.")
            continue
        plot_group(
            group_key,
            scenarios,
            out_dir,
            scenario_records=scenario_records if args.extra_plots else None,
        )


if __name__ == "__main__":
    main()

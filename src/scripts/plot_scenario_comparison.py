"""Plot a side-by-side comparison of estimator variants for a single scenario.

Layout per scenario:
  Left   — heatmap: variants × agents, colored by evaluator decision (all agents shown)
  Right  — 3 stacked line plots from the reference variant (full-ci-bucket):
              1. Sample coverage + true coverage (obs / gt_count)
              2. Observed count + estimated count + CI band
              3. Observed value + estimated value + CI band
  Both columns have equal width.

Usage:
    python src/scripts/plot_scenario_comparison.py <scenario> [options]

    <scenario>  dataset name, e.g. "siliconvalley_revenue"

Options:
    --data-dir PATH      run folder root (default: data/runs/unknown-query-estimation)
    --scenario-dir PATH  scenario JSON root (default: data/unknown-query-estimation)
    --out-dir PATH       output directory (default: same as --data-dir)
    --variants V ...     variants to show in heatmap (space-separated)
    --show               display interactively after saving

Edit VARIANT_LABELS / DEFAULT_VARIANTS at the top to customise display names and order.
"""

import argparse
import glob
import json
import math
import os
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.colors import ListedColormap

plt.rcParams.update(
    {
        "font.size": 22,
        "axes.titlesize": 24,
        "axes.labelsize": 22,
        "xtick.labelsize": 19,
        "ytick.labelsize": 19,
        "legend.fontsize": 18,
    }
)

# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_VARIANTS = [
    "none",
    "coverage-only",
    "observed-only",
    "full-bucket",
    "full-ci-bucket",
]

VARIANT_LABELS = {
    "none": "No Statistics",
    "coverage-only": "Sample Coverage Only",
    "observed-only": "Observed",
    "full-bucket": "Bucket Estimator",
    "full-ci-bucket": "Bucket Estimator with CI",
}

# Right-side plots are driven by this variant's data
REFERENCE_VARIANT = "full-ci-bucket"

DECISION_ORDER = ["batch_query", "query_more_agents", "final_result"]
DECISION_COLORS = ["#4C72B0", "#DD8452", "#55A868"]
DECISION_CMAP = ListedColormap(DECISION_COLORS)

COLOR_OBSERVED = "#6366f1"
COLOR_ESTIMATED = "#f97316"
COLOR_GT = "#ef4444"


# ── Data loading ──────────────────────────────────────────────────────────────


def load_records(variant_dir):
    files = glob.glob(os.path.join(variant_dir, "*iteration_records*.json"))
    if not files:
        return None
    with open(files[0]) as f:
        recs = json.load(f)
    recs.sort(key=lambda r: r["estimator_result"].get("agents_queried", 0))
    return recs


def load_scenario_config(scenario_dir, scenario_name):
    path = os.path.join(scenario_dir, f"{scenario_name}.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def extract_series(records):
    xs, obs_cnt, est_cnt, cnt_lo, cnt_hi = [], [], [], [], []
    obs_val, est_val, val_lo, val_hi = [], [], [], []
    sample_cov, decisions = [], []

    for r in records:
        er = r["estimator_result"]
        xs.append(er["agents_queried"])
        obs_cnt.append(er.get("observed_count"))
        est_cnt.append(er.get("estimated_count"))
        cnt_lo.append(er.get("estimated_count_lower"))
        cnt_hi.append(er.get("estimated_count_upper"))
        obs_val.append(er.get("observed_value"))
        est_val.append(er.get("estimated_value"))
        val_lo.append(er.get("estimated_value_lower"))
        val_hi.append(er.get("estimated_value_upper"))
        sample_cov.append(er.get("sample_coverage"))
        decisions.append(r.get("evaluator_decision"))

    return {
        "x": xs,
        "obs_cnt": obs_cnt,
        "est_cnt": est_cnt,
        "cnt_lo": cnt_lo,
        "cnt_hi": cnt_hi,
        "obs_val": obs_val,
        "est_val": est_val,
        "val_lo": val_lo,
        "val_hi": val_hi,
        "sample_cov": sample_cov,
        "decisions": decisions,
        "agents_total": records[0]["estimator_result"].get(
            "agents_total", len(records)
        ),
        "aggregation_type": records[-1]["estimator_result"].get(
            "aggregation_type", "value"
        ),
        "last_obs_cnt": next((v for v in reversed(obs_cnt) if v is not None), None),
    }


# ── Formatting helpers ────────────────────────────────────────────────────────


def human_format(value, _pos=None):
    if value is None or not math.isfinite(value):
        return str(value)
    for threshold, suffix in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if abs(value) >= threshold:
            v = value / threshold
            return f"{v:.1f}{suffix}" if v != int(v) else f"{v:.0f}{suffix}"
    return f"{value:.0f}" if abs(value) >= 1 else f"{value:.3g}"


def _ylim(primary_vals, band_vals=(), percentile=97, margin_frac=0.12):
    """Y-axis limits clipped at `percentile` to suppress extreme early CI outliers."""
    finite_all = sorted(
        v
        for v in list(primary_vals) + list(band_vals)
        if v is not None and math.isfinite(v)
    )
    if not finite_all:
        return None, None
    hi_idx = min(int(len(finite_all) * percentile / 100), len(finite_all) - 1)
    lo_idx = max(int(len(finite_all) * (1 - percentile / 100)), 0)
    ymin, ymax = finite_all[lo_idx], finite_all[hi_idx]
    span = max(abs(ymax - ymin), abs(ymax) * 0.01, 1e-9)
    return ymin - span * margin_frac, ymax + span * margin_frac


def _set_agent_xticks(ax, n_agents, show_labels):
    ax.set_xticks(range(1, n_agents + 1))
    if show_labels:
        ax.set_xticklabels(
            [f"A{i}" for i in range(1, n_agents + 1)], rotation=45, ha="right"
        )
        ax.set_xlabel("Agents queried")
    else:
        ax.set_xticklabels([])
        ax.tick_params(labelbottom=False)
    ax.set_xlim(0.5, n_agents + 0.5)


# ── Drawing: left side ────────────────────────────────────────────────────────


def draw_heatmap(ax, variants, series_map, n_agents):
    matrix = np.full((len(variants), n_agents), np.nan)
    for row_i, variant in enumerate(variants):
        ser = series_map.get(variant)
        if ser is None:
            continue
        for x, dec in zip(ser["x"], ser["decisions"]):
            col = x - 1
            if 0 <= col < n_agents:
                matrix[row_i, col] = DECISION_ORDER.index(dec)

    ax.imshow(
        np.ma.masked_invalid(matrix),
        cmap=DECISION_CMAP,
        vmin=0,
        vmax=2,
        aspect="auto",
        interpolation="none",
    )
    ax.set_xticks(range(n_agents))
    ax.set_xticklabels(
        [f"A{i}" for i in range(1, n_agents + 1)], rotation=45, ha="right"
    )
    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels([VARIANT_LABELS.get(v, v) for v in variants])
    ax.set_xlabel("Agents queried")
    ax.set_ylabel("Variant")

    patches = [
        mpatches.Patch(
            color=DECISION_COLORS[i], label=DECISION_ORDER[i].replace("_", " ")
        )
        for i in range(len(DECISION_ORDER))
    ]
    ax.legend(
        handles=patches,
        loc="upper right",
        bbox_to_anchor=(1.0, -0.30),
        ncol=3,
        frameon=True,
    )


# ── Drawing: right side ───────────────────────────────────────────────────────


def draw_coverage(ax, ser, gt_count, n_agents, show_labels):
    xs = [x for x, v in zip(ser["x"], ser["sample_cov"]) if v is not None]
    ys = [v for v in ser["sample_cov"] if v is not None]
    if xs:
        ax.plot(
            xs,
            ys,
            color=COLOR_ESTIMATED,
            linewidth=1.5,
            marker="o",
            markersize=3,
            label="Sample coverage",
        )

    if gt_count:
        xs_t = [x for x, v in zip(ser["x"], ser["obs_cnt"]) if v is not None]
        ys_t = [v / gt_count for v in ser["obs_cnt"] if v is not None]
        if xs_t:
            ax.plot(
                xs_t,
                ys_t,
                color=COLOR_OBSERVED,
                linewidth=1.5,
                marker="s",
                markersize=3,
                label=f"True coverage (obs / {int(gt_count)})",
            )

    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylabel("")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    _set_agent_xticks(ax, n_agents, show_labels)


def draw_estimation(
    ax,
    ser,
    field_obs,
    field_est,
    field_lo,
    field_hi,
    gt_value,
    y_label,
    n_agents,
    show_labels,
):
    x = ser["x"]

    if gt_value is not None and math.isfinite(gt_value):
        ax.axhline(
            gt_value,
            color=COLOR_GT,
            linewidth=1.2,
            linestyle="--",
            label=f"Ground truth ({human_format(gt_value)})",
            zorder=2,
        )

    xs_o = [xv for xv, v in zip(x, ser[field_obs]) if v is not None]
    ys_o = [v for v in ser[field_obs] if v is not None]
    if xs_o:
        ax.plot(
            xs_o,
            ys_o,
            color=COLOR_OBSERVED,
            linewidth=1.5,
            marker="o",
            markersize=3,
            label="Observed",
            zorder=3,
        )

    xs_e = [
        xv for xv, v in zip(x, ser[field_est]) if v is not None and math.isfinite(v)
    ]
    ys_e = [v for v in ser[field_est] if v is not None and math.isfinite(v)]
    if xs_e:
        ax.plot(
            xs_e,
            ys_e,
            color=COLOR_ESTIMATED,
            linewidth=1.5,
            marker="s",
            markersize=3,
            label="Estimated",
            zorder=3,
        )

    ci_x, ci_lo, ci_hi = [], [], []
    for xv, lo, hi in zip(x, ser[field_lo], ser[field_hi]):
        if (
            lo is not None
            and hi is not None
            and math.isfinite(lo)
            and math.isfinite(hi)
        ):
            ci_x.append(xv)
            ci_lo.append(lo)
            ci_hi.append(hi)
    if ci_x:
        ax.fill_between(
            ci_x, ci_lo, ci_hi, color=COLOR_ESTIMATED, alpha=0.15, label="CI"
        )

    anchor = [v for v in ys_o + ys_e if math.isfinite(v)]
    if gt_value is not None and math.isfinite(gt_value):
        anchor.append(gt_value)
    ymin, ymax = _ylim(anchor, ci_lo + ci_hi)
    if ymin is not None:
        ax.set_ylim(ymin, ymax)

    ax.set_ylabel("")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(human_format))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    _set_agent_xticks(ax, n_agents, show_labels)


# ── Main plot ─────────────────────────────────────────────────────────────────


def plot_scenario(
    scenario_name, series_map, gt_value, agg_type, n_agents, out_dir, show
):
    variants = [v for v in series_map if series_map[v] is not None]

    ref_ser = series_map.get(REFERENCE_VARIANT) or next(
        s for s in series_map.values() if s is not None
    )
    gt_count = ref_ser["last_obs_cnt"]

    # ── Sizing (all in inches) ────────────────────────────────────────────────
    heatmap_cell_h = 1.0

    right_row_h = 3.6
    right_gap = 0.35
    right_top = 0.5
    right_bot = 1.4
    right_content_h = 3 * right_row_h + 2 * right_gap  # pure axes height
    right_total_h = right_content_h + right_top + right_bot

    heat_content_h_in = len(variants) * heatmap_cell_h
    # Figure height fits both columns; add generous padding for heatmap decorations
    total_h = max(heat_content_h_in + 3.0, right_total_h)
    col_w_in = max(7.0, n_agents * 0.48)

    margin_l = 0.75
    margin_r = 0.3
    col_gap = 0.9
    fig_w = margin_l + 2 * col_w_in + col_gap + margin_r

    fig = plt.figure(figsize=(fig_w, total_h))

    def fw(x):
        return x / fig_w

    def fh(x):
        return x / total_h

    # Right: 3 plots stacked bottom → top, centered vertically in figure
    rx = fw(margin_l + col_w_in + col_gap)
    rw = fw(col_w_in)
    rh = fh(right_row_h)
    rg = fh(right_gap)
    bot = fh((total_h - right_total_h) / 2 + right_bot)

    ax_val = fig.add_axes([rx, bot, rw, rh])
    ax_cnt = fig.add_axes([rx, bot + rh + rg, rw, rh], sharex=ax_val)
    ax_cov = fig.add_axes([rx, bot + 2 * (rh + rg), rw, rh], sharex=ax_val)

    # Left: heatmap axes centered on the same midpoint as the right content block
    right_content_mid = bot + fh(right_content_h) / 2
    heat_bottom = right_content_mid - fh(heat_content_h_in) / 2
    ax_heat = fig.add_axes(
        [fw(margin_l), heat_bottom, fw(col_w_in), fh(heat_content_h_in)]
    )

    # ── Draw ─────────────────────────────────────────────────────────────────
    draw_heatmap(ax_heat, variants, series_map, n_agents)
    ax_heat.set_title("Evaluator decision per agent", pad=6)

    draw_coverage(ax_cov, ref_ser, gt_count, n_agents, show_labels=False)

    draw_estimation(
        ax_cnt,
        ref_ser,
        "obs_cnt",
        "est_cnt",
        "cnt_lo",
        "cnt_hi",
        float(gt_count) if gt_count else None,
        "Count",
        n_agents,
        show_labels=False,
    )

    draw_estimation(
        ax_val,
        ref_ser,
        "obs_val",
        "est_val",
        "val_lo",
        "val_hi",
        gt_value,
        agg_type.capitalize(),
        n_agents,
        show_labels=True,
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    for ext in ("png", "svg", "pdf"):
        out_path = os.path.join(out_dir, f"comparison_{scenario_name}.{ext}")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")

    if show:
        plt.show()
    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    repo_root = os.path.join(os.path.dirname(__file__), "..", "..")
    default_data = os.path.join(repo_root, "data", "runs", "unknown-query-estimation")
    default_scenario = os.path.join(repo_root, "data", "unknown-query-estimation")

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("scenario", help='Dataset name, e.g. "siliconvalley_revenue"')
    parser.add_argument("--data-dir", default=default_data)
    parser.add_argument("--scenario-dir", default=default_scenario)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS, metavar="V")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    scenario_dir = os.path.abspath(args.scenario_dir)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else data_dir
    os.makedirs(out_dir, exist_ok=True)

    config = load_scenario_config(scenario_dir, args.scenario)
    gt_value = config.get("extra", {}).get("full_answer")

    series_map = {}
    agg_type = "value"
    n_agents = 0
    for variant in args.variants:
        variant_dir = os.path.join(data_dir, f"{args.scenario}-{variant}")
        if not os.path.isdir(variant_dir):
            print(f"Warning: {variant_dir} not found, skipping '{variant}'.")
            series_map[variant] = None
            continue
        recs = load_records(variant_dir)
        if recs is None:
            print(f"Warning: no iteration records in {variant_dir}, skipping.")
            series_map[variant] = None
            continue
        ser = extract_series(recs)
        series_map[variant] = ser
        agg_type = ser["aggregation_type"]
        n_agents = max(n_agents, ser["agents_total"])

    if all(v is None for v in series_map.values()):
        print(f"Error: no data found for scenario '{args.scenario}'.", file=sys.stderr)
        sys.exit(1)

    if n_agents == 0:
        n_agents = max(
            (max(s["x"]) for s in series_map.values() if s is not None), default=1
        )

    plot_scenario(
        args.scenario, series_map, gt_value, agg_type, n_agents, out_dir, args.show
    )


if __name__ == "__main__":
    main()

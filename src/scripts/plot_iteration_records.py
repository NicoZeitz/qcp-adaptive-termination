#!/usr/bin/env python3
"""Plot iteration records from a QCP estimation run.

Produces two side-by-side panels from a single iteration records file:
  Left  — count (observed, estimated + bounds) vs. optional ground-truth count
  Right — aggregation-type value (observed, estimated + bounds) vs. optional ground truth

Optionally reads a JSONL trace file to extract completeness-evaluator reasoning
per step, shown as a numbered text panel below the plots.

Ground truth values default to the last record's observed value when not supplied.
The plot is saved as a PNG next to the iteration file (same name, .png extension).

Usage:
    python plot_iteration_records.py <iteration_file> \\
        [--traces <traces.jsonl>] [--count-gt N] [--value-gt V] [--output plot.png] [--show]
"""

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

DECISION_COLORS = {
    "batch_query": "#f59e0b",
    "query_more_agents": "#3b82f6",
    "final_result": "#10b981",
}
DECISION_DEFAULT_COLOR = "#6b7280"

Y_MARGIN_FACTOR = 1.1
Y_MARGIN_FULL = 1.15
BAND_CLIP_PERCENTILE = 97  # ignore band values above this percentile for the top panel
ERROR_ZOOM_PERCENTILE = 75  # y-axis cap for zoomed error panels


def load_records(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def load_reasoning_by_agents_queried(traces_path: str) -> dict[int, list[str]]:
    """Parse JSONL trace file and return reasoning texts keyed by agents_queried.

    Strategy:
    - Find CHAIN 'Query Coordinator - Completeness Evaluator' spans, which carry
      agents_queried in their langfuse.observation.input.
    - Find 'generation' LLM spans that fall within the CHAIN span's time window
      (they share the same parent but have no direct span_id link).
    - Extract reasoning from llm.output_messages.*.message.content (injected by
      capture_reasoning_hook as <reasoning>...</reasoning>).
    """
    spans = []
    with open(traces_path) as f:
        for line in f:
            try:
                spans.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    # Collect CHAIN CE spans with their agents_queried and time window
    ce_windows: list[tuple[int, int, int]] = []  # (agents_queried, start, end)
    for s in spans:
        if s.get("name") != "Query Coordinator - Completeness Evaluator":
            continue
        if s.get("attributes", {}).get("openinference.span.kind") != "CHAIN":
            continue
        raw_input = s.get("attributes", {}).get("langfuse.observation.input")
        if not raw_input:
            continue
        try:
            inp = json.loads(raw_input)
        except (json.JSONDecodeError, TypeError):
            continue
        agents_queried = inp.get("estimator_statistics", {}).get("agents_queried")
        if agents_queried is None:
            continue
        ce_windows.append((int(agents_queried), s.get("start_time", 0), s.get("end_time", 0)))

    # Collect generation spans and extract reasoning from message content
    result: dict[int, list[str]] = {}
    for s in spans:
        if s.get("name") != "generation":
            continue
        attrs = s.get("attributes", {})
        t = s.get("start_time", 0)

        agents_queried = None
        for n, start, end in ce_windows:
            if start <= t <= end:
                agents_queried = n
                break
        if agents_queried is None:
            continue

        # output.value is a JSON list of choice objects, each may have a "reasoning" field
        raw_output = attrs.get("output.value")
        if not raw_output:
            continue
        try:
            choices = json.loads(raw_output) if isinstance(raw_output, str) else raw_output
        except (json.JSONDecodeError, TypeError):
            continue
        for choice in choices if isinstance(choices, list) else []:
            reasoning = choice.get("reasoning")
            if reasoning:
                result.setdefault(agents_queried, []).append(reasoning)

    return result


def extract_series(records: list[dict]) -> dict:
    agents_queried = []
    observed_values, estimated_values, lower_bounds, upper_bounds = [], [], [], []
    observed_counts, estimated_counts, count_lowers, count_uppers = [], [], [], []
    sample_coverages = []
    decisions = []

    for rec in records:
        est = rec.get("estimator_result")
        if est is None:
            continue
        agents_queried.append(est["agents_queried"])
        observed_values.append(est.get("observed_value"))
        estimated_values.append(est.get("estimated_value"))
        lower_bounds.append(est.get("estimated_value_lower"))
        upper_bounds.append(est.get("estimated_value_upper"))
        observed_counts.append(est.get("observed_count"))
        estimated_counts.append(est.get("estimated_count"))
        count_lowers.append(est.get("estimated_count_lower"))
        count_uppers.append(est.get("estimated_count_upper"))
        sample_coverages.append(est.get("sample_coverage"))
        decisions.append(rec.get("evaluator_decision"))

    records_with_est = [r for r in records if r.get("estimator_result") is not None]
    last = records_with_est[-1]["estimator_result"] if records_with_est else {}
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
        "sample_coverage": sample_coverages,
        "decisions": decisions,
        "aggregation_type": last.get("aggregation_type", "value"),
        "last_observed_value": last.get("observed_value"),
        "last_observed_count": last.get("observed_count"),
    }


def human_format(value, _pos):
    if not math.isfinite(value):
        return str(value)
    for threshold, suffix in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if abs(value) >= threshold:
            v = value / threshold
            return f"{v:.0f}{suffix}" if v == int(v) else f"{v:.1f}{suffix}"
    if abs(value) < 1 and value != 0:
        return f"{value:.3g}"
    return f"{value:.0f}"


def nice_step(range_size: float, target_ticks: int = 8) -> float:
    if range_size == 0 or not math.isfinite(range_size):
        return 1.0
    raw = range_size / target_ticks
    mag = 10 ** math.floor(math.log10(raw))
    normalized = raw / mag
    for step in [1, 2, 5, 10]:
        if normalized <= step:
            return step * mag
    return 10 * mag


def draw_decision_markers(ax, x: list, decisions: list) -> None:
    """Draw decision markers as colored triangles along the bottom of the axes."""
    trans = ax.get_xaxis_transform()  # x in data coords, y in axes fraction
    seen = set()
    for xi, dec in zip(x, decisions, strict=False):
        if dec is None:
            continue
        color = DECISION_COLORS.get(dec, DECISION_DEFAULT_COLOR)
        ax.plot(
            xi,
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


def draw_estimation_panel(
    ax,
    x,
    ground_truth,
    observed,
    estimated,
    lower,
    upper,
    y_label,
    title,
    decisions,
    *,
    clip_bands: bool = True,
    y_formatter=None,
) -> None:
    ax.axhline(
        ground_truth,
        color="#ef4444",
        linewidth=1.5,
        linestyle="--",
        label="_nolegend_",
        zorder=2,
    )

    x_obs = [x[i] for i, v in enumerate(observed) if v is not None]
    y_obs = [v for v in observed if v is not None]
    if x_obs:
        ax.plot(
            x_obs,
            y_obs,
            color="#6366f1",
            linewidth=1.5,
            marker="o",
            markersize=4,
            label="Observed",
            zorder=3,
        )
        fmt = y_formatter if y_formatter else human_format
        ax.annotate(
            fmt(y_obs[-1], None),
            xy=(x_obs[-1], y_obs[-1]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            color="#6366f1",
        )

    est_x, est_y = [], []
    for i, v in enumerate(estimated):
        if v is not None:
            est_x.append(x[i])
            est_y.append(v)

    est_ci = []
    for i, v in enumerate(estimated):
        lo, hi = lower[i], upper[i]
        if v is not None and lo is not None and hi is not None:
            est_ci.append((x[i], v, lo, hi))

    if est_x:
        ax.plot(
            est_x,
            est_y,
            color="#f97316",
            linewidth=1.5,
            marker="s",
            markersize=4,
            label="Estimated",
            zorder=3,
        )
        if est_ci:
            bx = [t[0] for t in est_ci]
            bl = [t[2] for t in est_ci]
            bh = [t[3] for t in est_ci]
            ax.fill_between(bx, bl, bh, color="#f97316", alpha=0.15, label="Estimated bounds")
        fmt = y_formatter if y_formatter else human_format
        ax.annotate(
            fmt(est_y[-1], None),
            xy=(est_x[-1], est_y[-1]),
            xytext=(4, -10),
            textcoords="offset points",
            fontsize=7,
            color="#f97316",
        )

    # y-axis limits:
    # clip_bands=True  — anchor to observed+estimated only, tight margin
    # clip_bands=False — anchor to band extent but clip extreme outliers via percentile
    all_values = [v for v in y_obs + est_y + [ground_truth] if v is not None and math.isfinite(v)]
    band_values = [t[2] for t in est_ci] + [t[3] for t in est_ci]
    all_with_bands = sorted(v for v in all_values + band_values if v is not None and math.isfinite(v))

    if clip_bands:
        anchor = all_values
        margin = Y_MARGIN_FACTOR - 1
    else:
        # clip to BAND_CLIP_PERCENTILE to drop extreme early estimates
        if all_with_bands:
            n = len(all_with_bands)
            lo_idx = int(n * (1 - BAND_CLIP_PERCENTILE / 100))
            hi_idx = int(n * BAND_CLIP_PERCENTILE / 100)
            anchor = all_with_bands[lo_idx : hi_idx + 1] or all_with_bands
        else:
            anchor = all_values
        margin = Y_MARGIN_FULL - 1

    if anchor:
        y_min, y_max = min(anchor), max(anchor)
        span = max(abs(y_max - y_min), abs(y_max) * 0.01)
        ylim_range = (y_max - y_min) + 2 * span * margin
        step = nice_step(ylim_range)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(step))
        ax.set_ylim(y_min - span * margin, y_max + span * margin)

    draw_decision_markers(ax, x, decisions)

    ax.set_xlabel("Agents queried")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_ylabel(y_label)
    fmt = y_formatter if y_formatter else human_format
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt))

    # Annotate ground truth value on the right y-axis
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    gt_fmt = y_formatter if y_formatter else human_format
    ax2.set_yticks([ground_truth])
    ax2.set_yticklabels([gt_fmt(ground_truth, None)], color="#ef4444", fontsize=8)
    ax2.tick_params(axis="y", length=0)
    ax2.yaxis.set_tick_params(pad=2)

    ax.grid(True, alpha=0.3)
    ax.set_title(title)


def draw_coverage_panel(ax, x, coverage, observed_count, count_gt, decisions) -> None:
    """Plot sample_coverage (0–1) and actual observed/gt coverage, y-axis fixed to [0, 1]."""
    x_cov = [x[i] for i, v in enumerate(coverage) if v is not None]
    y_cov = [v for v in coverage if v is not None]
    if x_cov:
        ax.plot(
            x_cov,
            y_cov,
            color="#f97316",
            linewidth=1.5,
            marker="s",
            markersize=4,
            label="Sample coverage",
            zorder=3,
        )
        ax.annotate(
            f"{y_cov[-1]:.0%}",
            xy=(x_cov[-1], y_cov[-1]),
            xytext=(4, -10),
            textcoords="offset points",
            fontsize=7,
            color="#f97316",
        )

    if count_gt:
        x_act = [x[i] for i, v in enumerate(observed_count) if v is not None]
        y_act = [v / count_gt for v in observed_count if v is not None]
        if x_act:
            ax.plot(
                x_act,
                y_act,
                color="#6366f1",
                linewidth=1.5,
                marker="o",
                markersize=4,
                label="Observed / ground truth",
                zorder=3,
            )
            ax.annotate(
                f"{y_act[-1]:.0%}",
                xy=(x_act[-1], y_act[-1]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                color="#6366f1",
            )

    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_xlabel("Agents queried")
    ax.set_ylabel("Coverage")
    draw_decision_markers(ax, x, decisions)
    ax.grid(True, alpha=0.3)
    ax.set_title("Sample coverage")


def compute_error_series(series: dict, count_gt: float, value_gt: float) -> dict:
    """Per-step |observed - gt| and |estimated - gt| for value and count."""
    value_obs_err, value_est_err = [], []
    count_obs_err, count_est_err = [], []
    for obs_v, est_v, obs_c, est_c in zip(
        series["observed_value"],
        series["estimated_value"],
        series["observed_count"],
        series["estimated_count"],
        strict=False,
    ):
        value_obs_err.append(abs(obs_v - value_gt) if obs_v is not None else None)
        value_est_err.append(abs(est_v - value_gt) if est_v is not None and math.isfinite(est_v) else None)
        count_obs_err.append(abs(obs_c - count_gt) if obs_c is not None else None)
        count_est_err.append(abs(est_c - count_gt) if est_c is not None and math.isfinite(est_c) else None)
    return {
        "value_obs_err": value_obs_err,
        "value_est_err": value_est_err,
        "count_obs_err": count_obs_err,
        "count_est_err": count_est_err,
    }


def draw_error_panel(ax, x, obs_err, est_err, decisions, title, y_label, *, clip_bands: bool = False) -> None:
    """Plot |observed - gt| (blue) and |estimated - gt| (orange) per step.

    clip_bands=False — full range (all steps including inf-est outliers skipped)
    clip_bands=True  — zoom: y-axis anchored to finite observed errors only
    """
    x_obs = [x[i] for i, v in enumerate(obs_err) if v is not None]
    y_obs = [v for v in obs_err if v is not None]
    if x_obs:
        ax.plot(
            x_obs,
            y_obs,
            color="#6366f1",
            linewidth=1.5,
            marker="o",
            markersize=4,
            label="Observed",
            zorder=3,
        )

    x_est = [x[i] for i, v in enumerate(est_err) if v is not None]
    y_est = [v for v in est_err if v is not None]
    if x_est:
        ax.plot(
            x_est,
            y_est,
            color="#f97316",
            linewidth=1.5,
            marker="s",
            markersize=4,
            label="Estimated",
            zorder=3,
        )

    if clip_bands:
        # clip to BAND_CLIP_PERCENTILE across both series to drop early outliers
        all_finite = sorted(v for v in y_obs + y_est if math.isfinite(v))
        if all_finite:
            hi_idx = int(len(all_finite) * ERROR_ZOOM_PERCENTILE / 100)
            anchor = all_finite[: hi_idx + 1]
        else:
            anchor = []
    else:
        anchor = [v for v in y_obs + y_est if math.isfinite(v)]

    if anchor:
        y_max = max(anchor)
        margin = y_max * (Y_MARGIN_FACTOR - 1)
        ax.set_ylim(0, y_max + margin)

    ax.set_xlabel("Agents queried")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_ylabel(y_label)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(human_format))
    draw_decision_markers(ax, x, decisions)
    ax.grid(True, alpha=0.3)
    ax.set_title(title)


def save_reasoning_text(
    series: dict,
    reasoning_by_n: dict[int, list[str]],
    output_path: str,
    count_gt: float = 0,
    value_gt: float = 0,
) -> None:
    lines = []
    for n, dec, obs_v, est_v, obs_c, est_c in zip(
        series["agents_queried"],
        series["decisions"],
        series["observed_value"],
        series["estimated_value"],
        series["observed_count"],
        series["estimated_count"],
        strict=False,
    ):

        def fmt(v: float) -> str:
            if not math.isfinite(v):
                return str(v)
            if abs(v) >= 1e9:
                return f"{v / 1e9:.3f}G"
            if abs(v) >= 1e6:
                return f"{v / 1e6:.3f}M"
            if abs(v) >= 1e3:
                return f"{v / 1e3:.3f}k"
            if abs(v) < 0.001 and v != 0:
                return f"{v:.6f}"
            return f"{v:.3f}"

        texts = reasoning_by_n.get(n, [])
        parts = [f"[agents_queried={n}  decision={dec}]"]

        if obs_v is not None:
            obs_err = abs(obs_v - value_gt)
            parts.append(f"  value:  observed={fmt(obs_v)}  gt={fmt(value_gt)}  |obs error|={fmt(obs_err)}")
        if est_v is not None and math.isfinite(est_v):
            est_err = abs(est_v - value_gt)
            parts.append(f"           estimated={fmt(est_v)}  |est error|={fmt(est_err)}")
        if obs_c is not None:
            obs_err = abs(obs_c - count_gt)
            parts.append(f"  count:  observed={fmt(obs_c)}  gt={fmt(count_gt)}  |obs error|={fmt(obs_err)}")
        if est_c is not None and math.isfinite(est_c):
            est_err = abs(est_c - count_gt)
            parts.append(f"           estimated={fmt(est_c)}  |est error|={fmt(est_err)}")

        if texts:
            full = "\n\n".join(texts)
            parts.append(full)

        lines.append("\n".join(parts))

    Path(output_path).write_text("\n\n" + ("\n\n" + "—" * 80 + "\n\n").join(lines) + "\n")
    print(f"Reasoning saved to {output_path}")


def save_single_figure(fig, path_no_ext: str, fmt: str = "png") -> None:
    out = f"{path_no_ext}.{fmt}"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  {out}")


def save_subplot(fig, subplots_dir: Path, name: str) -> None:
    """Save a subplot in png, svg, and pdf into subplots_dir."""
    base = str(subplots_dir / name)
    for fmt in ("png", "svg", "pdf"):
        save_single_figure(fig, base, fmt)


def attach_legend(ax, *, add_gt: bool = False) -> None:
    """Add a legend inside ax, optionally prepending a Ground truth handle."""
    handles, labels = ax.get_legend_handles_labels()
    if add_gt:
        gt_h = mlines.Line2D([], [], color="#ef4444", linewidth=1.5, linestyle="--", label="Ground truth")
        handles, labels = handles + [gt_h], labels + ["Ground truth"]
    visible = [(handle, label) for handle, label in zip(handles, labels, strict=False) if not label.startswith("_")]
    if visible:
        hs, ls = zip(*visible, strict=False)
        ax.legend(hs, ls, fontsize=8, loc="best")


def plot(
    series: dict,
    count_gt: float,
    value_gt: float,
    output_path: str,
    show: bool = False,
    include_confidence_intervals: bool = True,
) -> None:
    agg = series["aggregation_type"]
    x = series["agents_queried"]
    n = len(x)
    no_bounds = [None] * n

    errs = compute_error_series(series, count_gt, value_gt)

    count_common = dict(
        observed=series["observed_count"],
        estimated=series["estimated_count"],
        lower=series["count_lower"] if include_confidence_intervals else no_bounds,
        upper=series["count_upper"] if include_confidence_intervals else no_bounds,
        y_label="Count",
        decisions=series["decisions"],
    )
    value_common = dict(
        observed=series["observed_value"],
        estimated=series["estimated_value"],
        lower=series["lower"] if include_confidence_intervals else no_bounds,
        upper=series["upper"] if include_confidence_intervals else no_bounds,
        y_label=agg.capitalize(),
        decisions=series["decisions"],
    )

    # ------------------------------------------------------------------
    # Individual subplot figures — saved into a subplots/ subfolder
    # ------------------------------------------------------------------
    subplots_dir = Path(output_path).parent / (Path(output_path).name + " - subplots")
    subplots_dir.mkdir(exist_ok=True)
    print(f"Saving subplots to {subplots_dir} ...")

    # Coverage
    fig_s, ax_s = plt.subplots(figsize=(8, 5))
    draw_coverage_panel(
        ax_s,
        x,
        series["sample_coverage"],
        series["observed_count"],
        count_gt,
        series["decisions"],
    )
    attach_legend(ax_s)
    save_subplot(fig_s, subplots_dir, "coverage")
    plt.close(fig_s)

    # Count full CI
    fig_s, ax_s = plt.subplots(figsize=(8, 5))
    draw_estimation_panel(
        ax_s,
        x,
        count_gt,
        **count_common,
        title="Count — Full Interval",
        clip_bands=False,
    )
    attach_legend(ax_s, add_gt=True)
    save_subplot(fig_s, subplots_dir, "count_full_ci")
    plt.close(fig_s)

    # Count zoomed
    fig_s, ax_s = plt.subplots(figsize=(8, 5))
    draw_estimation_panel(ax_s, x, count_gt, **count_common, title="Count — Zoomed", clip_bands=True)
    attach_legend(ax_s, add_gt=True)
    save_subplot(fig_s, subplots_dir, "count_zoomed")
    plt.close(fig_s)

    # Count error full
    fig_s, ax_s = plt.subplots(figsize=(8, 5))
    draw_error_panel(
        ax_s,
        x,
        errs["count_obs_err"],
        errs["count_est_err"],
        series["decisions"],
        title="Count — Absolute Error",
        y_label="|Count error|",
        clip_bands=False,
    )
    attach_legend(ax_s)
    save_subplot(fig_s, subplots_dir, "count_error")
    plt.close(fig_s)

    # Count error zoomed
    fig_s, ax_s = plt.subplots(figsize=(8, 5))
    draw_error_panel(
        ax_s,
        x,
        errs["count_obs_err"],
        errs["count_est_err"],
        series["decisions"],
        title="Count — Absolute Error — Zoomed",
        y_label="|Count error|",
        clip_bands=True,
    )
    attach_legend(ax_s)
    save_subplot(fig_s, subplots_dir, "count_error_zoomed")
    plt.close(fig_s)

    # Value full CI
    fig_s, ax_s = plt.subplots(figsize=(8, 5))
    draw_estimation_panel(
        ax_s,
        x,
        value_gt,
        **value_common,
        title=f"{agg.capitalize()} — Full Interval",
        clip_bands=False,
    )
    attach_legend(ax_s, add_gt=True)
    save_subplot(fig_s, subplots_dir, "value_full_ci")
    plt.close(fig_s)

    # Value zoomed
    fig_s, ax_s = plt.subplots(figsize=(8, 5))
    draw_estimation_panel(
        ax_s,
        x,
        value_gt,
        **value_common,
        title=f"{agg.capitalize()} — Zoomed",
        clip_bands=True,
    )
    attach_legend(ax_s, add_gt=True)
    save_subplot(fig_s, subplots_dir, "value_zoomed")
    plt.close(fig_s)

    # Value error full
    fig_s, ax_s = plt.subplots(figsize=(8, 5))
    draw_error_panel(
        ax_s,
        x,
        errs["value_obs_err"],
        errs["value_est_err"],
        series["decisions"],
        title=f"{agg.capitalize()} — Absolute Error",
        y_label=f"|{agg.capitalize()} error|",
        clip_bands=False,
    )
    attach_legend(ax_s)
    save_subplot(fig_s, subplots_dir, "value_error")
    plt.close(fig_s)

    # Value error zoomed
    fig_s, ax_s = plt.subplots(figsize=(8, 5))
    draw_error_panel(
        ax_s,
        x,
        errs["value_obs_err"],
        errs["value_est_err"],
        series["decisions"],
        title=f"{agg.capitalize()} — Absolute Error — Zoomed",
        y_label=f"|{agg.capitalize()} error|",
        clip_bands=True,
    )
    attach_legend(ax_s)
    save_subplot(fig_s, subplots_dir, "value_error_zoomed")
    plt.close(fig_s)

    # ------------------------------------------------------------------
    # Combined figure: 4 rows × 3 columns
    #   [0,0] coverage     | [0,1] count full CI       | [0,2] value full CI
    #   [1,0] (empty)      | [1,1] count zoomed        | [1,2] value zoomed
    #   [2,0] (empty)      | [2,1] count absolute error| [2,2] value absolute error
    #   [3,0] (empty)      | [3,1] count error zoomed  | [3,2] value error zoomed
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(4, 3, figsize=(24, 24))
    fig.subplots_adjust(hspace=0.35, wspace=0.3)

    draw_coverage_panel(
        axes[0, 0],
        x,
        series["sample_coverage"],
        series["observed_count"],
        count_gt,
        series["decisions"],
    )
    axes[1, 0].set_visible(False)
    axes[2, 0].set_visible(False)
    axes[3, 0].set_visible(False)

    draw_estimation_panel(
        axes[0, 1],
        x,
        count_gt,
        **count_common,
        title="Count — Full Interval",
        clip_bands=False,
    )
    draw_estimation_panel(axes[1, 1], x, count_gt, **count_common, title="Count — Zoomed", clip_bands=True)
    draw_error_panel(
        axes[2, 1],
        x,
        errs["count_obs_err"],
        errs["count_est_err"],
        series["decisions"],
        title="Count — Absolute Error",
        y_label="|Count error|",
        clip_bands=False,
    )
    draw_error_panel(
        axes[3, 1],
        x,
        errs["count_obs_err"],
        errs["count_est_err"],
        series["decisions"],
        title="Count — Absolute Error — Zoomed",
        y_label="|Count error|",
        clip_bands=True,
    )

    draw_estimation_panel(
        axes[0, 2],
        x,
        value_gt,
        **value_common,
        title=f"{agg.capitalize()} — Full Interval",
        clip_bands=False,
    )
    draw_estimation_panel(
        axes[1, 2],
        x,
        value_gt,
        **value_common,
        title=f"{agg.capitalize()} — Zoomed",
        clip_bands=True,
    )
    draw_error_panel(
        axes[2, 2],
        x,
        errs["value_obs_err"],
        errs["value_est_err"],
        series["decisions"],
        title=f"{agg.capitalize()} — Absolute Error",
        y_label=f"|{agg.capitalize()} error|",
        clip_bands=False,
    )
    draw_error_panel(
        axes[3, 2],
        x,
        errs["value_obs_err"],
        errs["value_est_err"],
        series["decisions"],
        title=f"{agg.capitalize()} — Absolute Error — Zoomed",
        y_label=f"|{agg.capitalize()} error|",
        clip_bands=True,
    )

    # Single shared legend below all panels
    cov_handles, cov_labels = axes[0, 0].get_legend_handles_labels()
    est_handles, est_labels = axes[0, 1].get_legend_handles_labels()
    gt_handle = mlines.Line2D([], [], color="#ef4444", linewidth=1.5, linestyle="--", label="Ground truth")
    all_handles = cov_handles + est_handles + [gt_handle]
    all_labels = cov_labels + est_labels + ["Ground truth"]
    seen_labels: set[str] = set()
    deduped_handles, deduped_labels = [], []
    for handle, label in zip(all_handles, all_labels, strict=False):
        if label not in seen_labels and not label.startswith("_"):
            seen_labels.add(label)
            deduped_handles.append(handle)
            deduped_labels.append(label)
    fig.legend(
        deduped_handles,
        deduped_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(deduped_handles),
        fontsize=9,
        frameon=True,
    )

    print("Saving combined plot ...")
    for fmt in ("png", "svg", "pdf"):
        save_single_figure(fig, str(Path(output_path)), fmt)
    plt.close(fig)

    if show:
        plt.show()


def load_final_result_value(run_prefix: str) -> float | None:
    """Read value from <prefix> - final_result.json if it exists."""
    path = Path(run_prefix).parent / f"{Path(run_prefix).name} - final_result.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            body = json.load(f)
        return float(body["result"]["data"][0][0])
    except Exception:
        return None


def resolve_run_files(run_prefix: str) -> tuple[Path, Path | None]:
    """Given a run prefix path, find the iteration records JSON and optional traces JSONL.

    The prefix is e.g. '/path/to/Stage 1-3 - 2026-06-24 12-20-04'.
    Iteration records file matches: <prefix> - iteration_records - *.json
    Traces file matches:            <prefix>_traces.jsonl
    """
    prefix = Path(run_prefix)
    folder = prefix.parent
    stem = prefix.name

    iteration_files = sorted(folder.glob(f"{stem} - iteration_records - *.json"))
    if not iteration_files:
        print(
            f"Error: no iteration records file found for prefix '{stem}' in {folder}",
            file=sys.stderr,
        )
        sys.exit(1)
    if len(iteration_files) > 1:
        print(
            f"Warning: multiple iteration records files found, using first: {iteration_files[0].name}",
            file=sys.stderr,
        )

    traces_file = folder / f"{stem}_traces.jsonl"
    return iteration_files[0], traces_file if traces_file.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot QCP estimation iteration records.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "run_prefix",
        help="Run prefix path, e.g. '../data/runs/foo/Stage 1-3 - 2026-06-24 12-20-04'",
    )
    parser.add_argument(
        "--count-gt",
        type=float,
        default=None,
        metavar="N",
        help="Ground truth for count (default: last record's observed_count)",
    )
    parser.add_argument(
        "--value-gt",
        type=float,
        default=None,
        metavar="V",
        help="Ground truth for the aggregation value (default: last record's observed_value)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively after saving",
    )
    parser.add_argument(
        "--no-confidence-intervals",
        action="store_true",
        help="Omit confidence interval bands from the plot",
    )
    args = parser.parse_args()

    iteration_file, traces_file = resolve_run_files(args.run_prefix)
    print(f"Iteration records: {iteration_file.name}")
    if traces_file:
        print(f"Traces: {traces_file.name}")

    records = load_records(str(iteration_file))
    if not records:
        print("No records found.", file=sys.stderr)
        sys.exit(1)

    series = extract_series(records)

    reasoning_by_n: dict[int, list[str]] | None = None
    if traces_file:
        reasoning_by_n = load_reasoning_by_agents_queried(str(traces_file))
        if not reasoning_by_n:
            print("Warning: no reasoning found in trace file.", file=sys.stderr)

    count_gt = args.count_gt if args.count_gt is not None else series["last_observed_count"]
    value_gt = args.value_gt if args.value_gt is not None else series["last_observed_value"]

    if count_gt is None:
        print(
            "Warning: no observed_count in records and --count-gt not provided; defaulting to 0.",
            file=sys.stderr,
        )
        count_gt = 0
    if value_gt is None:
        print(
            "Warning: no observed_value in records and --value-gt not provided; defaulting to 0.",
            file=sys.stderr,
        )
        value_gt = 0

    txt_path = args.run_prefix + " - completeness reasoning.txt"
    save_reasoning_text(
        series,
        reasoning_by_n if reasoning_by_n else {},
        txt_path,
        count_gt=count_gt,
        value_gt=value_gt,
    )

    plot(
        series,
        count_gt,
        value_gt,
        args.run_prefix,
        show=args.show,
        include_confidence_intervals=not args.no_confidence_intervals,
    )


if __name__ == "__main__":
    main()

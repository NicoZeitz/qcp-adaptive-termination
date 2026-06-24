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
import sys
import textwrap
from pathlib import Path
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
REASONING_WRAP_WIDTH = 110


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


def human_format(value, _pos):
    for threshold, suffix in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if abs(value) >= threshold:
            v = value / threshold
            return f"{v:.0f}{suffix}" if v == int(v) else f"{v:.1f}{suffix}"
    return f"{value:.0f}"


def nice_step(range_size: float, target_ticks: int = 8) -> float:
    import math
    if range_size == 0:
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
    for xi, dec in zip(x, decisions):
        if dec is None:
            continue
        color = DECISION_COLORS.get(dec, DECISION_DEFAULT_COLOR)
        ax.plot(xi, 0.02, marker="^", markersize=7, color=color,
                transform=trans, clip_on=False, zorder=5,
                label=dec if dec not in seen else "_nolegend_")
        seen.add(dec)


def draw_estimation_panel(ax, x, ground_truth,
                          observed, estimated, lower, upper,
                          obs_label, est_label, y_label, title,
                          decisions, *, clip_bands: bool = True) -> None:
    ax.axhline(ground_truth, color="#ef4444", linewidth=1.5,
               linestyle="--", label=f"Ground truth ({ground_truth:,.0f})", zorder=2)

    x_obs = [x[i] for i, v in enumerate(observed) if v is not None]
    y_obs = [v for v in observed if v is not None]
    if x_obs:
        ax.plot(x_obs, y_obs, color="#6366f1", linewidth=1.5,
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
        ax.plot(est_x, est_y, color="#f97316", linewidth=1.5,
                marker="s", markersize=4, label=est_label, zorder=3)
        ax.fill_between(est_x, est_lo, est_hi,
                        color="#f97316", alpha=0.15, label=f"{est_label} bounds")

    # y-axis limits:
    # clip_bands=True  — anchor to observed+estimated only, tight margin
    # clip_bands=False — anchor to band extent but clip extreme outliers via percentile
    all_values = [v for v in y_obs + est_y + [ground_truth] if v is not None]
    all_with_bands = sorted(v for v in all_values + est_lo + est_hi if v is not None)

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
        step = nice_step(y_max - y_min)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(step))
        ax.set_ylim(y_min - span * margin, y_max + span * margin)

    draw_decision_markers(ax, x, decisions)

    ax.set_xlabel("Agents queried")
    ax.set_ylabel(y_label)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(human_format))
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title(title)


def save_reasoning_text(series: dict, reasoning_by_n: dict[int, list[str]], output_path: str) -> None:
    lines = []
    for n, dec in zip(series["agents_queried"], series["decisions"]):
        texts = reasoning_by_n.get(n)
        if not texts:
            continue
        full = "\n\n".join(texts)
        header = f"[agents_queried={n}  decision={dec}]"
        wrapped = textwrap.fill(full, width=REASONING_WRAP_WIDTH, subsequent_indent="    ")
        lines.append(f"{header}\n{wrapped}")
    Path(output_path).write_text("\n\n" + ("\n\n" + "—" * 80 + "\n\n").join(lines) + "\n")
    print(f"Reasoning saved to {output_path}")


def plot(series: dict, count_gt: float, value_gt: float, output_path: str,
         show: bool = False) -> None:
    agg = series["aggregation_type"]
    x = series["agents_queried"]

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.subplots_adjust(hspace=0.35, wspace=0.25)

    common = dict(
        observed=series["observed_count"],
        estimated=series["estimated_count"],
        lower=series["count_lower"],
        upper=series["count_upper"],
        obs_label="Observed count",
        est_label="Estimated count",
        y_label="Count",
        decisions=series["decisions"],
    )
    draw_estimation_panel(axes[0, 0], x, count_gt, **common,
                          title="Count — full confidence interval", clip_bands=False)
    draw_estimation_panel(axes[1, 0], x, count_gt, **common,
                          title="Count — zoomed", clip_bands=True)

    common = dict(
        observed=series["observed_value"],
        estimated=series["estimated_value"],
        lower=series["lower"],
        upper=series["upper"],
        obs_label=f"Observed {agg}",
        est_label=f"Estimated {agg}",
        y_label=agg.capitalize(),
        decisions=series["decisions"],
    )
    draw_estimation_panel(axes[0, 1], x, value_gt, **common,
                          title=f"{agg.capitalize()} — full confidence interval", clip_bands=False)
    draw_estimation_panel(axes[1, 1], x, value_gt, **common,
                          title=f"{agg.capitalize()} — zoomed", clip_bands=True)

    fig.savefig(str(Path(output_path).with_suffix(".png")), dpi=150, bbox_inches="tight")
    fig.savefig(str(Path(output_path).with_suffix(".svg")), dpi=150, bbox_inches="tight")
    fig.savefig(str(Path(output_path).with_suffix(".pdf")), dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")
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
        print(f"Error: no iteration records file found for prefix '{stem}' in {folder}", file=sys.stderr)
        sys.exit(1)
    if len(iteration_files) > 1:
        print(f"Warning: multiple iteration records files found, using first: {iteration_files[0].name}", file=sys.stderr)

    traces_file = folder / f"{stem}_traces.jsonl"
    return iteration_files[0], traces_file if traces_file.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot QCP estimation iteration records.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("run_prefix",
                        help="Run prefix path, e.g. '../data/runs/foo/Stage 1-3 - 2026-06-24 12-20-04'")
    parser.add_argument("--count-gt", type=float, default=None, metavar="N",
                        help="Ground truth for count (default: last record's observed_count)")
    parser.add_argument("--value-gt", type=float, default=None, metavar="V",
                        help="Ground truth for the aggregation value (default: last record's observed_value)")
    parser.add_argument("--show", action="store_true",
                        help="Display the plot interactively after saving")
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
    value_gt = args.value_gt if args.value_gt is not None else load_final_result_value(args.run_prefix) or series["last_observed_value"]

    if count_gt is None:
        print("Warning: no observed_count in records and --count-gt not provided; defaulting to 0.", file=sys.stderr)
        count_gt = 0
    if value_gt is None:
        print("Warning: no observed_value in records and --value-gt not provided; defaulting to 0.", file=sys.stderr)
        value_gt = 0

    if reasoning_by_n:
        txt_path = str(Path(args.run_prefix).with_suffix(" - completeness reasoning.txt"))
        save_reasoning_text(series, reasoning_by_n, txt_path)

    plot(series, count_gt, value_gt, args.run_prefix, show=args.show)


if __name__ == "__main__":
    main()

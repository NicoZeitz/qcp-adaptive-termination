"""Cross-query aggregation for completeness evaluation."""

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.completeness_evaluation.plotting import save_aggregate_plots


def _build_step_frame(results_dir: Path, step_frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Derive one stop-level summary row per (query, step).

    Each ``*_evaluator.csv`` has one row per completeness check; the last row is
    the stop check. We take that row and add the derived step-level fields
    (``stop_fraction``, ``early_stop``, ``agents_queried_at_stop``) so the frame
    used by the cross-query plots is one row per (query, step).
    """
    step_rows: list[dict[str, Any]] = []
    for frame in step_frames:
        query = frame.iloc[0]["query"]
        if frame.empty:
            continue
        last = frame.iloc[-1].to_dict()
        agents_total = last.get("agents_total")
        agents_queried = last.get("agents_queried")
        last["agents_queried_at_stop"] = agents_queried
        last["stop_fraction"] = (agents_queried / agents_total) if agents_total else 0.0
        last["early_stop"] = bool(agents_queried < agents_total) if agents_queried is not None else None
        last["count_est_at_stop"] = last.get("estimated_count")
        last["count_upper_at_stop"] = last.get("estimated_count_upper")
        last["coverage_at_stop"] = last.get("coverage_achieved")
        last["query"] = query
        step_rows.append(last)
    return pd.DataFrame(step_rows)


def _aggregate_usage(results_dir: Path, input_dir: Path) -> pd.DataFrame | None:
    """Collapse each query's ``usage.json`` into one summary row (LLM + embeddings)."""
    rows: list[dict[str, Any]] = []
    for input_file in sorted(input_dir.glob("*.json")):
        query_slug = input_file.stem
        usage_path = results_dir / query_slug / "usage.json"
        if not usage_path.exists():
            continue
        with usage_path.open(encoding="utf-8") as f:
            usage = json.load(f)
        llm = usage["llm"]
        emb = usage["embeddings"]
        llm_input = sum(v.get("input_tokens", 0) for v in llm["per_model"].values())
        emb_tokens = sum(v.get("input_tokens", 0) for v in emb["per_model"].values())
        rows.append(
            {
                "query": query_slug,
                "llm_runs": llm["runs"],
                "llm_requests": llm["requests"],
                "llm_input_tokens": llm_input,
                "llm_cost": llm["cost"],
                "embedding_calls": emb["calls"],
                "embedding_tokens": emb_tokens,
                "embedding_cost": emb["cost"],
                "total_cost": usage["total_cost"],
                "time_s": usage["time_s"],
            }
        )
    if not rows:
        return None
    return pd.DataFrame(rows)


def aggregate(results_dir: Path, input_dir: Path) -> Path:
    """Concatenate per-step evaluator CSVs into top-level summary artifacts.

    Reads each per-query subfolder's ``*_evaluator.csv`` and writes a combined
    ``evaluator_summary_all_queries.csv`` (all per-check rows), the stop-level
    frame ``evaluator_step_summary_all_queries.csv``, and the cross-query
    aggregate plots, all at the top level of ``results_dir``.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    status_rows: list[dict[str, Any]] = []

    for input_file in sorted(input_dir.glob("*.json")):
        query_slug = input_file.stem
        run_dir = results_dir / query_slug
        step_paths = sorted(run_dir.glob("*_evaluator.csv")) if run_dir.is_dir() else []
        if not step_paths:
            status_rows.append({"query": query_slug, "status": "missing_output", "details": "no evaluator csv"})
            continue
        for step_path in step_paths:
            frame = pd.read_csv(step_path)
            frame.insert(0, "query", query_slug)
            frames.append(frame)
        status_rows.append({"query": query_slug, "status": "complete", "details": ""})

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined.to_csv(results_dir / "evaluator_summary_all_queries.csv", index=False)

        step_frame = _build_step_frame(results_dir, frames)
        step_frame.to_csv(results_dir / "evaluator_step_summary_all_queries.csv", index=False)
        save_aggregate_plots(step_frame, results_dir)

        logging.info("Aggregated %d/%d input queries to %s", len(frames), len(status_rows), results_dir)
    else:
        logging.warning("No evaluator summaries found under %s", results_dir)

    usage_frame = _aggregate_usage(results_dir, input_dir)
    if usage_frame is not None:
        usage_frame.to_csv(results_dir / "usage_summary.csv", index=False)
        logging.info("Aggregated usage for %d queries to %s", len(usage_frame), results_dir / "usage_summary.csv")

    pd.DataFrame(status_rows).to_csv(results_dir / "evaluator_run_status.csv", index=False)
    return results_dir

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
        if frame.empty:
            continue
        query = frame.iloc[0]["query"]
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


_USAGE_SUMS = [
    "requests",
    "input_tokens",
    "cached_input_tokens",
    "cache_write_tokens",
    "output_tokens",
    "reasoning_tokens",
    "cost",
    "cost_uncached",
]


def _load_usage_calls(results_dir: Path, input_dir: Path, completed: set[str]) -> pd.DataFrame | None:
    """Concatenate every completed query's per-call ``usage.csv``."""
    frames: list[pd.DataFrame] = []
    for input_file in sorted(input_dir.glob("*.json")):
        if input_file.stem not in completed:
            continue
        usage_path = results_dir / input_file.stem / "usage.csv"
        if not usage_path.exists():
            continue
        frame = pd.read_csv(usage_path, keep_default_na=False, na_values=[""])
        frame = frame.assign(**{c: 0 for c in _USAGE_SUMS if c not in frame.columns})
        frame.insert(0, "query", input_file.stem)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else None


def _usage_summary(calls: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Sum usage per group, split by call type and role."""
    calls = calls.assign(
        role=calls["role"].fillna("").astype(str),
        step_id=calls["step_id"].fillna("").astype(str),
    )
    grouped = calls.groupby([*by, "type", "role"], dropna=False)
    summary = pd.DataFrame(grouped[_USAGE_SUMS].sum())
    summary.insert(0, "calls", grouped.size())
    return summary.reset_index()


def _aggregate_usage(results_dir: Path, input_dir: Path, completed: set[str]) -> dict[str, pd.DataFrame]:
    """Usage per query, per (query, step) and per role, from the per-call usage CSVs."""
    calls = _load_usage_calls(results_dir, input_dir, completed)
    if calls is None or calls.empty:
        return {}
    per_query = pd.DataFrame(calls.groupby("query")[_USAGE_SUMS].sum())
    per_query.insert(0, "llm_runs", calls.loc[calls["type"] == "llm"].groupby("query").size())
    per_query.insert(1, "embedding_calls", calls.loc[calls["type"] == "embedding"].groupby("query").size())
    per_query = per_query.fillna(0).reset_index()
    return {
        "usage_summary": per_query,
        "usage_by_step": _usage_summary(calls, ["query", "step_id"]),
        "usage_by_role": _usage_summary(calls, []),
    }


def _query_status(results_dir: Path, query_slug: str, has_csv: bool) -> dict[str, Any]:
    """Status row of one query from its ``query_summary.json``."""
    summary_path = results_dir / query_slug / "query_summary.json"
    if not summary_path.exists():
        return {
            "query": query_slug,
            "status": "complete" if has_csv else "missing_output",
            "details": "" if has_csv else "no evaluator csv",
        }
    with summary_path.open(encoding="utf-8") as f:
        summary = json.load(f)
    error = summary.get("error") or ""
    return {
        "query": query_slug,
        "status": summary.get("status"),
        "details": error.strip().splitlines()[-1] if error else "",
        "steps_total": summary.get("steps_total"),
        "steps_built": len(summary.get("steps_built") or []),
        "compounded_steps": ",".join(summary.get("compounded_steps") or []),
        "full_answer": json.dumps(summary.get("full_answer"), default=str),
        "final_result": json.dumps(summary.get("final_result"), default=str),
        "final_scalar": summary.get("final_scalar"),
        "final_rel_err": summary.get("final_rel_err"),
        "combined_completeness": summary.get("combined_completeness"),
    }


def aggregate(results_dir: Path, input_dir: Path) -> Path:
    """Concatenate per-step evaluator CSVs into top-level summary artifacts.

    Reads each per-query subfolder's ``*_evaluator.csv`` and writes a combined
    ``evaluator_summary_all_queries.csv`` (all per-check rows), the stop-level
    frame ``evaluator_step_summary_all_queries.csv``, usage summaries, the query
    status table and the cross-query aggregate plots, all at the top level of ``results_dir``.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    status_rows: list[dict[str, Any]] = []
    completed: set[str] = set()

    for input_file in sorted(input_dir.glob("*.json")):
        query_slug = input_file.stem
        run_dir = results_dir / query_slug
        step_paths = sorted(run_dir.glob("*_evaluator.csv")) if run_dir.is_dir() else []
        summary = _query_status(results_dir, query_slug, bool(step_paths))
        status_rows.append(summary)
        if summary["status"] != "complete":
            continue
        completed.add(query_slug)
        for step_path in step_paths:
            frame = pd.read_csv(step_path)
            frame.insert(0, "query", query_slug)
            frames.append(frame)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined.to_csv(results_dir / "evaluator_summary_all_queries.csv", index=False)

        step_frame = _build_step_frame(results_dir, frames)
        step_frame.to_csv(results_dir / "evaluator_step_summary_all_queries.csv", index=False)
        save_aggregate_plots(step_frame, results_dir)

        logging.info("Aggregated %d steps of %d completed queries to %s", len(frames), len(completed), results_dir)
    else:
        logging.warning("No evaluator summaries found under %s", results_dir)

    for name, frame in _aggregate_usage(results_dir, input_dir, completed).items():
        frame.to_csv(results_dir / f"{name}.csv", index=False)
    if (results_dir / "usage_summary.csv").exists():
        logging.info("Aggregated usage to %s", results_dir / "usage_summary.csv")

    status = pd.DataFrame(status_rows)
    status.to_csv(results_dir / "evaluator_run_status.csv", index=False)
    print(
        f"[AGGREGATE] {results_dir}: " + ", ".join(f"{k}={v}" for k, v in status["status"].value_counts().items()),
        flush=True,
    )
    return results_dir

"""Cross-query aggregation for estimator evaluation."""

import logging
from pathlib import Path
from typing import Any

import pandas as pd


def aggregate(results_dir: Path, input_dir: Path) -> Path:
    """Concatenate per-query evaluator summaries into a top-level CSV.

    Reads each per-query subfolder's ``*_evaluator_summary.csv`` and writes a
    combined ``evaluator_summary_all_queries.csv`` at the top level of
    ``results_dir`` (next to the per-query subfolders).
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    status_rows: list[dict[str, Any]] = []

    for input_file in sorted(input_dir.glob("*.json")):
        query_slug = input_file.stem
        run_dir = results_dir / query_slug
        summary_paths = sorted(run_dir.glob("*_evaluator_summary.csv")) if run_dir.is_dir() else []
        if not summary_paths:
            status_rows.append({"query": query_slug, "status": "missing_output", "details": "no summary csv"})
            continue
        for summary_path in summary_paths:
            frame = pd.read_csv(summary_path)
            frame.insert(0, "query", query_slug)
            frames.append(frame)
        status_rows.append({"query": query_slug, "status": "complete", "details": ""})

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined.to_csv(results_dir / "evaluator_summary_all_queries.csv", index=False)
        logging.info("Aggregated %d/%d input queries to %s", len(frames), len(status_rows), results_dir)
    else:
        logging.warning("No evaluator summaries found under %s", results_dir)

    pd.DataFrame(status_rows).to_csv(results_dir / "evaluator_run_status.csv", index=False)
    return results_dir

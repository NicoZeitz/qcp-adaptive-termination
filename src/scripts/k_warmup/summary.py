"""Per-table summary writing for the k-warmup study."""

import logging
from pathlib import Path

import pandas as pd

from scripts.k_warmup.metrics import error_k, flatness_k
from scripts.k_warmup.worker import ERROR_THRESHOLDS, FLATNESS_THRESHOLDS, TableSummary


def write_table_summary(summaries: list[TableSummary], output_dir: Path):
    """Write each table's flatness/error k per ordering to a CSV.

    Args:
        summaries: the per-table k-warmup summaries
        output_dir: directory to write the summary CSV
    """
    if not summaries:
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for summary in summaries:
        for ordering, traj in summary.trajectories.items():
            row = {
                "table": summary.table,
                "ordering": ordering,
                "agents_total": summary.agents_total,
                "true_count": summary.true_count,
            }
            for thr in FLATNESS_THRESHOLDS:
                row[f"flat_k_{int(thr * 100)}"] = flatness_k(traj, thr)
            for thr in ERROR_THRESHOLDS:
                row[f"error_k_{int(thr * 100)}"] = error_k(traj, thr)
            row["final_estimated"] = traj.estimated[-1] if traj.estimated else None
            row["final_error"] = traj.error[-1] if traj.error else None
            rows.append(row)

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "k_summary.csv", index=False)
    logging.info("Wrote k summary: %d rows to %s", len(frame), output_dir / "k_summary.csv")

    failure_rows = [
        {"table": s.table, "ordering": ordering, "error": error}
        for s in summaries
        for ordering, error in s.failures.items()
    ]
    if failure_rows:
        pd.DataFrame(failure_rows).to_csv(output_dir / "k_failures.csv", index=False)

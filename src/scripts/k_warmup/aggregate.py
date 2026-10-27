"""Cross-table aggregation for the k-warmup study.

Merges per-table k summaries into a per-ordering distribution of k across tables
and writes a summary CSV with the distribution and a point pick (default p90).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.k_warmup.worker import ERROR_THRESHOLDS, FLATNESS_THRESHOLDS, STRATEGIES

_K_COLUMNS = [f"flat_k_{int(t * 100)}" for t in FLATNESS_THRESHOLDS] + [
    f"error_k_{int(t * 100)}" for t in ERROR_THRESHOLDS
]


def _column_floats(frame: pd.DataFrame, col: str) -> np.ndarray:
    """Extract a column as a float ndarray with NaN removed."""
    import numpy as np  # noqa: PLC0415

    vals = np.asarray(pd.to_numeric(frame[col], errors="coerce"), dtype=float)
    return vals[~np.isnan(vals)]


def _quantile_like(sub: pd.DataFrame, col: str, q: float) -> float | None:
    """Quantile of a k column from a dataframe subset, ignoring NaN/None."""
    vals = _column_floats(sub, col)
    if vals.size == 0:
        return None
    return float(np.quantile(vals, q))


def _filter_ordering(frame: pd.DataFrame, ordering: str) -> pd.DataFrame:
    """Rows of a frame matching one ordering as a DataFrame."""
    return frame.loc[frame["ordering"] == ordering]


def aggregate_k_summaries(results_dir: Path, input_dir: Path) -> Path:
    """Merge per-table k summaries into a per-ordering distribution + point pick.

    Args:
        results_dir: directory holding per-query run folders
        input_dir: input directory (unused beyond preserving signature)

    Returns:
        (Path): the aggregated k distribution CSV path
    """
    del input_dir
    rows: list[pd.DataFrame] = []
    for summary_path in sorted(results_dir.glob("*/k_summary.csv")):
        rows.append(pd.read_csv(summary_path))

    if not rows:
        logging.warning("No k summaries found under %s", results_dir)
        return results_dir / "k_warmup_distribution.csv"

    frame = pd.concat(rows, ignore_index=True)
    rows_found = frame if not frame.empty else pd.DataFrame(columns=["table", "ordering"])
    # Persist the raw merged per-table k values.
    raw_path = results_dir / "k_warmup_all_tables.csv"
    rows_found.to_csv(raw_path, index=False)
    logging.info("Aggregated %d table rows to %s", len(rows_found), raw_path)

    dist_rows: list[dict[str, object]] = []
    for ordering in [str(s) for s in STRATEGIES]:
        sub = _filter_ordering(frame, ordering)
        row: dict[str, object] = {"ordering": ordering, "tables": int(len(sub))}
        for c in _K_COLUMNS:
            row[f"p50_{c}"] = _quantile_like(sub, c, 0.5)
            row[f"p90_{c}"] = _quantile_like(sub, c, 0.9)
        dist_rows.append(row)

    dist_frame = pd.DataFrame(dist_rows)
    dist_path = results_dir / "k_warmup_distribution.csv"
    dist_frame.to_csv(dist_path, index=False)
    logging.info("Wrote k distribution to %s", dist_path)

    _plot_distributions(frame, results_dir)
    return dist_path


def _plot_distributions(frame: pd.DataFrame, results_dir: Path):
    """Plot the k distribution per ordering for the flatness thresholds."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        logging.warning("matplotlib unavailable, skipping plots: %s", exc)
        return

    for thr in FLATNESS_THRESHOLDS:
        col = f"flat_k_{int(thr * 100)}"
        fig, ax = plt.subplots(figsize=(8, 5))
        orderings = [str(s) for s in STRATEGIES]
        for ordering in orderings:
            sub = _filter_ordering(frame, ordering)
            vals = _column_floats(sub, col)
            if vals.size == 0:
                continue
            ax.scatter(
                np.full(vals.size, ordering),
                vals,
                alpha=0.5,
                s=30,
                label=ordering,
            )
        ax.set_title(f"k (flatness threshold {thr:.0%}) by ordering")
        ax.set_xlabel("Ordering")
        ax.set_ylabel("k (agents queried at stability)")
        ax.legend()
        fig.tight_layout()
        for ext in ("png", "pdf", "svg"):
            fig.savefig(results_dir / f"k_warmup_flat_{int(thr * 100)}.{ext}", dpi=150, bbox_inches="tight")
        plt.close(fig)


def aggregate(results_dir: Path, input_dir: Path) -> Path:
    """K-warmup entry-point aggregator (mirrors other evaluation modules)."""
    return aggregate_k_summaries(results_dir, input_dir)

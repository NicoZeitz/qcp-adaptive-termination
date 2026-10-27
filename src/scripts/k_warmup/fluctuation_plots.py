"""Script: plot estimator one-step fluctuation per table and for all tables.

Reads the k-warmup trajectory CSVs (``table_*/trajectories/*.csv``) and plots,
for the posterior ordering, the relative one-step fluctuation of the estimator's
total-entity estimate. One figure per table plus one aggregate figure.

Usage:
    python -m scripts.k_warmup.fluctuation_plots <results_dir>
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qcp_adaptive_termination.models import SelectionStrategyType

_FLUCTUATION_COL = "rel_fluctuation"


def _load_trajectories(results_dir: Path, ordering: str) -> list[pd.DataFrame]:
    """Load all posterior trajectory CSVs into a list of frames."""
    frames: list[pd.DataFrame] = []
    for path in sorted(results_dir.glob("table_*/trajectories/*.csv")):
        frame = pd.read_csv(path)
        if frame["ordering"].iloc[0] != ordering:
            continue
        est = frame["estimated"].to_numpy(dtype=float)
        fluct = np.full_like(est, np.nan)
        fluct[1:] = np.abs(np.diff(est)) / np.maximum(np.abs(est[:-1]), 1e-12)
        frame[_FLUCTUATION_COL] = fluct
        frame["table"] = path.parent.parent.name
        frames.append(frame)
    return frames


def _save(fig, path: Path):
    """Save a figure in all configured formats."""
    for ext in ("png", "pdf", "svg"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_table(frames: list[pd.DataFrame], output_dir: Path):
    """Plot one fluctuation curve per table (posterior only)."""
    out = output_dir / "fluctuation_per_table"
    out.mkdir(parents=True, exist_ok=True)
    for frame in frames:
        table = frame["table"].iloc[0]
        fig, ax1 = plt.subplots(figsize=(9, 5))
        x = frame["agents_queried"]
        ax1.plot(x, frame[_FLUCTUATION_COL], marker="o", ms=3, label="one-step |ΔN̂|/N̂")
        ax1.axhline(0.10, color="red", ls="--", alpha=0.7, label="10% tolerance")
        ax1.set_xlabel("Agents queried")
        ax1.set_ylabel("Relative fluctuation |ΔN̂|/N̂")
        ax1.set_title(f"Estimator fluctuation — {table} (true={frame['true_count'].iloc[0]:.0f})")
        ax1.legend()

        ax2 = ax1.twinx()
        ax2.plot(x, frame["estimated"], color="navy", alpha=0.35, ls=":", label="N̂")
        ax2.set_ylabel("N̂")
        _save(fig, out / f"{table}_fluctuation")


def plot_aggregate(frames: list[pd.DataFrame], output_dir: Path):
    """Plot the fluctuation curves of all tables on one axes (normalized x)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for frame in frames:
        agents_total = frame["agents_queried"].max()
        x = frame["agents_queried"] / agents_total
        ax.plot(x, frame[_FLUCTUATION_COL], alpha=0.6, marker=".", ms=3, label=frame["table"].iloc[0])
    ax.axhline(0.10, color="red", ls="--", alpha=0.8, label="10% tolerance")
    ax.set_xlabel("Agents queried (fraction of total)")
    ax.set_ylabel("Relative fluctuation |ΔN̂|/N̂")
    ax.set_title("Estimator one-step fluctuation — all tables (posterior)")
    ax.legend(fontsize=7, ncol=2)
    _save(fig, output_dir / "fluctuation_all_tables")


def run_script(args: list[str]):
    """Plot fluctuation curves for a results directory."""
    results_dir = Path(args[0] if args else ".")
    output_dir = results_dir
    ordering = SelectionStrategyType.POSTERIOR.value
    frames = _load_trajectories(results_dir, ordering)
    if not frames:
        print(f"No posterior trajectories under {results_dir}")
        return
    plot_per_table(frames, output_dir)
    plot_aggregate(frames, output_dir)
    print(f"Wrote fluctuation plots for {len(frames)} tables to {output_dir}")


if __name__ == "__main__":
    run_script(sys.argv[1:])

"""Reproduce a Fig-6-style grid for the unknown-unknown estimators.

Rows   : Ideal (λ=0, ρ=0) | Realistic (λ=4, ρ=1) | Rare (λ=4, ρ=0)
Columns: varying number of sources w
X-axis : total sample size (n_per_source × w), capped at 500
Y-axis : SUM estimate
Lines  : Observed · Naive · Freq · Bucket-Naive · Bucket-Freq · Ground Truth

Usage:
    python figure_synthetic.py
    python figure_synthetic.py --count-estimator chao84
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
import random

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from qcp_adaptive_termination.estimator import Estimator, ValueEstimator, CountEstimator
from qcp_adaptive_termination.estimator.data_item import DataItem

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument(
    "--count-estimator",
    default="chao92",
    choices=[e.value for e in CountEstimator],
    help="Count estimator to use (default: chao92)",
)
args = parser.parse_args()
COUNT_ESTIMATOR = CountEstimator(args.count_estimator)

# ── Experiment grid ───────────────────────────────────────────────────────────

REGIMES = [
    ("Ideal\nλ=0, ρ=0", 0.0, 0.0),
    ("Realistic\nλ=4, ρ=1", 4.0, 1.0),
    ("Rare\nλ=4, ρ=0", 4.0, 0.0),
]
W_VALUES = [100, 10, 5]
N_POPULATION = 100
SEED = 42
SAMPLE_SIZES = list(range(80, 501, 60))  # 80, 140, 200, ..., 500

# ── Synthetic data helpers ────────────────────────────────────────────────────


def build_population(n: int, seed: int) -> dict[str, float]:
    random.seed(seed)
    return {
        f"company_{i + 1:03d}": 10 + i * (1000 - 10) // (n - 1) if n > 1 else 500
        for i in range(n)
    }


def build_publicity(
    companies: dict[str, float], lambda_: float, rho: float
) -> dict[str, float]:
    names_by_value = sorted(companies, key=lambda k: companies[k], reverse=True)
    values = [companies[name] for name in names_by_value]
    v_min, v_max = min(values), max(values)
    v_span = max(v_max - v_min, 1e-12)

    rank_prior = {
        name: 1.0 / math.pow(rank + 1, lambda_) if lambda_ > 0 else 1.0
        for rank, name in enumerate(names_by_value)
    }
    value_signal = {name: (companies[name] - v_min) / v_span for name in names_by_value}
    mixed = {
        name: (1.0 - rho) * rank_prior[name] + rho * value_signal[name]
        for name in names_by_value
    }
    total = sum(mixed.values()) or 1.0
    return {name: mixed[name] / total for name in names_by_value}


def sample_source(
    source_id: str,
    n: int,
    companies: dict[str, float],
    publicity: dict[str, float],
    rng: random.Random,
) -> list[DataItem]:
    names = list(companies.keys())
    target = min(n, len(names))
    selected: list[DataItem] = []
    for _ in range(target):
        weights = [max(publicity[name], 1e-12) for name in names]
        chosen = rng.choices(names, weights=weights, k=1)[0]
        selected.append(
            DataItem(name=chosen, value=companies[chosen], source_id=source_id)
        )
        names.remove(chosen)
    return selected


def run_experiment(
    n_per_source: int,
    w: int,
    lambda_: float,
    rho: float,
    seed: int,
    count_estimator: CountEstimator,
) -> dict[str, float]:
    print(
        f"Running experiment: n_per_source={n_per_source}, w={w}, λ={lambda_}, ρ={rho}, count_estimator={count_estimator.value}"
    )
    companies = build_population(N_POPULATION, seed)
    publicity = build_publicity(companies, lambda_, rho)
    true_sum = sum(companies.values())

    rng = random.Random(seed + n_per_source * 1000 + w)
    sources = [
        sample_source(f"s{i}", n_per_source, companies, publicity, rng)
        for i in range(w)
    ]
    e = Estimator(sources)

    return {
        "true_sum": true_sum,
        "observed": e.sample_sum(),
        "naive": e.estimated_sum(ValueEstimator.NAIVE, count_estimator),
        "freq": e.estimated_sum(ValueEstimator.FREQUENCY, count_estimator),
        "freq_f12": e.estimated_sum(ValueEstimator.FREQUENCY_F12, count_estimator),
        "bucket_naive": e.estimated_sum(
            ValueEstimator.NAIVE, count_estimator, use_bucket=True
        ),
        "bucket_freq": e.estimated_sum(
            ValueEstimator.FREQUENCY, count_estimator, use_bucket=True
        ),
        "bucket_freq_f12": e.estimated_sum(
            ValueEstimator.FREQUENCY_F12, count_estimator, use_bucket=True
        ),
        "monte_carlo": e.estimated_sum(
            ValueEstimator.NAIVE, count_estimator, use_monte_carlo=True
        ),
    }


def _run_experiment_star(args):
    return args, run_experiment(*args)


# ── Plotting ──────────────────────────────────────────────────────────────────

LINE_STYLES: dict[str, dict] = {
    "observed": dict(color="#1f77b4", ls="-.", marker="o", ms=3.5, label="Observed"),
    "naive": dict(color="#ff7f0e", ls="-.", marker="s", ms=3.5, label="Naive"),
    "freq": dict(color="#2ca02c", ls="-.", marker="^", ms=3.5, label="Freq"),
    # "freq_f12":    dict(color="#2ca02c", ls="-.", marker="v", ms=3.5, label="Freq f12"),
    # "bucket_naive": dict(color="#9467bd", ls=":", marker="D", ms=3.5, label="Bucket-Naive"),
    "bucket_freq": dict(
        color="#d62728", ls=":", marker="x", ms=4.0, label="Bucket-Freq"
    ),
    # "bucket_freq_f12": dict(color="#d62728", ls=":",  marker="X", ms=4.0, label="Bucket-Freq f12"),
    "monte_carlo": dict(
        color="#8c564b", ls="--", marker="P", ms=3.5, label="Monte Carlo"
    ),
    "true_sum": dict(
        color="gray", alpha=0.5, ls="-", marker=None, label="Ground Truth"
    ),
}


if __name__ == "__main__":
    # Build all experiment parameter combos
    all_params = [
        (max(1, total // w), w, lambda_, rho, SEED, COUNT_ESTIMATOR)
        for w in W_VALUES
        for _, lambda_, rho in REGIMES
        for total in SAMPLE_SIZES
    ]

    print(
        f"Running {len(all_params)} experiments in parallel (count estimator: {COUNT_ESTIMATOR.value})..."
    )
    results_cache: dict[tuple, dict[str, float]] = {}
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(_run_experiment_star, p): p for p in all_params}
        for future in as_completed(futures):
            params, result = future.result()
            results_cache[params] = result

    fig, axes = plt.subplots(
        len(REGIMES),
        len(W_VALUES),
        figsize=(4.5 * len(W_VALUES), 3.5 * len(REGIMES)),
        sharex=False,
        sharey=False,
    )

    for col, w in enumerate(W_VALUES):
        for row, (regime_label, lambda_, rho) in enumerate(REGIMES):
            ax = axes[row][col]

            results = [
                results_cache[
                    (max(1, total // w), w, lambda_, rho, SEED, COUNT_ESTIMATOR)
                ]
                for total in SAMPLE_SIZES
            ]
            xs = SAMPLE_SIZES

            for key, style in LINE_STYLES.items():
                ys_plot = [
                    r[key] if math.isfinite(r[key]) else float("nan") for r in results
                ]
                if all(math.isnan(v) for v in ys_plot):
                    continue
                lw = 1.6 if key == "true_sum" else 1.4
                ax.plot(xs, ys_plot, **style, lw=lw)

            ax.set_xlabel("Sample size", fontsize=8)
            ax.set_xlim(SAMPLE_SIZES[0] - 10, SAMPLE_SIZES[-1] + 10)
            ax.set_xticks(SAMPLE_SIZES[::2])
            ax.tick_params(labelsize=7)
            ax.yaxis.set_major_formatter(
                ticker.FuncFormatter(
                    lambda v, _: f"{v / 1000:.1f}k" if abs(v) >= 1000 else f"{v:.0f}"
                )
            )
            ax.grid(True, alpha=0.25, lw=0.5)

            if row == 0:
                ax.set_title(f"w={w}", fontsize=10, fontweight="bold")
            if col == 0:
                ax.set_ylabel(regime_label + "\n\nSUM estimate", fontsize=8)

    legend_handles = [
        mlines.Line2D(
            [],
            [],
            label=style["label"],
            color=style["color"],
            ls=style["ls"],
            marker=style.get("marker"),
            markersize=style.get("ms", 5),
        )
        for style in LINE_STYLES.values()
    ]
    fig.legend(
        legend_handles,
        [str(h.get_label()) for h in legend_handles],
        loc="lower center",
        ncol=len(LINE_STYLES),
        fontsize=9,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.04),
    )

    fig.suptitle(
        f"Synthetic data with varying sources (w), publicity skew (λ), value correlation (ρ)\n"
        f"Population N={N_POPULATION}, count estimator={COUNT_ESTIMATOR.value}, Y-axis=SUM",
        fontsize=11,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    filename = f"figure_synthetic_{COUNT_ESTIMATOR.value}.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved {filename}")

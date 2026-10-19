"""Monte Carlo count estimation via RBF-surfaced KL-divergence search."""

import math
import random
from collections.abc import Sequence

import numpy as np
from scipy.interpolate import RBFInterpolator

from qcp_adaptive_termination.estimator.count_estimators import estimate_count
from qcp_adaptive_termination.estimator.frequency import FrequencyStats
from qcp_adaptive_termination.models.estimation import CountEstimator

MC_DEFAULT_NB_RUNS = 200
MC_DEFAULT_LAMBDA_GRID = tuple(round(v * 0.1, 1) for v in range(-4, 5))
MC_RANDOM_SEED = None
MC_DEFAULT_MAX_THETA_POINTS = 7
MC_LAPLACE_ALPHA = 0.1


def _rbf_search_min(
    points: Sequence[tuple[float, float, float]],
    count_lower_bound: int,
    count_upper_bound: int,
    lambda_grid: Sequence[float],
) -> tuple[int, float]:
    """
    Fit a thin-plate-spline RBF surface over (N, λ, KL) points and return the minimum.

    Falls back to the raw minimum point when there are fewer than 3 points or
    when all N or λ values are identical (degenerate 2D surface).
    """
    if len(points) < 3:
        best_count, best_lambda, _ = min(points, key=lambda p: p[2])
        return int(best_count), float(best_lambda)

    count_values = {p[0] for p in points}
    lambda_values = {p[1] for p in points}
    if len(count_values) < 2 or len(lambda_values) < 2:
        best_count, best_lambda, _ = min(points, key=lambda p: p[2])
        return int(best_count), float(best_lambda)

    xy = np.array([[p[0], p[1]] for p in points], dtype=float)
    z = np.array([p[2] for p in points], dtype=float)
    xy_scale = xy.std(axis=0)
    xy_scale[xy_scale == 0] = 1.0
    try:
        rbf = RBFInterpolator(xy / xy_scale, z, kernel="thin_plate_spline")
    except np.linalg.LinAlgError:
        best_count, best_lambda, _ = min(points, key=lambda p: p[2])
        return int(best_count), float(best_lambda)

    candidates = np.array(
        [[n, lam] for n in range(count_lower_bound, count_upper_bound + 1) for lam in lambda_grid],
        dtype=float,
    )
    scores = rbf(candidates / xy_scale)
    best_idx = int(np.argmin(scores))
    return int(candidates[best_idx, 0]), float(candidates[best_idx, 1])


def mc_count(
    stats: FrequencyStats,
    count_estimator: CountEstimator,
    *,
    nb_runs: int = MC_DEFAULT_NB_RUNS,
    lambda_grid: tuple[float, ...] = MC_DEFAULT_LAMBDA_GRID,
    random_seed: int | None = MC_RANDOM_SEED,
    max_theta_points: int = MC_DEFAULT_MAX_THETA_POINTS,
    laplace_alpha: float = MC_LAPLACE_ALPHA,
) -> int:
    """
    Estimate total unique entity count via Monte Carlo simulation.

    Searches over a grid of (N, λ) parameters where N is the candidate population
    size and λ governs the skew of an exponential publicity distribution.
    For each candidate, simulates the sampling process and measures KL-divergence
    from the observed frequency statistics. Fits an RBF surface over the grid
    and returns the N at the minimum.
    """
    estimated_count = estimate_count(stats, count_estimator)
    count_upper_bound = (
        stats.observed_unique_entities * 2
        if not math.isfinite(estimated_count)
        else max(stats.observed_unique_entities, int(math.ceil(estimated_count)))
    )

    # If Chao is not finite or collapses to the observed count, there is no
    # meaningful N-grid search to perform for Monte Carlo.
    if count_upper_bound == stats.observed_unique_entities:
        return stats.observed_unique_entities

    jump = max(1, math.ceil((count_upper_bound - stats.observed_unique_entities) / max_theta_points))
    i_last = (count_upper_bound - stats.observed_unique_entities + 1) // jump
    count_grid = [stats.observed_unique_entities + i * jump for i in range(max(i_last, 1))]
    if count_grid[-1] < count_upper_bound:
        count_grid.append(count_upper_bound)
    elif count_grid[-1] > count_upper_bound:
        count_grid[-1] = count_upper_bound

    source_sizes = [len(src) for src in stats.source_samples]
    points: list[tuple[float, float, float]] = [
        (
            float(candidate_count),
            float(candidate_lambda),
            _mc_distance(stats, candidate_count, candidate_lambda, source_sizes, nb_runs, random_seed, laplace_alpha),
        )
        for candidate_count in count_grid
        for candidate_lambda in lambda_grid
    ]
    best_count, _ = _rbf_search_min(points, stats.observed_unique_entities, count_upper_bound, lambda_grid)
    return int(best_count)


def _mc_distance(
    stats: FrequencyStats,
    theta_n: int,
    theta_lambda: float,
    n_w: Sequence[int],
    nb_runs: int,
    random_seed: int | None,
    laplace_alpha: float,
) -> float:
    """Average KL-divergence between observed and simulated f-statistic distributions."""
    rng = random.Random(random_seed)
    total = 0.0

    # Observed f-statistics: fs[k] = number of entities seen exactly k times.
    observed_frequencies: dict[int, int] = {}
    for record in stats.entities.values():
        k = record.abundance
        observed_frequencies[k] = observed_frequencies.get(k, 0) + 1

    for _ in range(nb_runs):
        # Simulate one integrated dataset Q under parameters (theta_n, theta_lambda).
        simulated_abundance: dict[int, int] = {}  # entity_idx -> count across all workers
        for worker_size in n_w:
            local: set[int] = set()
            target = min(worker_size, theta_n)
            while len(local) < target:
                idx = int(math.floor(rng.random() * theta_n))
                p = math.exp(-idx * theta_lambda / max(theta_n, 1))
                if rng.random() <= p:
                    local.add(idx)
            for idx in local:
                simulated_abundance[idx] = simulated_abundance.get(idx, 0) + 1

        # Convert simulation to f-statistics.
        simulated_frequencies: dict[int, int] = {}
        for cnt in simulated_abundance.values():
            simulated_frequencies[cnt] = simulated_frequencies.get(cnt, 0) + 1

        # All frequency counts present in either distribution.
        all_counts = set(observed_frequencies.keys()) | set(simulated_frequencies.keys())
        if not all_counts:
            continue

        alpha = laplace_alpha
        # Smooth F_S: entities with counts only in Q get alpha mass.
        p_denom = sum(observed_frequencies.values()) + alpha * len(all_counts)
        q_denom = sum(simulated_frequencies.values()) + alpha * len(all_counts)

        kl = 0.0
        for k in all_counts:
            p = (observed_frequencies.get(k, 0) + alpha) / p_denom
            q = (simulated_frequencies.get(k, 0) + alpha) / q_denom
            kl += p * math.log(p / q)
        total += kl

    return total / max(nb_runs, 1)

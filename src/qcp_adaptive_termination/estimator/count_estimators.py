"""Count estimators: estimate the total number of unique entities in the population."""

import math
import sys
from typing import assert_never

import numba
import numpy as np
from scipy.optimize import linprog
from scipy.stats import norm, poisson

from qcp_adaptive_termination.estimator.frequency import FrequencyStats
from qcp_adaptive_termination.models.estimation import (
    ConfidenceIntervalType,
    CountEstimator,
)


def estimate_count(
    stats: FrequencyStats,
    count_estimator: CountEstimator,
) -> float:
    """Estimated total unique entity count (observed + missing), no buckets."""
    if count_estimator is CountEstimator.CHAO92:
        estimated = _chao92(stats)
    elif count_estimator is CountEstimator.CHAO84:
        estimated = _chao84(stats)
    elif count_estimator is CountEstimator.JACKKNIFE1:
        estimated = _jackknife1(stats)
    elif count_estimator is CountEstimator.JACKKNIFE2:
        estimated = _jackknife2(stats)
    elif count_estimator is CountEstimator.UNSEEN:
        estimated = _unseen(stats)
    else:
        assert_never(count_estimator)

    return max(stats.observed_unique_entities, estimated)


@numba.njit(cache=True)
def chao92_scalar(
    unique_count: int,
    observation_count: int,
    singleton_count: int,
    pair_moment: int,
) -> float:  # type: ignore[return]
    """Chao92 unique-count estimate from raw partition statistics."""
    sample_coverage = 1.0 - singleton_count / observation_count
    if sample_coverage <= 0.0:
        return np.inf

    gamma_sq = (
        (unique_count / sample_coverage) * (pair_moment / (observation_count * (observation_count - 1))) - 1.0
        if (observation_count > 1 and pair_moment > 0)
        else 0.0
    )
    if gamma_sq < 0.0:
        gamma_sq = 0.0
    estimated_count = (
        unique_count / sample_coverage + observation_count * (1.0 - sample_coverage) / sample_coverage * gamma_sq
    )
    if estimated_count < unique_count:
        estimated_count = float(unique_count)
    return estimated_count


def _chao92(stats: FrequencyStats) -> float:
    """Chao92 estimated unique count from FrequencyStats."""
    sample_coverage = stats.sample_coverage()
    if sample_coverage == 0:
        return math.inf

    gamma_sq = stats.γ_sq()
    if not math.isfinite(gamma_sq):
        return math.inf

    estimated_count = (stats.observed_unique_entities / sample_coverage) + (
        stats.observed_total_entities * (1.0 - sample_coverage) / sample_coverage
    ) * gamma_sq
    return estimated_count


def _chao84_scalar(unique_count: int, singleton_count: int, doubleton_count: int) -> float:
    """Chao84 estimated unique count from raw partition statistics."""
    if doubleton_count == 0:
        return float(unique_count)
    return max(float(unique_count), unique_count + (singleton_count * singleton_count) / (2.0 * doubleton_count))


def _jackknife1_scalar(unique_count: int, singleton_count: int) -> float:
    """First-order jackknife estimated unique count from raw partition statistics."""
    return float(unique_count + singleton_count)


def _jackknife2_scalar(unique_count: int, singleton_count: int, doubleton_count: int) -> float:
    """Second-order jackknife estimated unique count from raw partition statistics."""
    return float(unique_count + 2 * singleton_count - doubleton_count)


def _chao84(stats: FrequencyStats) -> float:
    """Chao84 estimated unique count from FrequencyStats."""
    return _chao84_scalar(stats.observed_unique_entities, stats.singleton_count, stats.doubleton_count)


def _jackknife1(stats: FrequencyStats) -> float:
    """First-order jackknife estimated unique count from FrequencyStats."""
    return _jackknife1_scalar(stats.observed_unique_entities, stats.singleton_count)


def _jackknife2(stats: FrequencyStats) -> float:
    """Second-order jackknife estimated unique count from FrequencyStats."""
    return _jackknife2_scalar(stats.observed_unique_entities, stats.singleton_count, stats.doubleton_count)


def _pair_moment(stats: FrequencyStats) -> int:
    """Sum of count*(count-1) across all entities (used by Chao92's CV²)."""
    return sum((idx + 1) * idx * f for idx, f in enumerate(stats.frequencies))


def _unseen(stats: FrequencyStats) -> float:
    """Estimate total unique entity count via linear programming, falling back to chao92 on failure."""
    try:
        return _unseen_lp(stats)
    except Exception:
        return _chao92(stats)


def _unseen_lp(stats: FrequencyStats) -> float:
    """LP-based estimate of the total unique entity count."""
    grid_factor = 1.1
    alpha = 0.5

    xlp_min = 1.0 / (stats.observed_total_entities * max(10, stats.observed_total_entities))

    sparse_species = 0.0
    sparse_mass = 0.0
    f_lp_raw = [0.0] * len(stats.frequencies)
    for idx, freq_value in enumerate(stats.frequencies):
        if freq_value == 0:
            continue

        i = idx + 1
        radius = int(math.ceil(math.sqrt(i)))
        lo = max(0, idx - radius)
        hi = min(len(stats.frequencies) - 1, idx + radius)
        local_sum = sum(stats.frequencies[j] for j in range(lo, hi + 1))

        if local_sum < 2.0 * math.sqrt(i):
            sparse_species += freq_value
            sparse_mass += (i / stats.observed_total_entities) * freq_value
        else:
            f_lp_raw[idx] = float(freq_value)

    dense_positions = [i for i, v in enumerate(f_lp_raw) if v > 0.0]
    if not dense_positions:
        return max(float(stats.observed_unique_entities), sparse_species)

    f_max = max(dense_positions) + 1
    f_lp = f_lp_raw[:f_max] + [0.0] * int(math.ceil(math.sqrt(f_max)))

    xlp_max = f_max / stats.observed_total_entities
    if xlp_max <= xlp_min:
        return _chao92(stats)

    grid_size = int(math.ceil(math.log(xlp_max / xlp_min) / math.log(grid_factor))) + 1
    xlp = [xlp_min * (grid_factor**i) for i in range(max(grid_size, 2))]

    m = len(xlp)
    r = len(f_lp)
    coef = np.zeros((r, m), dtype=float)
    for i in range(r):
        k = i + 1
        for j, x in enumerate(xlp):
            coef[i, j] = poisson.pmf(k, stats.observed_total_entities * x) / x

    # Variables are [y_1..y_m, s_plus_1, s_minus_1, ..., s_plus_r, s_minus_r]
    # with y_j = x_j * h_j
    lp1_obj = np.zeros(m + 2 * r, dtype=float)
    for i, freq_value in enumerate(f_lp):
        w = 1.0 / math.sqrt(freq_value + 1.0)
        lp1_obj[m + 2 * i] = w
        lp1_obj[m + 2 * i + 1] = w

    a_ub = np.zeros((2 * r, m + 2 * r), dtype=float)
    b_ub = np.zeros(2 * r, dtype=float)
    for i in range(r):
        a_ub[2 * i, :m] = coef[i, :]
        a_ub[2 * i + 1, :m] = -coef[i, :]
        a_ub[2 * i, m + 2 * i] = -1.0
        a_ub[2 * i + 1, m + 2 * i + 1] = -1.0
        b_ub[2 * i] = f_lp[i]
        b_ub[2 * i + 1] = -f_lp[i]

    a_eq = np.zeros((1, m + 2 * r), dtype=float)
    a_eq[0, :m] = 1.0
    lp_mass = max(1e-12, 1.0 - sparse_mass)
    b_eq = np.array([lp_mass], dtype=float)

    bounds = [(0.0, None)] * (m + 2 * r)
    res1 = linprog(
        c=lp1_obj,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not res1.success:
        return _chao92(stats)

    lp2_obj = np.zeros(m + 2 * r, dtype=float)
    lp2_obj[:m] = 1.0 / np.array(xlp, dtype=float)

    a2_ub = np.vstack([a_ub, lp1_obj])
    b2_ub = np.concatenate([b_ub, np.array([res1.fun + alpha], dtype=float)])

    res2 = linprog(
        c=lp2_obj,
        A_ub=a2_ub,
        b_ub=b2_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not res2.success:
        return _chao92(stats)

    y = res2.x[:m]
    lp_species = float(sum(y_i / x_i for y_i, x_i in zip(y, xlp, strict=True)))
    return sparse_species + lp_species


# ── Confidence intervals ─────────────────────────────────────────────────────


def estimate_count_bounds(
    stats: FrequencyStats,
    confidence_interval_type: ConfidenceIntervalType,
    confidence_epsilon: float,
) -> tuple[float, float]:
    """Lower and upper bounds for the estimated total unique entity count."""
    if confidence_epsilon <= 0.0 or confidence_epsilon > 1.0:
        raise ValueError("confidence_epsilon must be in (0, 1]")

    if confidence_interval_type is ConfidenceIntervalType.CHAO84:
        lower_bound, upper_bound = _chao84_confidence_interval(stats, confidence_epsilon)
    elif confidence_interval_type is ConfidenceIntervalType.GOOD_TURING:
        lower_bound, upper_bound = _good_turing_confidence_interval(stats, confidence_epsilon)
    else:
        assert_never(confidence_interval_type)

    return max(stats.observed_unique_entities, lower_bound), max(stats.observed_unique_entities, upper_bound)


def _good_turing_confidence_interval(stats: FrequencyStats, confidence_epsilon: float) -> tuple[float, float]:
    """Confidence interval for total unique entity count using a Good-Turing error bound."""
    numeric_eps = sys.float_info.epsilon
    observation_count = stats.observed_total_entities
    singleton_count = stats.singleton_count
    unique_count = stats.observed_unique_entities

    missing_mass_hat = singleton_count / observation_count  # Good-Turing estimate of missing mass
    error = (2.0 * math.sqrt(2.0) + math.sqrt(3.0)) * math.sqrt(math.log(3.0 / confidence_epsilon) / observation_count)

    missing_mass_upper = min(1.0, missing_mass_hat + error)
    missing_mass_lower = max(0.0, missing_mass_hat - error)

    estimated_count_upper = unique_count / max(numeric_eps, 1.0 - missing_mass_upper)
    estimated_count_lower = unique_count / max(numeric_eps, 1.0 - missing_mass_lower)
    return estimated_count_lower, estimated_count_upper


def _chao84_confidence_interval(stats: FrequencyStats, confidence_epsilon: float) -> tuple[float, float]:
    """Confidence interval for total unique entity count using the asymmetric Chao84 interval."""
    singleton_count = stats.singleton_count
    doubleton_count = stats.doubleton_count
    unique_count = stats.observed_unique_entities

    # Chao84 is undefined without doubletons; its point estimate would collapse the interval to the observed count.
    if doubleton_count == 0 and singleton_count > 0:
        return _good_turing_confidence_interval(stats, confidence_epsilon)

    estimated_count = _chao84(stats)
    estimated_count_minus_observed = estimated_count - unique_count
    if estimated_count_minus_observed == 0:
        return (unique_count, unique_count)

    z_prime = float(norm.ppf(1.0 - confidence_epsilon / 2.0))

    variance = doubleton_count * (
        0.5 * (singleton_count / doubleton_count) ** 2
        + (singleton_count / doubleton_count) ** 3
        + 0.25 * (singleton_count / doubleton_count) ** 4
    )

    spread = math.exp(z_prime * math.sqrt(math.log(1.0 + variance / (estimated_count_minus_observed**2))))

    estimated_count_lower = unique_count + estimated_count_minus_observed / spread
    estimated_count_upper = unique_count + estimated_count_minus_observed * spread
    return estimated_count_lower, estimated_count_upper

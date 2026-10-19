"""Value estimators: estimate the attribute values of missing entities."""

import math
from typing import assert_never

from qcp_adaptive_termination.estimator.frequency import FrequencyStats
from qcp_adaptive_termination.models.estimation import ValueEstimator


def estimate_sum(stats: FrequencyStats, value_estimator: ValueEstimator, estimated_count: float) -> float:
    """
    Estimated total SUM including the contribution from missing entities.

    Returns inf if the delta is not finite.
    """
    d = estimate_delta(stats, value_estimator, estimated_count)
    return math.inf if not math.isfinite(d) else stats.observed_value_sum + d


def estimate_delta(
    stats: FrequencyStats,
    value_estimator: ValueEstimator,
    estimated_count: float,
) -> float:
    """
    Estimated SUM contribution of missing entities (Δ = φ_D - φ_K).

    ``estimated_count`` is the precomputed N_hat (observed + missing) so the
    caller can reuse a single count estimate across value estimators.
    """
    if value_estimator is ValueEstimator.NAIVE:
        return _naive_delta(stats, estimated_count)
    if value_estimator is ValueEstimator.FREQUENCY:
        return _frequency_delta(stats, estimated_count)
    if value_estimator is ValueEstimator.GOOD_TURING:
        return _good_turing_delta(stats)
    if value_estimator is ValueEstimator.FREQUENCY_F12:
        return _frequency_f12_delta(stats, estimated_count)
    assert_never(value_estimator)


def _naive_delta(stats: FrequencyStats, estimated_count: float) -> float:
    """
    Estimated SUM of missing entities using mean substitution.

    Assumes missing entities have the same average value as the observed ones:
    Δ = (φ_K / c) * (N_hat - c).
    """
    if not math.isfinite(estimated_count):
        return math.inf
    return (stats.observed_value_sum / stats.observed_unique_entities) * (
        estimated_count - stats.observed_unique_entities
    )


def _frequency_delta(stats: FrequencyStats, estimated_count: float) -> float:
    """
    Estimated SUM of missing entities using the average singleton value.

    Uses φ_f1 / f1 as the value estimate instead of the overall mean,
    since singletons are the best proxy for unseen entities.
    Returns 0 when there are no singletons.
    """
    singleton_count = stats.singleton_count
    if singleton_count == 0:
        return 0.0
    if not math.isfinite(estimated_count):
        return math.inf
    return (stats.observed_singleton_value_sum / singleton_count) * (estimated_count - stats.observed_unique_entities)


def _good_turing_delta(stats: FrequencyStats) -> float:
    """
    Estimated SUM of missing entities using Good-Turing frequency smoothing.

    Does not require a separate count estimator: derives Δ directly from
    the singleton sum and the ratio f1 / (n - f1), which is the Good-Turing
    estimate of the missing probability mass scaled to the observed aggregate.
    Returns 0 when there are no singletons or all observations are singletons.
    """
    singleton_count = stats.singleton_count
    if singleton_count == 0:
        return 0.0

    denom = stats.observed_total_entities - singleton_count
    return stats.observed_singleton_value_sum * stats.observed_unique_entities / denom if denom > 0 else 0.0


def _frequency_f12_delta(stats: FrequencyStats, estimated_count: float) -> float:
    """
    Estimated SUM of missing entities using the average singleton+doubleton value.

    Uses φ_f12 / (f1 + f2) as the value estimate, blending singletons and
    doubletons to reduce variance compared to singletons alone.
    Returns 0 when there are neither singletons nor doubletons.
    """
    doubleton_count = stats.doubleton_count
    denom = stats.singleton_count + doubleton_count
    if denom == 0:
        return 0.0
    if not math.isfinite(estimated_count):
        return math.inf
    return (stats.observed_singleton_doubleton_value_sum / denom) * (estimated_count - stats.observed_unique_entities)

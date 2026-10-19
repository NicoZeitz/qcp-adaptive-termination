"""Build `StepUncertainty` metadata from estimator statistics."""

from qcp_adaptive_termination.models import (
    AggregationType,
    EstimatorStatistics,
    StepUncertainty,
    StopReason,
)


def build_step_uncertainty(
    step_id: str,
    estimator_statistics: EstimatorStatistics | None,
    stop_reason: StopReason,
) -> StepUncertainty:
    """Record the uncertainty metadata for one step from its final estimator stats."""
    if estimator_statistics is None:
        return StepUncertainty(step_id=step_id, stop_reason=stop_reason)

    observed = estimator_statistics.observed_count
    n_hat = estimator_statistics.estimated_count
    n_lower = estimator_statistics.estimated_count_lower
    n_upper = estimator_statistics.estimated_count_upper

    completeness = observed / n_hat if observed and n_hat else None
    completeness_lower = observed / n_upper if observed and n_upper else None
    completeness_upper = observed / n_lower if observed and n_lower else None

    value_point: dict[str, float] = {}
    value_lower: dict[str, float] = {}
    value_upper: dict[str, float] = {}

    for agg in estimator_statistics.aggregation_types:
        if agg == AggregationType.COUNT:
            continue
        key = agg.value
        if agg == AggregationType.SUM:
            if estimator_statistics.estimated_value_sum is not None:
                value_point[key] = estimator_statistics.estimated_value_sum
            if estimator_statistics.estimated_value_sum_lower is not None:
                value_lower[key] = estimator_statistics.estimated_value_sum_lower
            if estimator_statistics.estimated_value_sum_upper is not None:
                value_upper[key] = estimator_statistics.estimated_value_sum_upper
        elif agg == AggregationType.AVG:
            if estimator_statistics.estimated_value_avg is not None:
                value_point[key] = estimator_statistics.estimated_value_avg
            if estimator_statistics.estimated_value_avg_lower is not None:
                value_lower[key] = estimator_statistics.estimated_value_avg_lower
            if estimator_statistics.estimated_value_avg_upper is not None:
                value_upper[key] = estimator_statistics.estimated_value_avg_upper
        elif agg == AggregationType.MIN and estimator_statistics.estimated_value_min is not None:
            value_point[key] = estimator_statistics.estimated_value_min
        elif agg == AggregationType.MAX and estimator_statistics.estimated_value_max is not None:
            value_point[key] = estimator_statistics.estimated_value_max
        elif agg == AggregationType.Q25 and estimator_statistics.estimated_value_q25 is not None:
            value_point[key] = estimator_statistics.estimated_value_q25
        elif agg == AggregationType.MEDIAN and estimator_statistics.estimated_value_median is not None:
            value_point[key] = estimator_statistics.estimated_value_median
        elif agg == AggregationType.Q75 and estimator_statistics.estimated_value_q75 is not None:
            value_point[key] = estimator_statistics.estimated_value_q75

    return StepUncertainty(
        step_id=step_id,
        completeness=completeness,
        completeness_lower=completeness_lower,
        completeness_upper=completeness_upper,
        observed_count=observed,
        estimated_count=n_hat,
        estimated_count_lower=n_lower,
        estimated_count_upper=n_upper,
        value_point=value_point,
        value_lower=value_lower,
        value_upper=value_upper,
        min_confirmed=estimator_statistics.estimated_value_min is not None,
        max_confirmed=estimator_statistics.estimated_value_max is not None,
        stop_reason=stop_reason,
    )

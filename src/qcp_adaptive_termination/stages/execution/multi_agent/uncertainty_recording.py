"""Build `StepUncertainty` metadata from the estimator at stop time."""

from qcp_adaptive_termination.estimator import Estimator
from qcp_adaptive_termination.models import (
    AggregationType,
    StepUncertainty,
    StopReason,
)


def build_step_uncertainty(
    step_id: str,
    estimator: Estimator | None,
    stop_reason: StopReason,
) -> StepUncertainty:
    """Record the uncertainty metadata for one step from its final estimator.

    Args:
        step_id: the step id
        estimator: the final estimator
        stop_reason: why execution stopped

    Returns:
        (StepUncertainty): the step's uncertainty metadata
    """
    if estimator is None:
        return StepUncertainty(step_id=step_id, stop_reason=stop_reason)

    observed = estimator.sample_count()
    n_hat = estimator.estimated_count()
    n_lower, n_upper = estimator.estimated_count_bounds()

    completeness = observed / n_hat if observed and n_hat else None
    completeness_lower = observed / n_upper if observed and n_upper else None
    completeness_upper = observed / n_lower if observed and n_lower else None

    value_point: dict[str, float] = {}
    value_lower: dict[str, float] = {}
    value_upper: dict[str, float] = {}

    for agg in AggregationType:
        if agg == AggregationType.COUNT:
            continue
        key = agg.value
        if agg == AggregationType.SUM:
            value_point[key] = estimator.estimated_sum()
            lo, hi = estimator.estimated_sum_bounds()
            value_lower[key] = lo
            value_upper[key] = hi
        elif agg == AggregationType.AVG:
            value_point[key] = estimator.estimated_avg()
            lo, hi = estimator.estimated_avg_bounds()
            value_lower[key] = lo
            value_upper[key] = hi
        elif agg == AggregationType.MIN:
            est_min = estimator.estimated_min()
            if est_min is not None:
                value_point[key] = est_min
        elif agg == AggregationType.MAX:
            est_max = estimator.estimated_max()
            if est_max is not None:
                value_point[key] = est_max
        elif agg == AggregationType.Q25:
            value_point[key] = estimator.estimated_quantile(0.25)
        elif agg == AggregationType.MEDIAN:
            value_point[key] = estimator.estimated_median()
        elif agg == AggregationType.Q75:
            value_point[key] = estimator.estimated_quantile(0.75)

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
        min_confirmed=estimator.estimated_min() is not None,
        max_confirmed=estimator.estimated_max() is not None,
        stop_reason=stop_reason,
    )

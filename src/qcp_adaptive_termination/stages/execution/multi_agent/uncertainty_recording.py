"""Build `StepUncertainty` metadata from the estimator at stop time."""

from qcp_adaptive_termination.estimator import Estimator
from qcp_adaptive_termination.models import (
    AggregationType,
    StepUncertainty,
    StopReason,
)


def build_step_uncertainty(
    estimator: Estimator | None,
    stop_reason: StopReason,
    aggregation_types: list[AggregationType],
) -> StepUncertainty:
    """Record the uncertainty metadata for one step from its final estimator.

    Args:
        estimator: the final estimator
        stop_reason: why execution stopped
        aggregation_types: the aggregation types declared in the aligned schema

    Returns:
        (StepUncertainty): the step's uncertainty metadata
    """
    if stop_reason == StopReason.ALL_AGENTS or (stop_reason == StopReason.LLM and estimator is None):
        # Hardcoded full completeness; no CI or estimation details to report.
        return StepUncertainty(completeness=1.0, stop_reason=stop_reason)

    assert estimator is not None, "estimator stops and LLM-with-estimator stops require an estimator"
    observed = estimator.sample_count()
    n_hat = estimator.estimated_count()
    n_lower, n_upper = estimator.estimated_count_bounds()

    completeness = observed / n_hat
    completeness_lower = observed / n_upper
    completeness_upper = observed / n_lower

    observed_value: dict[str, float] = {}
    estimated_value: dict[str, float] = {}
    estimated_value_lower: dict[str, float] = {}
    estimated_value_upper: dict[str, float] = {}

    for agg in aggregation_types:
        if agg == AggregationType.COUNT:
            continue
        key = agg.value
        if agg == AggregationType.SUM:
            observed_value[key] = estimator.sample_sum()
            estimated_value[key] = estimator.estimated_sum()
            lo, hi = estimator.estimated_sum_bounds()
            estimated_value_lower[key] = lo
            estimated_value_upper[key] = hi
        elif agg == AggregationType.AVG:
            observed_value[key] = estimator.sample_avg()
            estimated_value[key] = estimator.estimated_avg()
            lo, hi = estimator.estimated_avg_bounds()
            estimated_value_lower[key] = lo
            estimated_value_upper[key] = hi
        elif agg == AggregationType.MIN:
            observed_value[key] = estimator.sample_min()
            estimated_min = estimator.estimated_min()
            if estimated_min is not None:
                estimated_value[key] = estimated_min
        elif agg == AggregationType.MAX:
            observed_value[key] = estimator.sample_max()
            estimated_max = estimator.estimated_max()
            if estimated_max is not None:
                estimated_value[key] = estimated_max
        elif agg == AggregationType.Q25:
            observed_value[key] = estimator.sample_quantile(0.25)
            estimated_value[key] = estimator.estimated_quantile(0.25)
        elif agg == AggregationType.MEDIAN:
            observed_value[key] = estimator.sample_median()
            estimated_value[key] = estimator.estimated_median()
        elif agg == AggregationType.Q75:
            observed_value[key] = estimator.sample_quantile(0.75)
            estimated_value[key] = estimator.estimated_quantile(0.75)

    return StepUncertainty(
        completeness=completeness,
        completeness_lower=completeness_lower,
        completeness_upper=completeness_upper,
        observed_count=observed,
        estimated_count=n_hat,
        estimated_count_lower=n_lower,
        estimated_count_upper=n_upper,
        observed_value=observed_value,
        estimated_value=estimated_value,
        estimated_value_lower=estimated_value_lower,
        estimated_value_upper=estimated_value_upper,
        stop_reason=stop_reason,
    )

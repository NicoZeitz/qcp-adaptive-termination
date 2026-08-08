from enum import Enum


class EstimatorMode(Enum):
    FULL = "full"
    """Coverage, observed count + value, estimated count + value with intervals.

    Example:
        Aggregation type: sum
        Agents queried: 2 /  5 (40.00%)
        New unique entries added by this agent: 4
        Sample Coverage: 0.500
        Observed count: 6
        Estimated count: 13.000
        Estimated count confidence interval: [6.300, 59.358]
        Observed sum: 880.0
        Estimated sum: 1700.000
        Estimated sum confidence interval: [-555.879, 28043.966]
    """

    COUNT_ONLY = "count_only"
    """Coverage, observed count, estimated count with intervals. No value estimates.

    Example:
        Aggregation type: sum
        Agents queried: 2 /  5 (40.00%)
        New unique entries added by this agent: 4
        Sample Coverage: 0.500
        Observed count: 6
        Estimated count: 13.000
        Estimated count confidence interval: [6.300, 59.358]
    """

    VALUE_ONLY = "value_only"
    """Coverage, observed value, estimated value with intervals. No count estimates.

    Example:
        Aggregation type: sum
        Agents queried: 2 /  5 (40.00%)
        New unique entries added by this agent: 4
        Sample Coverage: 0.500
        Observed sum: 880.0
        Estimated sum: 1700.000
        Estimated sum confidence interval: [-555.879, 28043.966]
    """

    OBSERVED_ONLY = "observed_only"
    """Coverage, observed count and value. No estimates.

    Example:
        Aggregation type: sum
        Agents queried: 2 /  5 (40.00%)
        New unique entries added by this agent: 4
        Sample Coverage: 0.500
        Observed count: 6
        Observed sum: 880.0
    """

    COVERAGE_ONLY = "coverage_only"
    """Coverage and agent progress only. No counts or values.

    Example:
        Aggregation type: sum
        Agents queried: 2 /  5 (40.00%)
        New unique entries added by this agent: 4
        Sample Coverage: 0.500
    """

    NONE = "none"
    """No statistics injected — evaluator decides from the result list alone."""

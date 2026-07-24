from contextlib import contextmanager
from enum import Enum
from typing import Generator

from pydantic import BaseModel, ConfigDict

from qtypes.selection_type import SelectionType
from unknown_unknown import CountEstimator, Estimator, ValueEstimator


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


class ExecutionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    continue_after_final_result: bool = False

    selection_type: SelectionType = SelectionType.SEQUENTIAL
    align_schema_agent_limit: int = 1000

    estimator_mode: EstimatorMode = EstimatorMode.FULL
    count_estimator: CountEstimator = Estimator.DEFAULT_COUNT_ESTIMATOR
    value_estimator: ValueEstimator = Estimator.DEFAULT_VALUE_ESTIMATOR
    use_bucket: bool = True
    use_monte_carlo: bool = False
    include_confidence_intervals: bool = True

    _instance: "ExecutionConfig | None" = None

    @classmethod
    def set(cls, **kwargs) -> "ExecutionConfig":
        cls._instance = cls(**kwargs)
        return cls._instance

    @classmethod
    def get(cls) -> "ExecutionConfig":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    @contextmanager
    def override(cls, **kwargs) -> Generator["ExecutionConfig", None, None]:
        previous = cls._instance
        try:
            current = cls.get()
            cls._instance = current.model_copy(update=kwargs)
            yield cls._instance
        finally:
            cls._instance = previous

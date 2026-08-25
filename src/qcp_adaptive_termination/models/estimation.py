"""
Estimation data models.

Defines `EstimatorMode` (enum controlling what statistics are reported),
`EstimatorStatistics` (per-iteration stats snapshot), `AlignedResultList`
and `AlignedSchemaMetadata` (schema-aligned result accumulation).
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from qcp_adaptive_termination.models.result_type import ResultType, SerializedResultType


class AggregationType(StrEnum):
    COUNT = "count"
    MIN = "min"
    MAX = "max"
    SUM = "sum"
    AVG = "avg"


class EstimatorMode(StrEnum):
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


class AlignedSchemaMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    aggregation_types: list[AggregationType] = Field(..., min_length=1)
    keys: list[str] = Field(..., min_length=1)
    value_column: str | None = None
    columns: list[str] = Field(default_factory=list, min_length=1)
    source_agent_column: str


class AlignedResultList(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    data: SerializedResultType = ResultType()
    schema_metadata: AlignedSchemaMetadata


class EstimatorStatistics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_coverage: float | None = None

    new_unique_count: int

    agents_queried: int
    agents_total: int

    aggregation_types: list[AggregationType]

    observed_count: int
    estimated_count: float | None = None
    estimated_count_lower: float | None = None
    estimated_count_upper: float | None = None

    observed_value_sum: float | None = None
    observed_value_avg: float | None = None
    observed_value_min: float | None = None
    observed_value_max: float | None = None

    estimated_value_sum: float | None = None
    estimated_value_sum_lower: float | None = None
    estimated_value_sum_upper: float | None = None
    estimated_value_avg: float | None = None
    estimated_value_avg_lower: float | None = None
    estimated_value_avg_upper: float | None = None
    estimated_value_min: float | None = None
    estimated_value_max: float | None = None

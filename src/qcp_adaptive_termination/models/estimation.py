"""
Estimation data models.

Defines `EstimatorMode` (enum controlling what statistics are reported),
`EstimatorStatistics` (per-iteration stats snapshot), `AlignedResultList`
and `AlignedSchemaMetadata` (schema-aligned result accumulation).
"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict

from qcp_adaptive_termination.models.result import ResultType
from qcp_adaptive_termination.models.serializers import SerializedResultType

AggregationType = Literal["count", "min", "max", "sum", "avg"]


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


class AlignedSchemaMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    aggregation_type: AggregationType
    keys: list[str]
    value_column: str | None = None
    columns: list[str] = []
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

    aggregation_type: AggregationType

    observed_count: int
    estimated_count: float | None = None
    estimated_count_lower: float | None = None
    estimated_count_upper: float | None = None

    observed_value: float | None = None
    estimated_value: float | None = None
    estimated_value_lower: float | None = None
    estimated_value_upper: float | None = None

from pydantic import BaseModel, ConfigDict

from qtypes.aligned_result_list import AggregationType


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

from pydantic import BaseModel

from qtypes.aligned_result_list import AggregationType


class EstimatorStatistics(BaseModel):
    sample_coverage: float
    agents_queried: int
    agents_total: int
    aggregation_type: AggregationType
    observed_value: float | None = None
    estimated_value: float | None = None
    estimated_value_lower: float | None = None
    estimated_value_upper: float | None = None

from pydantic import BaseModel, ConfigDict

from qtypes.aligned_result_list import AggregationType


class EstimatorStatistics(BaseModel):
    model_config = ConfigDict(extra="forbid", ser_json_inf_nan="strings")

    sample_coverage: float
    agents_queried: int
    agents_total: int
    aggregation_type: AggregationType
    observed_value: float | None = None
    estimated_value: float | None = None
    estimated_value_lower: float | None = None
    estimated_value_upper: float | None = None

from pydantic import BaseModel

from qtypes.aligned_result_list import AggregationType
from unknown_unknown.estimator import Estimator


class EstimatorStatistics(BaseModel):
    estimator: Estimator
    sample_coverage: float
    agents_queried: int
    agents_total: int
    aggregation_type: AggregationType
    observed_value: float | None = None
    estimated_value: float | None = None
    estimated_value_lower: float | None = None
    estimated_value_upper: float | None = None

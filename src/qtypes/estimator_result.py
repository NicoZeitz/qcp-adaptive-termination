from dataclasses import dataclass

from qtypes.aligned_result_list import AggregationType


@dataclass
class EstimatorResult:
    sample_coverage: float
    agents_queried: int
    agents_total: int
    aggregation_type: AggregationType
    observed_count: int | None = None
    estimated_count: float | None = None
    estimated_count_lower: float | None = None
    estimated_count_upper: float | None = None
    observed_sum: float | None = None
    estimated_sum: float | None = None
    estimated_sum_lower: float | None = None
    estimated_sum_upper: float | None = None
    observed_avg: float | None = None
    estimated_avg: float | None = None
    estimated_avg_lower: float | None = None
    estimated_avg_upper: float | None = None
    estimated_min: float | None = None
    estimated_max: float | None = None

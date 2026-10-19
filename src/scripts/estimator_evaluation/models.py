"""Data models for estimator evaluation."""

from dataclasses import dataclass


@dataclass(slots=True)
class GroundTruth:
    count: int
    value_sum: float | None
    value_avg: float | None
    value_min: float | None
    value_max: float | None


@dataclass(slots=True)
class BatchRecord:
    batch_num: int
    cumulative_agents: int
    decision: str
    sample_coverage: float | None
    observed_count: int | None
    estimated_count: float | None
    observed_value_sum: float | None
    estimated_value_sum: float | None
    observed_value_avg: float | None
    estimated_value_avg: float | None
    coverage_achieved: float


@dataclass(slots=True)
class StepResult:
    step_id: str
    agents_total: int
    agents_queried_at_stop: int
    stop_fraction: float
    early_stop: bool
    ground_truth: GroundTruth
    batches: list[BatchRecord]

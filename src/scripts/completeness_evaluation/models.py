"""Data models for completeness evaluation."""

from dataclasses import dataclass

from qcp_adaptive_termination.models import AggregationType


@dataclass(slots=True)
class GroundTruth:
    """Ground truth for one step's answer (distinct count + value stats)."""

    count: int
    value_sum: float | None
    value_avg: float | None
    value_min: float | None
    value_max: float | None


@dataclass(slots=True)
class BatchRecord:
    """One completeness check, recorded after each queried agent within a step.

    Every per-step CSV row corresponds to one :class:`BatchRecord` (one row per
    agent queried). Dynamic per-check fields hold the estimator/LLM state at that
    check; step-level fields (``agents_total``, ``true_*``, ``result``, ...) are
    repeated on each row so the CSV is self-contained.
    """

    # -- per-check identity / outcome --
    batch_num: int
    cumulative_agents: int
    decision: str  # all_agents | estimator | llm | query_more_agents
    stop: bool
    agents_queried_fraction: float
    # -- aligned-result-list size --
    results_count: int | None
    unique_results_count: int | None
    # -- estimator state --
    sample_coverage: float | None
    observed_count: int | None
    estimated_count: float | None
    estimated_count_lower: float | None
    estimated_count_upper: float | None
    predicted_remaining: float | None
    realized_unseen: float | None
    c_worst: float | None
    c_middle: float | None
    c_best: float | None
    warmup_reached: bool
    # -- sub-decisions --
    estimator_decision: str | None  # stop | continue | warmup | n/a
    llm_decision: str | None  # stop | continue | not_called
    # -- values (primary aggregation) --
    aggregation: str | None
    observed_value: float | None
    estimated_value: float | None
    estimated_value_lower: float | None
    estimated_value_upper: float | None
    # -- calibration vs truth --
    coverage_achieved: float  # observed_unique / true_count
    true_count: int
    true_value: float | None
    observed_error: float | None
    estimated_error: float | None


@dataclass(slots=True)
class StepResult:
    """Result for one completeness evaluation step."""

    step_id: str
    agents_total: int
    agents_queried_at_stop: int
    stop_fraction: float
    early_stop: bool
    stop_reason: str
    ground_truth: GroundTruth
    checks: list[BatchRecord]
    aggregation: AggregationType | None
    step_budget_t: float | None
    produced_result: str | None
    produced_scalar: float | None
    result_rel_err_avg: float | None
    result_rel_err_sum: float | None

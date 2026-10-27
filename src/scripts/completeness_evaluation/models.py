"""Data models for completeness evaluation."""

from dataclasses import dataclass

from qcp_adaptive_termination.models import AggregationType


@dataclass(slots=True)
class GroundTruth:
    """Ground truth for one step's answer (distinct count + value stats)."""

    count: int
    """Distinct count of rows in the ground truth"""
    value_sum: float | None
    """Sum of the aggregated value column"""
    value_avg: float | None
    """Average of the aggregated value column"""
    value_min: float | None
    """Minimum of the aggregated value column"""
    value_max: float | None
    """Maximum of the aggregated value column"""


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
    """Index of the queried batch within the step"""
    cumulative_agents: int
    """Total agents queried up to this check"""
    decision: str  # all_agents | estimator | llm | query_more_agents
    """Which component decided the outcome (all_agents | estimator | llm | query_more_agents)"""
    stop: bool
    """Whether execution stopped at this check"""
    agents_queried_fraction: float
    """Fraction of the total agents queried"""
    # -- aligned-result-list size --
    results_count: int | None
    """Size of the aligned result list"""
    unique_results_count: int | None
    """Number of distinct entries in the result list"""
    # -- estimator state --
    sample_coverage: float | None
    """Sampled coverage observed by the estimator"""
    observed_count: int | None
    """Number of rows observed so far"""
    estimated_count: float | None
    """Estimated total distinct count"""
    estimated_count_lower: float | None
    """Lower bound of the estimated count"""
    estimated_count_upper: float | None
    """Upper bound of the estimated count"""
    predicted_remaining: float | None
    """Predicted number of remaining unseen rows"""
    realized_unseen: float | None
    """Realized number of unseen rows"""
    c_worst: float | None
    """Worst-case calibration coefficient"""
    c_middle: float | None
    """Middle calibration coefficient"""
    c_best: float | None
    """Best-case calibration coefficient"""
    estimator_active: bool
    """Whether the estimator is eligible to decide (started at estimator_start_agents)"""
    # -- sub-decisions --
    estimator_decision: str | None  # stop | continue | warmup | n/a
    """Estimator decision (stop | continue | warmup | n/a)"""
    llm_decision: str | None  # stop | continue | not_called
    """LLM decision (stop | continue | not_called)"""
    # -- values (primary aggregation) --
    aggregation: str | None
    """Primary aggregation type"""
    observed_value: float | None
    """Observed value of the primary aggregation"""
    estimated_value: float | None
    """Estimated value of the primary aggregation"""
    estimated_value_lower: float | None
    """Lower bound of the estimated value"""
    estimated_value_upper: float | None
    """Upper bound of the estimated value"""
    # -- calibration vs truth --
    coverage_achieved: float  # observed_unique / true_count
    """Observed unique rows over the true count"""
    true_count: int
    """Ground-truth distinct count"""
    true_value: float | None
    """Ground-truth aggregated value"""
    observed_error: float | None
    """Relative error of the observed result"""
    estimated_error: float | None
    """Relative error of the estimated result"""


@dataclass(slots=True)
class StepResult:
    """Result for one completeness evaluation step."""

    step_id: str
    """Unique identifier for the step"""
    agents_total: int
    """Total agents available in the step"""
    agents_queried_at_stop: int
    """Agents queried when execution stopped"""
    stop_fraction: float
    """Fraction of agents queried at stop"""
    early_stop: bool
    """Whether the step stopped early"""
    stop_reason: str
    """Why the step stopped"""
    ground_truth: GroundTruth
    """Ground-truth answer for the step"""
    checks: list[BatchRecord]
    """Per-agent completeness check records"""
    aggregation: AggregationType | None
    """Primary aggregation type used"""
    step_budget_t: float | None
    """Step execution budget in seconds"""
    produced_result: str | None
    """Final produced result string"""
    produced_scalar: float | None
    """Final produced scalar value"""
    result_rel_err_avg: float | None
    """Average relative result error"""
    result_rel_err_sum: float | None
    """Sum of relative result errors"""

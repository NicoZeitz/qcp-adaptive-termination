"""Data models for completeness evaluation."""

from dataclasses import dataclass, field

from qcp_adaptive_termination.models import AggregationType

# Target coverage levels at which we report the earliest possible stop point.
# Keys in per-check records and CSV columns use ``int(target * 100)`` (e.g. 95).
COVERAGE_TARGETS: tuple[float, ...] = (0.8, 0.9, 0.95, 0.99, 1.0)


@dataclass(slots=True)
class GroundTruth:
    """Ground truth for one step's answer (distinct count + value stats)."""

    count: int
    """Distinct count of rows in the ground truth"""
    value_count: int
    """Distinct rows with a non-missing value"""
    value_sum: float | None
    """Sum of the aggregated value column"""
    value_avg: float | None
    """Average of the aggregated value column"""
    value_min: float | None
    """Minimum of the aggregated value column"""
    value_max: float | None
    """Maximum of the aggregated value column"""
    value_median: float | None
    """Median of the aggregated value column"""


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
    # -- earliest possible stop analysis (step-level, repeated per row) --
    earliest_stop: dict[str, int | None] = field(default_factory=dict)
    """First agent index reaching each coverage target (key like ``95``); None if never reached"""
    agents_saved: dict[str, int | None] = field(default_factory=dict)
    """Agents queried beyond the earliest stop point for each coverage target"""
    earliest_scalar_stop_agents: int | None = None
    """First agent index where the observed scalar equals the true scalar (value steps)"""
    llm_status: str | None = None
    """LLM outcome (stop | continue | undecided) or why it was not called"""
    long_list: bool = False
    """Whether the deduplicated result list exceeded the long-list threshold"""
    observed_by_aggregation: dict[str, float | None] = field(default_factory=dict)
    """Observed value for every aggregation type of the step"""
    e2e_true_count: int | None = None
    """End-to-end ground-truth distinct count (all previous steps complete)"""
    e2e_true_value: float | None = None
    """End-to-end ground-truth aggregated value"""
    e2e_coverage_achieved: float | None = None
    """Observed rows that belong to the end-to-end truth over the end-to-end true count"""
    e2e_observed_error: float | None = None
    """Relative error of the observed result against the end-to-end truth"""


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
    """Ground-truth answer for the step given the actual previous step results"""
    e2e_ground_truth: GroundTruth
    """Ground-truth answer for the step assuming all previous steps were complete"""
    step_sql: str
    """SQL used for the step's agents (compounded or canonical)"""
    compounded: bool
    """Whether the step's filter was rewritten from previous step results"""
    compound_source: str | None
    """Where the compounded filter values came from (produced | observed | none)"""
    checks: list[BatchRecord]
    """Per-agent completeness check records"""
    aggregation: AggregationType | None
    """Primary aggregation type used"""
    step_budget_t: float | None
    """Step completeness threshold"""
    produced_result: str | None
    """Final produced result string"""
    produced_scalar: float | None
    """Produced value of the primary aggregation (scalar answer, or the list answer aggregated)"""
    produced_value_source: str | None
    """How the produced value was obtained (scalar | aggregated)"""
    produced_rel_err: float | None
    """Relative error of the produced scalar against the primary aggregation truth"""
    e2e_produced_rel_err: float | None
    """Relative error of the produced scalar against the end-to-end truth"""
    produced_precision: float | None
    """Fraction of produced keys that are in the ground truth (list results)"""
    produced_recall: float | None
    """Fraction of ground-truth keys present in the produced result (list results)"""
    produced_used_estimator: bool
    """Whether the result producer was given the estimator"""

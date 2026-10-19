"""
Deterministic completeness decision based on the estimator.

Implements the worst-case completeness stop rule from the plan:
stop when `observed / N_hat_upper(epsilon_step) >= t_step`, where
`epsilon_step = 1 - t_step` and `t_step` comes from the step's Lusser budget.
"""

import math

from qcp_adaptive_termination.models import EstimatorStatistics, StepBudget
from qcp_adaptive_termination.models.execution import EstimatorDecision
from qcp_adaptive_termination.observability import context, tracing


def run_deterministic_estimator(
    estimator_statistics: EstimatorStatistics,
    step_budget: StepBudget | None,
    agents_queried: int,
) -> EstimatorDecision:
    """
    Decide whether to stop based on worst-case completeness.

    Returns an `EstimatorDecision` carrying the stop flag and the estimator
    statistics (so downstream hooks and the result-producing agent can use them).
    """
    config = context.get_config()

    # Warm-up: never stop deterministically before enough agents have been queried.
    if agents_queried <= config.warmup_agents:
        tracing.indented_logging(f"Estimator: warm-up ({agents_queried} <= {config.warmup_agents} agents) — continue")
        return EstimatorDecision(stop=False, estimator_statistics=estimator_statistics)

    # No budget (e.g. find queries or no multi-agent budget) → no deterministic stop.
    if step_budget is None:
        return EstimatorDecision(stop=False, estimator_statistics=estimator_statistics)

    observed = estimator_statistics.observed_count
    if observed == 0:
        return EstimatorDecision(stop=False, estimator_statistics=estimator_statistics)

    # Worst-case completeness: observed / N_hat_upper(epsilon_step).
    upper = estimator_statistics.estimated_count_upper
    if upper is None or not math.isfinite(upper) or upper <= 0:
        return EstimatorDecision(stop=False, estimator_statistics=estimator_statistics)

    c_worst = observed / upper
    stop = c_worst >= step_budget.t_step

    tracing.indented_logging(
        f"Estimator: C_worst={c_worst:.4f} (observed={observed}, N_hat_upper={upper:.2f}) "
        f"threshold={step_budget.t_step:.4f} → {'STOP' if stop else 'continue'}"
    )
    return EstimatorDecision(stop=stop, estimator_statistics=estimator_statistics)

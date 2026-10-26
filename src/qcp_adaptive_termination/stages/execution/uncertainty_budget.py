"""Uncertainty budget computation for a query plan."""

import math
from typing import assert_never

from qcp_adaptive_termination.models import ApportionmentType, StepBudget
from qcp_adaptive_termination.models.query_plan import (
    MultiAgentQueryPlanStep,
    QueryPlan,
)


def compute_step_budgets(
    query_plan: QueryPlan,
    target_confidence: float,
    apportionment: ApportionmentType,
) -> dict[str, StepBudget]:
    """Compute per-step completeness thresholds for a query plan.

    Args:
        query_plan: the query plan to budget
        target_confidence: the global confidence target
        apportionment: how to split the target across steps

    Returns:
        (dict[str, StepBudget]): per-step budgets keyed by step id
    """
    multi_steps = [s for s in query_plan.steps if isinstance(s, MultiAgentQueryPlanStep)]
    if not multi_steps:
        return {}

    if apportionment is ApportionmentType.EQUAL:
        t_step = target_confidence ** (1.0 / len(multi_steps))
        return {step.id: StepBudget(step_id=step.id, t_step=t_step) for step in multi_steps}

    if apportionment is ApportionmentType.WEIGHTED:
        weights = [math.log(len(step.to_agent)) for step in multi_steps]
        total_weight = sum(weights)

        budgets: dict[str, StepBudget] = {}
        for step, weight in zip(multi_steps, weights, strict=True):
            t_step = target_confidence ** (weight / total_weight)
            budgets[step.id] = StepBudget(step_id=step.id, t_step=t_step)

        return budgets

    assert_never(apportionment)

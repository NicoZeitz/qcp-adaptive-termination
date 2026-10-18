"""
Uncertainty budget models.

Defines `ApportionmentType` (how the global confidence threshold is split
across multi-agent steps) and `StepBudget` (per-step completeness threshold).
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ApportionmentType(StrEnum):
    WEIGHTED = "weighted"
    """Weight each step by ln(number of agents); bigger steps get more budget."""

    EQUAL = "equal"
    """Equal apportionment: t_step = global^(1/n_steps)."""


class StepBudget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: str
    t_step: float = Field(..., gt=0, lt=1)
    """Completeness threshold for this step: stop when C_worst >= t_step.

    The confidence error probability for the step is derived as
    epsilon = 1 - t_step at the point of use (the deterministic stop rule).
    """

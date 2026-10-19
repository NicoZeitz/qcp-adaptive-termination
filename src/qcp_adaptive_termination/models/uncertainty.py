"""
Uncertainty models.

Defines `ApportionmentType` (how the global confidence threshold is split
across multi-agent steps), `StepBudget` (per-step completeness threshold),
and `StepUncertainty` (per-step recorded uncertainty metadata for synthesis).
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
    """
    Completeness threshold for this step: stop when C_worst >= t_step.

    The confidence error probability for the step is derived as
    epsilon = 1 - t_step at the point of use (the deterministic stop rule).
    """


class StopReason(StrEnum):
    ESTIMATOR = "estimator"
    LLM = "llm"
    ALL_AGENTS = "all_agents"
    NEED = "need"


class StepUncertainty(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: str
    """The query plan step this uncertainty belongs to."""

    completeness: float | None = None
    """observed / N_hat at stop time (None if not estimable)."""

    completeness_lower: float | None = None
    """observed / N_hat_upper at stop time (worst-case completeness)."""

    completeness_upper: float | None = None
    """observed / N_hat_lower at stop time (best-case completeness)."""

    observed_count: int | None = None
    estimated_count: float | None = None
    estimated_count_lower: float | None = None
    estimated_count_upper: float | None = None

    value_point: dict[str, float] = Field(default_factory=dict)
    """Per-aggregation point estimate (e.g. {"sum": 1300.0})."""

    value_lower: dict[str, float] = Field(default_factory=dict)
    """Per-aggregation lower CI bound."""

    value_upper: dict[str, float] = Field(default_factory=dict)
    """Per-aggregation upper CI bound."""

    min_confirmed: bool | None = None
    """True if the observed min is confirmed (no missing entity below it)."""

    max_confirmed: bool | None = None
    """True if the observed max is confirmed (no missing entity above it)."""

    stop_reason: StopReason | None = None
    """How this step stopped: estimator, llm, all_agents, or need."""

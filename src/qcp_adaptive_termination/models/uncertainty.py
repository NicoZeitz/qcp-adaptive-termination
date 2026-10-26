"""
Uncertainty models.

Defines `ApportionmentType` (how the global confidence threshold is split
across multi-agent steps), `StepBudget` (per-step completeness threshold),
and `StepUncertainty` (per-step recorded uncertainty metadata for synthesis).
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ApportionmentType(StrEnum):
    """How the global confidence threshold is split across steps."""

    WEIGHTED = "weighted"
    """Weight each step by ln(number of agents); bigger steps get more budget."""

    EQUAL = "equal"
    """Equal apportionment: t_step = global^(1/n_steps)."""


class StepBudget(BaseModel):
    """Per-step completeness threshold derived from the global budget."""

    model_config = ConfigDict(extra="forbid")

    step_id: str = Field(description="The query plan step this budget belongs to")
    t_step: float = Field(
        ...,
        gt=0,
        lt=1,
        description="Completeness threshold for this step: stop when C_worst >= t_step",
    )


class StopReason(StrEnum):
    """The reason a query plan step stopped."""

    ESTIMATOR = "estimator"
    LLM = "llm"
    ALL_AGENTS = "all_agents"


class StepUncertainty(BaseModel):
    """Recorded uncertainty metadata for a single query plan step."""

    model_config = ConfigDict(extra="forbid")

    step_id: str = Field(description="The query plan step this uncertainty belongs to")

    completeness: float | None = Field(
        default=None, description="observed / N_hat at stop time (None if not estimable)"
    )

    completeness_lower: float | None = Field(
        default=None,
        description="observed / N_hat_upper at stop time (worst-case completeness)",
    )

    completeness_upper: float | None = Field(
        default=None,
        description="observed / N_hat_lower at stop time (best-case completeness)",
    )

    observed_count: int | None = Field(default=None, description="Observed entity count at stop time")
    estimated_count: float | None = Field(default=None, description="Estimated total entity count")
    estimated_count_lower: float | None = Field(default=None, description="Lower bound of the estimated count")
    estimated_count_upper: float | None = Field(default=None, description="Upper bound of the estimated count")

    value_point: dict[str, float] = Field(default_factory=dict, description="Per-aggregation point estimate")
    value_lower: dict[str, float] = Field(default_factory=dict, description="Per-aggregation lower CI bound")
    value_upper: dict[str, float] = Field(default_factory=dict, description="Per-aggregation upper CI bound")

    min_confirmed: bool | None = Field(
        default=None,
        description="True if the observed min is confirmed (no missing entity below it)",
    )
    max_confirmed: bool | None = Field(
        default=None,
        description="True if the observed max is confirmed (no missing entity above it)",
    )
    stop_reason: StopReason | None = Field(
        default=None, description="How this step stopped: estimator, llm, or all_agents"
    )

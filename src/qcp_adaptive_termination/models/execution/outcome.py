"""Outcome models for a query plan step's answer logic."""

from pydantic import BaseModel, ConfigDict, Field

from qcp_adaptive_termination.models import SerializedNeedList
from qcp_adaptive_termination.models.result_type import (
    SerializedOptionalResultType,
)
from qcp_adaptive_termination.models.uncertainty import StopReason


class RaiseNeedOutcome(BaseModel):
    """Outcome requesting additional information from the query execution."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    needs: SerializedNeedList = Field(default=[], description="What information is needed to proceed.")
    partial_answer: SerializedOptionalResultType = Field(
        default=None, description="Partial answer produced before raising the need."
    )
    has_partial_answer: bool = Field(default=False, description="Whether a partial answer is available.")


class StopDecision(BaseModel):
    """
    Completeness evaluator decided to stop.

    Carries the stop reason; the caller is responsible for producing the final
    result (via the result-producing agent).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    stop_reason: StopReason = Field(description="Why the completeness evaluator stopped.")

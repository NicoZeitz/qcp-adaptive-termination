from pydantic import BaseModel, ConfigDict

from qcp_adaptive_termination.models import SerializedNeedList
from qcp_adaptive_termination.models.result_type import (
    SerializedOptionalResultType,
)
from qcp_adaptive_termination.models.uncertainty import StopReason


class RaiseNeedOutcome(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    needs: SerializedNeedList = []
    partial_answer: SerializedOptionalResultType = None
    has_partial_answer: bool = False


class StopDecision(BaseModel):
    """
    Completeness evaluator decided to stop.

    Carries the stop reason; the caller is responsible for producing the final
    result (via the result-producing agent).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    stop_reason: StopReason

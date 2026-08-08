from typing import Literal

from qcp_adaptive_termination.models.estimation import (
    AlignedResultList,
)
from qcp_adaptive_termination.models.serializers import (
    SerializedNeedList,
    SerializedOptionalResultType,
    SerializedResultType,
)

from pydantic import BaseModel, ConfigDict


class RaiseNeedOutcome(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    needs: SerializedNeedList = []
    partial_answer: SerializedOptionalResultType = None
    has_partial_answer: bool = False


EvaluatorDecision = Literal["final_result", "query_more_agents"]


class FinalResultOutcome(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    result: SerializedResultType
    result_list: AlignedResultList
    decision: EvaluatorDecision = "final_result"


class QueryMoreAgentsOutcome(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    result_list: AlignedResultList
    decision: EvaluatorDecision = "query_more_agents"

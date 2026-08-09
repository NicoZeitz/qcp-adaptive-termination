from typing import Literal

from pydantic import BaseModel, ConfigDict

from qcp_adaptive_termination.models import need_type, result_type
from qcp_adaptive_termination.models.need_type import SerializedNeedList
from qcp_adaptive_termination.models.result_type import SerializedOptionalResultType, SerializedResultType
from qcp_adaptive_termination.models.estimation import (
    AlignedResultList,
)


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

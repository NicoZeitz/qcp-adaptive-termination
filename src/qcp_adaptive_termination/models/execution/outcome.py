from typing import Literal

from pydantic import BaseModel, ConfigDict

from qcp_adaptive_termination.models import SerializedNeedList
from qcp_adaptive_termination.models.estimation import (
    AlignedResultList,
    EstimatorStatistics,
)
from qcp_adaptive_termination.models.result_type import (
    SerializedOptionalResultType,
    SerializedResultType,
)
from qcp_adaptive_termination.models.uncertainty import StopReason


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
    stop_reason: StopReason = StopReason.ESTIMATOR


class QueryMoreAgentsOutcome(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    result_list: AlignedResultList
    decision: EvaluatorDecision = "query_more_agents"


class EstimatorDecision(BaseModel):
    """
    Outcome of the deterministic completeness estimator.

    Carries the decision (stop / continue) plus the estimator statistics, so
    downstream hooks and the result-producing agent can use them.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    stop: bool
    estimator_statistics: EstimatorStatistics | None = None

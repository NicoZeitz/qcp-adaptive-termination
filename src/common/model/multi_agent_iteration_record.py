from typing import Literal

from pydantic import BaseModel, ConfigDict

from common.model.estimation.estimator_statistics import EstimatorStatistics
from common.model.execution.outcome import EvaluatorDecision
from common.model.serializers import SerializedResultType

IterationDecision = EvaluatorDecision | Literal["batch_query"]


class MultiAgentIterationRecord(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    step_id: str
    agent_name: str
    result_list: SerializedResultType
    estimator_result: EstimatorStatistics | None
    evaluator_decision: IterationDecision | None
    queried_agents: list[str] = []
    remaining_agents: list[str] = []

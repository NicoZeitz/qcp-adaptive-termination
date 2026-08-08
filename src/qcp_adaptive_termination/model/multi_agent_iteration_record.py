from typing import Literal

from pydantic import BaseModel, ConfigDict

from qcp_adaptive_termination.model.estimation.estimator_statistics import (
    EstimatorStatistics,
)
from qcp_adaptive_termination.model.execution.outcome import EvaluatorDecision
from qcp_adaptive_termination.model.serializers import SerializedResultType


class MultiAgentIterationRecord(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    step_id: str
    agent_name: str
    result_list: SerializedResultType
    estimator_result: EstimatorStatistics | None
    evaluator_decision: EvaluatorDecision | Literal["batch_query"] | None
    queried_agents: list[str] = []
    remaining_agents: list[str] = []

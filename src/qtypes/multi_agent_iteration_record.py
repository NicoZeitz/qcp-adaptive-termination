from pydantic import BaseModel, ConfigDict

from qtypes.estimator_statistics import EstimatorStatistics
from qtypes.outcome import EvaluatorDecision
from qtypes.serializers import SerializedResultType


class MultiAgentIterationRecord(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    step_id: str
    agent_name: str
    result_list: SerializedResultType
    estimator_result: EstimatorStatistics | None
    evaluator_decision: EvaluatorDecision | None
    queried_agents: list[str] = []
    remaining_agents: list[str] = []

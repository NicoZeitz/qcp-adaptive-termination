from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, JsonValue, field_serializer, field_validator

from qtypes.estimator_statistics import EstimatorStatistics
from qtypes.result import deserialize_result, serialize_result

EvaluatorDecision = Literal["final_result", "query_more_agents"]


class MultiAgentIterationRecord(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    step_id: str
    agent_name: str
    result_list: pd.DataFrame
    estimator_result: EstimatorStatistics | None
    evaluator_decision: EvaluatorDecision | None
    evaluator_next_agents: list[str] = []

    @field_serializer("result_list")
    def serialize_result_list(self, result_list: pd.DataFrame) -> JsonValue:
        return serialize_result(result_list)

    @field_validator("result_list", mode="before")
    @classmethod
    def validate_result_list(
        cls: type["MultiAgentIterationRecord"], value: Any
    ) -> pd.DataFrame:
        if isinstance(value, pd.DataFrame):
            return value
        elif isinstance(value, dict):
            return deserialize_result(value)
        return value

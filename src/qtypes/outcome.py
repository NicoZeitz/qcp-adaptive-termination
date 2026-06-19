from qtypes.aligned_result_list import AlignedResultList
from qtypes.multi_agent_iteration_record import EvaluatorDecision
from qtypes.need import NeedType, deserialize_need, serialize_need
from qtypes.result import ResultType, deserialize_result, serialize_result


from pydantic import BaseModel, ConfigDict, JsonValue, field_serializer, field_validator


from typing import Any


class RaiseNeedOutcome(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    needs: list[NeedType] = []
    partial_answer: ResultType | None = None
    has_partial_answer: bool = False

    @field_serializer("partial_answer")
    def serialize_answer(self, partial_answer: ResultType | None) -> JsonValue:
        if partial_answer is None:
            return None
        return serialize_result(partial_answer)

    @field_validator("partial_answer", mode="before")
    @classmethod
    def validate_partial_answer(
        cls: type["RaiseNeedOutcome"], value: Any
    ) -> ResultType:
        if value is None:
            return value
        elif isinstance(value, ResultType):
            return value
        elif isinstance(value, dict):
            return deserialize_result(value)
        return value

    @field_serializer("needs")
    def serialize_needs(self, needs: list[NeedType]) -> list[str]:
        return [serialize_need(need) for need in needs]

    @field_validator("needs", mode="before")
    @classmethod
    def validate_needs(cls: type["RaiseNeedOutcome"], value: Any) -> list[NeedType]:
        if isinstance(value, list):
            return [deserialize_need(need) for need in value]
        raise ValueError(f"Needs must be a list, got {type(value)}: {value}")


class FinalResultOutcome(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    result: ResultType
    decision: EvaluatorDecision = "final_result"

    @field_serializer("result")
    def serialize_result(self, result: ResultType) -> JsonValue:
        return serialize_result(result)

    @field_validator("result", mode="before")
    @classmethod
    def validate_result(cls: type["FinalResultOutcome"], value: Any) -> ResultType:
        if isinstance(value, ResultType):
            return value
        elif isinstance(value, dict):
            return deserialize_result(value)
        return value


class QueryMoreAgentsOutcome(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    result_list: AlignedResultList
    decision: EvaluatorDecision = "query_more_agents"

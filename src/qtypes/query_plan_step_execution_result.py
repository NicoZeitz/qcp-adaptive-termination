from abc import ABC
from typing import Any

from pydantic import BaseModel, ConfigDict, JsonValue, field_serializer, field_validator
from qtypes.need import NeedType, deserialize_need, serialize_need
from qtypes.query_plan_step import QueryPlanStep
from qtypes.result import ResultType, deserialize_result, serialize_result


class QueryPlanStepExecutionResult(ABC, BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    success: bool
    step: QueryPlanStep


class SuccessfulExecution(QueryPlanStepExecutionResult, BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    success: bool = True
    result: ResultType
    execution_finished: bool = False

    @field_serializer("result")
    def serialize_result(self, result: ResultType) -> JsonValue:
        return serialize_result(result)

    @field_validator("result", mode="before")
    @classmethod
    def validate_result(cls: type["SuccessfulExecution"], value: Any) -> ResultType:
        if isinstance(value, ResultType):
            return value
        elif isinstance(value, dict):
            return deserialize_result(value)
        return value


class FailedExecution(QueryPlanStepExecutionResult, BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    success: bool = False
    needs: list[NeedType]
    partial_answer: ResultType | None = None
    has_partial_answer: bool = False

    @field_serializer("partial_answer")
    def serialize_partial_answer(self, partial_answer: ResultType | None) -> JsonValue:
        if partial_answer is None:
            return None
        return serialize_result(partial_answer)

    @field_validator("partial_answer", mode="before")
    @classmethod
    def validate_partial_answer(
        cls: type["FailedExecution"], value: Any
    ) -> ResultType | None:
        if value is None:
            return None
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
    def validate_needs(cls: type["FailedExecution"], value: Any) -> list[NeedType]:
        if isinstance(value, list):
            return [deserialize_need(need) for need in value]
        raise ValueError(f"Needs must be a list, got {type(value)}: {value}")

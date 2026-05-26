from typing import Any

from pydantic import BaseModel, ConfigDict, JsonValue, field_serializer, field_validator

from qtypes.query_plan_step_execution_result import FailedExecution
from qtypes.result import ResultType, deserialize_result, serialize_result


class ExecutionResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    success: bool
    result: ResultType | None = None
    failed_execution: FailedExecution | None = None

    @field_serializer("result")
    def serialize_result(self, result: ResultType | None) -> JsonValue:
        if result is None:
            return None
        return serialize_result(result)

    @field_validator("result", mode="before")
    @classmethod
    def validate_result(cls: type["ExecutionResult"], value: Any) -> ResultType:
        if isinstance(value, dict):
            return deserialize_result(value)
        return value

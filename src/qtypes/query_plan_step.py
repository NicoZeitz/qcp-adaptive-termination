from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, JsonValue, field_serializer, field_validator

from qtypes.query_dependency import QueryDependency
from qtypes.result import ResultType, deserialize_result, serialize_result


class QueryPlanStep(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    id: str
    query: str
    to_agent: Literal["Coordinator"] | str
    dependencies: list[QueryDependency]
    answer: ResultType | None

    @field_serializer("answer")
    def serialize_answer(self, answer: ResultType | None) -> JsonValue:
        if answer is None:
            return None

        return serialize_result(answer)

    @field_validator("answer", mode="before")
    @classmethod
    def validate_answer(cls: type["QueryPlanStep"], value: Any) -> ResultType | None:
        if value is None:
            return None

        if isinstance(value, dict):
            return deserialize_result(value)
        return value

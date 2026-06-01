from typing import Any, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_serializer,
    field_validator,
)

from qtypes.query_dependency import QueryDependency
from qtypes.result import ResultType, deserialize_result, serialize_result


class BaseQueryPlanStep(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    id: str
    query: str
    dependencies: list[QueryDependency]
    answer: ResultType | None

    @field_serializer("answer")
    def serialize_answer(self, answer: ResultType | None) -> JsonValue:
        if answer is None:
            return None

        return serialize_result(answer)

    @field_validator("answer", mode="before")
    @classmethod
    def validate_answer(
        cls: type["BaseQueryPlanStep"], value: Any
    ) -> ResultType | None:
        if value is None:
            return None

        if isinstance(value, dict):
            return deserialize_result(value)
        return value


class SingleAgentQueryPlanStep(BaseQueryPlanStep):
    to_agent: str

    @property
    def has_multiple_to_agents(self) -> bool:
        return False

    @property
    def to_agents(self) -> list[str]:
        return [self.to_agent]


class MultiAgentQueryPlanStep(BaseQueryPlanStep):
    to_agent: list[str] = Field(min_length=2)
    merge_task: str

    @property
    def has_multiple_to_agents(self) -> bool:
        return True

    @property
    def to_agents(self) -> list[str]:
        return self.to_agent


QueryPlanStep: TypeAlias = SingleAgentQueryPlanStep | MultiAgentQueryPlanStep

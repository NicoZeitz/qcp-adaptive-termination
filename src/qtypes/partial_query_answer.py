from typing import Any, Self

from pydantic import BaseModel, ConfigDict, JsonValue, field_serializer, field_validator

from qtypes.need import NeedType, deserialize_need, serialize_need
from qtypes.result import ResultType, deserialize_result, serialize_result


class PartialQueryAnswerDummy(BaseModel):
    pass


class PartialQueryAnswer(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    answer: ResultType | None = None
    has_answer: bool = False
    needs: list[NeedType] = []

    def set_answer(self, answer: ResultType) -> Self:
        self.answer = answer
        self.has_answer = True
        return self

    def add_needs(self, needs: set[NeedType] | list[NeedType]) -> Self:
        self.needs = list(set(self.needs).union(needs))
        return self

    @property
    def has_needs(self) -> bool:
        return len(self.needs) > 0

    @property
    def is_final(self) -> bool:
        return self.has_answer or self.has_needs

    @field_serializer("answer")
    def serialize_answer(self, answer: ResultType | None) -> JsonValue:
        if answer is None:
            return None
        return serialize_result(answer)

    @field_validator("answer", mode="before")
    @classmethod
    def validate_answer(cls: type["PartialQueryAnswer"], value: Any) -> ResultType:
        if isinstance(value, ResultType):
            return value
        elif isinstance(value, dict):
            return deserialize_result(value)
        return value

    @field_serializer("needs")
    def serialize_needs(self, needs: list[NeedType]) -> list[str]:
        return [serialize_need(need) for need in needs]

    @field_validator("needs", mode="before")
    @classmethod
    def validate_needs(cls: type["PartialQueryAnswer"], value: Any) -> list[NeedType]:
        if isinstance(value, list):
            return [deserialize_need(need) for need in value]
        raise ValueError(f"Needs must be a list, got {type(value)}: {value}")

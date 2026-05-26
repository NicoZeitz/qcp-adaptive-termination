from qtypes.result import ResultType, deserialize_result, serialize_result


from pydantic import BaseModel, ConfigDict, field_serializer, field_validator


from typing import Any


class CollaborativeQueryAnsweringResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    result: ResultType

    @field_serializer("result")
    def serialize_result(self, result: ResultType) -> dict[str, Any]:
        return serialize_result(result)

    @field_validator("result", mode="before")
    @classmethod
    def validate_result(
        cls: type["CollaborativeQueryAnsweringResult"], value: Any
    ) -> ResultType:
        if isinstance(value, dict):
            return deserialize_result(value)
        return value

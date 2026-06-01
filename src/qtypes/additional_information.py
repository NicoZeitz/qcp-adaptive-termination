from typing import Any

from pydantic import BaseModel, ConfigDict, field_serializer, field_validator

from qtypes.result import ResultType, deserialize_result, serialize_result


class AdditionalInformation(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    step_id: str
    dependency: str
    data: ResultType

    @field_serializer("data")
    def serialize_data(self, data: ResultType) -> dict[str, Any]:
        return serialize_result(data)

    @field_validator("data", mode="before")
    @classmethod
    def validate_data(cls: type["AdditionalInformation"], value: Any) -> ResultType:
        if isinstance(value, dict):
            return deserialize_result(value)
        return value

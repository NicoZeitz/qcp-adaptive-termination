from typing import Literal, Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, JsonValue, field_serializer, field_validator

from qtypes.result import deserialize_result, serialize_result

AggregationType = Literal["count", "min", "max", "sum", "avg"]


class AlignedSchemaMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    aggregation_type: AggregationType
    keys: list[str]
    value_column: str | None = None


class AlignedResultList(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    data: pd.DataFrame = pd.DataFrame()
    schema: AlignedSchemaMetadata | None = None

    @field_serializer("data")
    def serialize_data(self, data: pd.DataFrame) -> JsonValue:
        return serialize_result(data)

    @field_validator("data", mode="before")
    @classmethod
    def validate_data(cls: type["AlignedResultList"], value: Any) -> pd.DataFrame:
        if isinstance(value, pd.DataFrame):
            return value
        elif isinstance(value, dict):
            return deserialize_result(value)
        return value

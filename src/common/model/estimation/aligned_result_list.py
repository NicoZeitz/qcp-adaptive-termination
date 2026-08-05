from typing import Literal

from pydantic import BaseModel, ConfigDict

from common.model.result import ResultType
from common.model.serializers import SerializedResultType

AggregationType = Literal["count", "min", "max", "sum", "avg"]


class AlignedSchemaMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    aggregation_type: AggregationType
    keys: list[str]
    value_column: str | None = None
    columns: list[str] = []
    source_agent_column: str


class AlignedResultList(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    data: SerializedResultType = ResultType()
    schema_metadata: AlignedSchemaMetadata

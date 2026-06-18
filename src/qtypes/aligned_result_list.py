from dataclasses import dataclass, field
from typing import Literal

import pandas as pd
from pydantic import BaseModel

AggregationType = Literal["count", "min", "max", "sum", "avg"]


class AlignedSchemaMetadata(BaseModel):
    aggregation_type: AggregationType
    keys: list[str]
    value_column: str | None = None


@dataclass
class AlignedResultList:
    data: pd.DataFrame = field(default_factory=pd.DataFrame)
    schema: AlignedSchemaMetadata | None = None

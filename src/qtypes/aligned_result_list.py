from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

AggregationType = Literal["count", "min", "max", "sum", "avg"]


@dataclass
class AlignedSchemaMetadata:
    aggregation_type: AggregationType
    keys: list[str]  # entity identifier columns
    value_column: str | None  # numeric column; None for count/listing queries


@dataclass
class AlignedResultList:
    data: pd.DataFrame = field(default_factory=pd.DataFrame)
    schema: AlignedSchemaMetadata | None = None

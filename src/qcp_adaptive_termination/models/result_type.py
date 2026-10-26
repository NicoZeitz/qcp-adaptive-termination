"""Serialization and type model for query plan results."""

from typing import Annotated, Any, Literal, overload

import numpy as np
import pandas as pd
from pandas import DataFrame, MultiIndex
from pydantic import BeforeValidator, JsonValue, PlainSerializer, SerializationInfo
from pydantic.json_schema import SkipJsonSchema

ResultType = DataFrame


def preview_dataframe(
    df: DataFrame | None,
    top_rows: int = 5,
    bottom_rows: int = 5,
    show_dtypes: bool = True,
    show_row_numbers: bool = True,
) -> str:
    """Render a preview string of a dataframe for display to the LLM."""
    if df is None:
        return "None"
    if df.empty:
        return df.__repr__()
    df_copy = df.copy()
    if show_dtypes:
        columns = [(c, df_copy[c].dtype) for c in df_copy.columns]
        df_copy.columns = MultiIndex.from_tuples(columns, names=["name", "type"])
    if not show_row_numbers:
        df_copy.index = [""] * len(df_copy)
    with pd.option_context("display.max_rows", top_rows + bottom_rows + 1):
        return df_copy.__repr__()


@overload
def serialize(result: ResultType, context: Literal["preview"]) -> str:
    """Serialize a result dataframe to a preview string."""
    ...


@overload
def serialize(result: ResultType, context: Literal["full"] = "full") -> dict[str, Any]:
    """Serialize a result dataframe to a full dict."""
    ...


def serialize(result: ResultType, context: Literal["preview", "full"] = "full") -> str | dict[str, Any]:
    """Serialize a result dataframe to a preview string or full dict."""
    if context == "preview":
        return preview_dataframe(result)

    # Replace NaN with None for JSON serialization (null in JSON)
    result_with_none = result.replace(np.nan, None)
    result_with_none = result_with_none.replace({np.inf: "Infinity", -np.inf: "-Infinity"})
    return result_with_none.to_dict(orient="split")


def deserialize(data: dict[str, Any]) -> ResultType:
    """Build a result dataframe from its serialized dict form."""
    df = DataFrame(**data)
    # Convert None values back to NaN
    return df.replace({None: np.nan})


def is_result_type(value: Any) -> bool:
    """Return whether the value is a result dataframe."""
    return isinstance(value, DataFrame)


def try_parse_value(value: Any) -> ResultType:
    """Coerce an LLM-provided value into a result dataframe."""
    if value is None:
        return DataFrame()
    elif isinstance(value, DataFrame):
        return value
    elif isinstance(value, str):
        return DataFrame({"result": [value]})
    elif isinstance(value, dict):
        return DataFrame({k: [v] for k, v in value.items()})
    elif isinstance(value, list):
        return DataFrame({"result": value})
    elif isinstance(value, (int, float, bool)):
        return DataFrame({"result": [value]})
    elif isinstance(value, (set, tuple)):
        return DataFrame({"result": list(value)})
    else:
        # Only list types that we want to hint to the LLM in the list
        raise ValueError(f"Answer must be a DataFrame, string or None, got {type(value)}: {value}")


def failed_result() -> ResultType:
    """Create a result dataframe indicating a failed query."""
    return DataFrame({"error": ["Failed to execute query"]})


def _serialize_result(result: ResultType, info: SerializationInfo) -> JsonValue:
    """Serialize a result according to the serialization context."""
    if info.context == "preview":
        return serialize(result, context="preview")
    else:
        return serialize(result, context="full")


def _validate_result(value: Any) -> ResultType:
    """Validate and deserialize a raw value into a result dataframe."""
    if isinstance(value, ResultType):
        return value
    elif isinstance(value, dict):
        return deserialize(value)
    return value


SerializedResultType = SkipJsonSchema[
    Annotated[
        ResultType,
        BeforeValidator(_validate_result),
        PlainSerializer(_serialize_result, return_type=JsonValue),
    ]
]


def _serialize_optional_result(result: ResultType | None, info: SerializationInfo) -> JsonValue:
    """Serialize an optional result, handling None."""
    if result is None:
        return None
    return _serialize_result(result, info)


def _validate_optional_result(value: Any) -> ResultType | None:
    """Validate an optional raw value into a result dataframe or None."""
    if value is None:
        return None
    return _validate_result(value)


SerializedOptionalResultType = SkipJsonSchema[
    Annotated[
        ResultType | None,
        BeforeValidator(_validate_optional_result),
        PlainSerializer(_serialize_optional_result, return_type=JsonValue),
    ]
]

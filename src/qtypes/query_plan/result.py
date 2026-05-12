from typing import Any, cast

from pandas import DataFrame
from pydantic import JsonValue

from utils.preview import preview_dataframe

ResultType = DataFrame


def try_parse_value_to_result_type(value: Any) -> ResultType:
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
        raise ValueError(
            f"Answer must be a DataFrame, string or None, got {type(value)}: {value}"
        )


def is_result_type(value: Any) -> bool:
    return isinstance(value, DataFrame)


def get_empty_result() -> ResultType:
    return DataFrame()


def get_failed_result() -> ResultType:
    return DataFrame({"error": ["Failed to execute query"]})


def result_to_span_data(result: ResultType) -> JsonValue:
    return preview_dataframe(result)


def result_to_json_preview(result: ResultType) -> JsonValue:
    return preview_dataframe(result)


def result_to_json(result: ResultType) -> JsonValue:
    return cast(JsonValue, result.to_dict(orient="records"))

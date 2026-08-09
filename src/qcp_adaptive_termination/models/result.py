from typing import Any

import numpy as np
from pandas import DataFrame

ResultType = DataFrame


def serialize_result(result: ResultType) -> dict[str, Any]:
    # Replace NaN with None for JSON serialization (null in JSON)
    result_with_none = result.replace(np.nan, None)
    result_with_none = result_with_none.replace({np.inf: "Infinity", -np.inf: "-Infinity"})
    return result_with_none.to_dict(orient="split")


def deserialize_result(data: dict[str, Any]) -> ResultType:
    df = DataFrame(**data)
    # Convert None values back to NaN
    return df.replace({None: np.nan})


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
        raise ValueError(f"Answer must be a DataFrame, string or None, got {type(value)}: {value}")


def get_failed_result() -> ResultType:
    return DataFrame({"error": ["Failed to execute query"]})

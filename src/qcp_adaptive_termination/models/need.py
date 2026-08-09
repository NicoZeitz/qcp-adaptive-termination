from typing import Any

NeedType = str


def serialize_need(need: NeedType) -> str:
    return need


def deserialize_need(data: str) -> NeedType:
    return data


def exception_to_need(exception: Exception) -> NeedType:
    return f"ERROR: {str(exception)}"


def try_parse_value_to_need_type(value: Any) -> set[NeedType]:
    if isinstance(value, str):
        return {value}
    elif isinstance(value, list):
        return {item for item in value if isinstance(item, str)}
    else:
        raise ValueError(f"Needs must be a string or a list of strings, got {type(value)}: {value}")

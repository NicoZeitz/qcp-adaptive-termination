from typing import Annotated, Any

from pydantic import BeforeValidator, JsonValue, PlainSerializer
from pydantic.json_schema import SkipJsonSchema

NeedType = str


def serialize(need: NeedType) -> str:
    return need


def deserialize(data: str) -> NeedType:
    return data


def from_exception(exception: Exception) -> NeedType:
    return f"ERROR: {str(exception)}"


def try_parse_value(value: Any) -> set[NeedType]:
    if isinstance(value, str):
        return {value}
    elif isinstance(value, list):
        return {item for item in value if isinstance(item, str)}
    else:
        raise ValueError(f"Needs must be a string or a list of strings, got {type(value)}: {value}")


def _validate_need_list(value: Any) -> list[NeedType]:
    if isinstance(value, list):
        return [deserialize(need) for need in value]
    raise ValueError(f"Needs must be a list, got {type(value)}: {value}")


def _serialize_need_list(needs: list[NeedType]) -> JsonValue:
    return [serialize(need) for need in needs]


SerializedNeedList = SkipJsonSchema[
    Annotated[
        list[NeedType],
        BeforeValidator(_validate_need_list),
        PlainSerializer(_serialize_need_list, return_type=JsonValue),
    ]
]

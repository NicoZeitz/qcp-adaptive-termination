"""Models for the needs a query plan step raises."""

from typing import Annotated, Any, Literal

from pydantic import BeforeValidator, JsonValue, PlainSerializer
from pydantic.json_schema import SkipJsonSchema

NeedType = str


def serialize(need: NeedType, context: Literal["preview", "full"] = "full") -> str:
    """Return the serialized form of a need."""
    return need


def deserialize(data: str) -> NeedType:
    """Parse a need from its serialized string form."""
    return data


def from_exception(exception: Exception) -> NeedType:
    """Build a need describing the given exception."""
    return f"ERROR: {str(exception)}"


def try_parse_value(value: Any) -> set[NeedType]:
    """Coerce an LLM-provided value into a set of needs."""
    if isinstance(value, str):
        return {value}
    elif isinstance(value, list):
        return {item for item in value if isinstance(item, str)}
    else:
        raise ValueError(f"Needs must be a string or a list of strings, got {type(value)}: {value}")


def _validate_need_list(value: Any) -> list[NeedType]:
    """Validate and deserialize a raw list of needs."""
    if isinstance(value, list):
        return [deserialize(need) for need in value]
    raise ValueError(f"Needs must be a list, got {type(value)}: {value}")


def _serialize_need_list(needs: list[NeedType]) -> JsonValue:
    """Serialize a list of needs to JSON."""
    return [serialize(need) for need in needs]


SerializedNeedList = SkipJsonSchema[
    Annotated[
        list[NeedType],
        BeforeValidator(_validate_need_list),
        PlainSerializer(_serialize_need_list, return_type=JsonValue),
    ]
]

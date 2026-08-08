from typing import Annotated, Any

from pydantic import BeforeValidator, JsonValue, PlainSerializer, SerializationInfo
from pydantic.json_schema import SkipJsonSchema

from qcp_adaptive_termination.models.need import (
    NeedType,
    deserialize_need,
    serialize_need,
)
from qcp_adaptive_termination.models.result import (
    ResultType,
    deserialize_result,
    serialize_result as serialize_result_type,
)
from qcp_adaptive_termination.preview import result_to_json_preview


def validate_result(value: Any) -> ResultType:
    if isinstance(value, ResultType):
        return value
    elif isinstance(value, dict):
        return deserialize_result(value)
    return value


def serialize_result(result: ResultType, info: SerializationInfo) -> JsonValue:
    if info.context == "preview":
        return result_to_json_preview(result)
    return serialize_result_type(result)


def validate_optional_result(value: Any) -> ResultType | None:
    if value is None:
        return None
    return validate_result(value)


def serialize_optional_result(
    result: ResultType | None, info: SerializationInfo
) -> JsonValue:
    if result is None:
        return None
    return serialize_result(result, info)


def validate_need_list(value: Any) -> list[NeedType]:
    if isinstance(value, list):
        return [deserialize_need(need) for need in value]
    raise ValueError(f"Needs must be a list, got {type(value)}: {value}")


def serialize_need_list(needs: list[NeedType]) -> JsonValue:
    return [serialize_need(need) for need in needs]


SerializedResultType = SkipJsonSchema[
    Annotated[
        ResultType,
        BeforeValidator(validate_result),
        PlainSerializer(serialize_result, return_type=JsonValue),
    ]
]

SerializedOptionalResultType = SkipJsonSchema[
    Annotated[
        ResultType | None,
        BeforeValidator(validate_optional_result),
        PlainSerializer(serialize_optional_result, return_type=JsonValue),
    ]
]

SerializedNeedList = SkipJsonSchema[
    Annotated[
        list[NeedType],
        BeforeValidator(validate_need_list),
        PlainSerializer(serialize_need_list, return_type=JsonValue),
    ]
]

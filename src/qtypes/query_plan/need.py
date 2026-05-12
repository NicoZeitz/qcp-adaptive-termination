from typing import Any

from pydantic import JsonValue

from utils.preview import CHAR_LIMIT_SPAN, preview_text

NeedType = str


def try_parse_value_to_need_type(value: Any) -> set[NeedType]:
    if isinstance(value, str):
        return {value}
    elif isinstance(value, list):
        return {item for item in value if isinstance(item, str)}
    else:
        raise ValueError(
            f"Needs must be a string or a list of strings, got {type(value)}: {value}"
        )


def needs_to_span_data(needs: list[NeedType]) -> JsonValue:
    return [preview_text(need, CHAR_LIMIT_SPAN) for need in needs]


def needs_to_json_value(needs: list[NeedType]) -> JsonValue:
    return [need for need in needs]

def exception_to_need(exception: Exception) -> NeedType:
    return f"ERROR: {str(exception)}"
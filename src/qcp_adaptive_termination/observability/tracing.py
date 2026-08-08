import logging
import math
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from agents import CustomSpanData, Span, custom_span
from langfuse import get_client
from pydantic import BaseModel

_span_by_id: dict[str | None, int] = {}
_span_context_var = ContextVar[Span[CustomSpanData] | None](
    "current_span", default=None
)


def serialize_data_for_tracing(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return serialize_data_for_tracing(value.model_dump(context="preview"))
    elif isinstance(value, list):
        return [serialize_data_for_tracing(item) for item in value]
    elif isinstance(value, dict):
        return {key: serialize_data_for_tracing(val) for key, val in value.items()}
    elif isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    else:
        return value


def append_to_span_input(
    data: dict[str, Any], span: Span[CustomSpanData] | None = None
):
    data = serialize_data_for_tracing(data)
    span = span or _span_context_var.get()
    if span is not None:
        span.span_data.data["input"] = {**span.span_data.data.get("input", {}), **data}


def append_to_span_output(
    data: dict[str, Any], span: Span[CustomSpanData] | None = None
):
    data = serialize_data_for_tracing(data)
    span = span or _span_context_var.get()
    if span is not None:
        span.span_data.data["output"] = {
            **span.span_data.data.get("output", {}),
            **data,
        }


def get_current_span() -> Span[CustomSpanData] | None:
    return _span_context_var.get()


def indented_logging(message: str, log_level: int = logging.INFO):
    span = _span_context_var.get()
    if span is not None:
        parent_depth = _span_by_id.get(span.parent_id, 0)
        logging.log(log_level, "  " * (parent_depth + 1) + message)
    else:
        logging.log(log_level, message)


def _update_langfuse_span_metadata(span: Span[CustomSpanData]):
    langfuse = get_client()
    if "input" in span.span_data.data:
        langfuse.update_current_span(input=span.span_data.data["input"])
    if "output" in span.span_data.data:
        langfuse.update_current_span(output=span.span_data.data["output"])
    metadata = span.span_data.data.copy()
    metadata.pop("input", None)
    metadata.pop("output", None)
    langfuse.update_current_span(metadata=metadata)


@contextmanager
def own_span(
    span_name: str,
    data: dict[str, Any] | None = None,
    input: dict[str, Any] | None = None,
    output: dict[str, Any] | None = None,
):
    with custom_span(span_name, data=data) as span:
        token = _span_context_var.set(span)
        try:
            if input is not None:
                append_to_span_input(input, span)

            parent_depth = _span_by_id.get(span.parent_id, 0)
            _span_by_id[span.span_id] = parent_depth + 1

            logging.info("  " * parent_depth + f"Start span: {span_name}")
            yield span
            logging.info("  " * parent_depth + f"End span: {span_name}")

            if output is not None:
                append_to_span_output(output, span)

            _update_langfuse_span_metadata(span)
        finally:
            _span_context_var.reset(token)

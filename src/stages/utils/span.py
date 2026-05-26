from contextlib import contextmanager
from contextvars import ContextVar
import logging
from agents import CustomSpanData, Span, custom_span
from typing import Any

from langfuse import get_client
from pydantic import BaseModel


def model_dump(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump()
    elif isinstance(value, list):
        return [model_dump(item) for item in value]
    elif isinstance(value, dict):
        return {key: model_dump(val) for key, val in value.items()}
    else:
        return value


def append_to_span_input(
    data: dict[str, Any], span: Span[CustomSpanData] | None = None
):
    data = model_dump(data)
    span = span or span_context_var.get()
    if span is not None:
        span.span_data.data["input"] = {**span.span_data.data.get("input", {}), **data}


def append_to_span_output(
    data: dict[str, Any], span: Span[CustomSpanData] | None = None
):
    data = model_dump(data)
    span = span or span_context_var.get()
    if span is not None:
        span.span_data.data["output"] = {
            **span.span_data.data.get("output", {}),
            **data,
        }


_span_by_id: dict[str | None, int] = {}
span_context_var = ContextVar[Span[CustomSpanData] | None]("current_span", default=None)


@contextmanager
def own_custom_span(
    span_name: str,
    data: dict[str, Any] | None = None,
    input: dict[str, Any] | None = None,
    output: dict[str, Any] | None = None,
    print_start_end: bool = True,
):
    with custom_span(span_name, data=data) as span:
        token = span_context_var.set(span)
        try:
            if input is not None:
                append_to_span_input(input, span)

            parent_depth = _span_by_id.get(span.parent_id, 0)
            _span_by_id[span.span_id] = parent_depth + 1
            if print_start_end:
                logging.info("  " * parent_depth + f"Start span: {span_name}")
            else:
                logging.info("  " * parent_depth + f"Span: {span_name}")
            yield span
            if print_start_end:
                logging.info("  " * parent_depth + f"End span: {span_name}")

            if output is not None:
                append_to_span_output(output, span)

            langfuse = get_client()

            if "input" in span.span_data.data:
                langfuse.update_current_span(input=span.span_data.data["input"])
            if "output" in span.span_data.data:
                langfuse.update_current_span(output=span.span_data.data["output"])

            metadata = span.span_data.data.copy()
            metadata.pop("input", None)
            metadata.pop("output", None)
            langfuse.update_current_span(metadata=metadata)
        finally:
            span_context_var.reset(token)

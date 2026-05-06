from contextlib import contextmanager
from agents import Span, custom_span
from typing import Any
from utils.initialization import langfuse


def append_to_span_input(span: Span, data: dict[str, Any]):
    span.span_data.data["input"] = {**span.span_data.data.get("input", {}), **data}


def append_to_span_output(span: Span, data: dict[str, Any]):
    span.span_data.data["output"] = {**span.span_data.data.get("output", {}), **data}


_span_by_id: dict[str | None, int] = {}


@contextmanager
def own_custom_span(span_name: str, data: dict[str, Any] | None = None):
    with custom_span(span_name, data=data) as span:
        parent_depth = _span_by_id.get(span.parent_id, 0)
        _span_by_id[span.span_id] = parent_depth + 1
        print("  " * parent_depth + span_name)
        yield span

        if "input" in span.span_data.data:
            langfuse.update_current_span(input=span.span_data.data["input"])
        if "output" in span.span_data.data:
            langfuse.update_current_span(output=span.span_data.data["output"])

        metadata = span.span_data.data.copy()
        metadata.pop("input", None)
        metadata.pop("output", None)
        langfuse.update_current_span(metadata=metadata)

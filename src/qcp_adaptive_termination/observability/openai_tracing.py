from typing import Any

from agents.tracing.processors import BackendSpanExporter

_OPENAI_METADATA_MAX_CHARS = 512


class TruncatingBackendSpanExporter(BackendSpanExporter):
    """
    OpenAI exporter that also truncates span_data.data payloads and metadata strings.

    The upstream exporter sanitizes input/output/usage but may still emit large
    custom span payloads under span_data.data, which the OpenAI traces ingest API
    can reject when above ~10KB. It also does not truncate top-level metadata string
    values, which the API caps at 512 characters.
    """

    def __init__(self, *args: Any, max_field_bytes: int = 9500, **kwargs: Any):
        super().__init__(*args, **kwargs)
        # Use a stricter per-field cap for OpenAI ingestion to avoid 400 payload-too-large.
        self._OPENAI_TRACING_MAX_FIELD_BYTES = max_field_bytes

    def _sanitize_for_openai_tracing_api(self, payload_item: dict[str, Any]) -> dict[str, Any]:
        sanitized = super()._sanitize_for_openai_tracing_api(payload_item)

        sanitized = self._truncate_metadata(sanitized)

        span_data = sanitized.get("span_data")
        if not isinstance(span_data, dict):
            return sanitized

        # Also sanitize custom data payloads, which are not handled by upstream.
        if "data" not in span_data:
            return sanitized

        truncated_data = self._truncate_span_field_value(span_data["data"])
        if truncated_data is span_data["data"]:
            return sanitized

        sanitized_span_data = dict(span_data)
        sanitized_span_data["data"] = truncated_data

        sanitized_payload = dict(sanitized)
        sanitized_payload["span_data"] = sanitized_span_data
        return sanitized_payload

    def _truncate_metadata(self, payload_item: dict[str, Any]) -> dict[str, Any]:
        metadata = payload_item.get("metadata")
        if not isinstance(metadata, dict):
            return payload_item
        truncated: dict[str, Any] = {}
        mutated = False
        for k, v in metadata.items():
            if isinstance(v, str) and len(v) > _OPENAI_METADATA_MAX_CHARS:
                truncated[k] = v[:_OPENAI_METADATA_MAX_CHARS]
                mutated = True
            else:
                truncated[k] = v
        if not mutated:
            return payload_item
        result = dict(payload_item)
        result["metadata"] = truncated
        return result

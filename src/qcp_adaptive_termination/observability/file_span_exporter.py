"""OpenTelemetry span exporter that appends spans as JSON lines to a file."""

import json
import threading
from collections.abc import Sequence
from io import TextIOBase
from pathlib import Path

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult


class FileSpanExporter(SpanExporter):
    """An OpenTelemetry SpanExporter that appends spans as JSON lines to a file."""

    def __init__(
        self,
        file_path: str | Path,
    ) -> None:
        """Initialize the exporter with the target file path."""
        self.file_path = Path(file_path) if isinstance(file_path, str) else file_path
        self._lock = threading.Lock()
        self._file: TextIOBase | None = None
        self._wrote_header = False

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Write the given spans to the file as JSON lines."""
        with self._lock:
            if not self._file and isinstance(self.file_path, Path):
                self._file = self.file_path.open("a+")

            if not self._file:
                return SpanExportResult.FAILURE

            for span in spans:
                span_dict = {
                    "name": span.name,
                    "parent_id": span.parent.span_id if span.parent else None,
                    "start_time": span.start_time,
                    "end_time": span.end_time,
                    "attributes": span.attributes,
                    "status": {
                        "status_code": span.status.status_code.name,
                        "description": span.status.description,
                    },
                }
                self._file.write(json.dumps(span_dict, default=lambda x: dict(x)) + "\n")
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        """Release any resources held by the exporter."""
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Flush buffered spans to disk."""
        with self._lock:
            if self._file:
                self._file.flush()
                if self._file is not self.file_path:
                    # don't close the file if it was passed in
                    self._file.close()
        return True

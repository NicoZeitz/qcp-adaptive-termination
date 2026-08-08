from pathlib import Path
import threading
import json
from typing import IO, Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult


class FileSpanExporter(SpanExporter):
    def __init__(
        self,
        file_path: str | Path,
    ) -> None:
        self.file_path = Path(file_path) if isinstance(file_path, str) else file_path
        self._lock = threading.Lock()
        self._file: IO[str] | None = None
        self._wrote_header = False

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        with self._lock:
            if not self._file:
                if isinstance(self.file_path, Path):
                    self._file = self.file_path.open("a+")

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
                self._file.write(  # ty: ignore
                    json.dumps(span_dict, default=lambda x: dict(x)) + "\n"
                )
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        with self._lock:
            if self._file:
                self._file.flush()
                if self._file is not self.file_path:
                    # don't close the file if it was passed in
                    self._file.close()
        return True

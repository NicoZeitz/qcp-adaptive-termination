import logging
import os
from pathlib import Path

import nest_asyncio
from agents import set_trace_processors, set_tracing_export_api_key
from agents.tracing.processors import BatchTraceProcessor
from langfuse import get_client
from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor
from opentelemetry import trace
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from qcp_adaptive_termination.tracing.file_span_exporter import FileSpanExporter
from qcp_adaptive_termination.tracing.openai_tracing import (
    TruncatingBackendSpanExporter,
)

__all__ = [
    "FileSpanExporter",
    "TruncatingBackendSpanExporter",
    "setup_logging",
    "setup_tracing",
    "setup_langfuse",
]


def setup_logging(log_name: str, log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{log_name}.log"
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logging.getLogger().addHandler(logging.FileHandler(log_file))
    logging.info(f"Logging initialized. Log file: {log_file}")


def setup_langfuse() -> bool:
    langfuse = get_client()
    if not langfuse.auth_check():
        logging.error("Langfuse authentication failed. Check credentials and host.")
        return False
    return True


def setup_tracing(log_name: str, log_dir: Path) -> None:
    tracing_api_key = os.getenv("OPENAI_API_KEY")
    if tracing_api_key:
        set_tracing_export_api_key(tracing_api_key)
        max_field_bytes = int(os.getenv("OPENAI_TRACING_MAX_FIELD_BYTES", "9500"))
        openai_exporter = TruncatingBackendSpanExporter(
            api_key=tracing_api_key,
            max_field_bytes=max_field_bytes,
        )
        set_trace_processors([BatchTraceProcessor(exporter=openai_exporter)])
    else:
        logging.warning(
            "No OpenAI tracing API key found; OpenAI trace export is disabled."
        )

    tracing_log_file = log_dir / f"{log_name}_traces.jsonl"
    nest_asyncio.apply()
    trace.get_tracer_provider().add_span_processor(  # ty: ignore
        BatchSpanProcessor(FileSpanExporter(tracing_log_file))
    )
    OpenAIAgentsInstrumentor().instrument(exclusive_processor=False)
    logging.info(f"Tracing initialized. Trace file: {tracing_log_file}")

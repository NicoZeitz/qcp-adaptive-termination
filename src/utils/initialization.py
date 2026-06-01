from datetime import datetime
import os
from pathlib import Path
import logging
from typing import Literal
from agents import set_default_openai_client, set_trace_processors
from agents.tracing.processors import BatchTraceProcessor
from dotenv import load_dotenv
import nest_asyncio
import sys
from openai import AsyncOpenAI
from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor
from langfuse import get_client
from opentelemetry import trace
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from utils.cache import Cache
from utils.file_span_exporter import FileSpanExporter
from utils.load_prompt import LangfuseProcessor
from utils.openai_tracing import TruncatingBackendSpanExporter


def initialize(
    log_name: str = f"{datetime.now().strftime('%Y-%m-%d %H-%M-%S')}",
    log_dir: Path = Path("../.logs"),
    cache_dir: Path = Path("../.cache"),
    provider: Literal["University", "Work"] = "University",
    throw_on_cache_miss: bool = False,
    disable_cache: bool = False,
):
    # Load env
    load_dotenv(override=True)

    # Set up logging FIRST (before any logging.info() calls)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{log_name}.log"
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    rootLogger = logging.getLogger()

    fileHandler = logging.FileHandler(log_file)
    rootLogger.addHandler(fileHandler)

    logging.info(f"Logging initialized. Log file: {log_file}")

    # Set up cache
    cache_dir.mkdir(parents=True, exist_ok=True)
    Cache.instance().set_cache_folder(cache_dir)
    if disable_cache:
        Cache.instance().set_mode("DISABLED")
        logging.info("Cache completely disabled")
    elif throw_on_cache_miss:
        Cache.instance().set_mode("READ_AND_WRITE_THROW_ON_MISS")
    else:
        Cache.instance().set_mode("READ_AND_WRITE")

    # Set up langfuse
    langfuse = get_client()
    if not langfuse.auth_check():
        logging.error(
            "Langfuse Authentication failed. Please check your credentials and host."
        )
        sys.exit(1)

    # Set up OpenAI Client
    custom_client = (
        AsyncOpenAI(base_url=os.getenv("BASE_URL"), api_key=os.getenv("API_KEY"))
        if provider == "Work"
        else AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    )
    set_default_openai_client(custom_client)

    # Keep OpenAI trace export enabled, but trim payload fields before upload.
    # Langfuse and file exporters below still receive full, untruncated spans.
    tracing_api_key = os.getenv("OPENAI_API_KEY")
    if tracing_api_key:
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

    # Set up tracing
    tracing_log_file = log_dir / f"{log_name}_traces.jsonl"
    nest_asyncio.apply()
    trace.get_tracer_provider().add_span_processor(LangfuseProcessor())  # type: ignore
    trace.get_tracer_provider().add_span_processor(  # type: ignore
        BatchSpanProcessor(FileSpanExporter(tracing_log_file))
    )
    OpenAIAgentsInstrumentor().instrument(exclusive_processor=False)
    logging.info(f"Tracing initialized. Trace file: {tracing_log_file}")

from datetime import datetime
import os
from pathlib import Path
import logging
from typing import Literal, cast
from agents import set_default_openai_client, set_tracing_export_api_key
from dotenv import load_dotenv
import nest_asyncio
import sys
from openai import AsyncOpenAI
from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor
from langfuse import get_client
from opentelemetry import trace
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from stages.utils.cache import Cache
from stages.utils.file_span_exporter import FileSpanExporter
from stages.utils.load_prompt import LangfuseProcessor


def initialize(
    log_name: str = f"{datetime.now().strftime('%Y-%m-%d %H-%M-%S')}",
    log_dir: Path = Path("../.logs"),
    cache_dir: Path = Path("../.cache"),
    provider: Literal["University", "Work"] = "University",
    throw_on_cache_miss: bool = False,
):
    # Load env
    load_dotenv()

    # Set up cache
    cache_dir.mkdir(parents=True, exist_ok=True)
    Cache.instance().set_cache_folder(cache_dir)
    if throw_on_cache_miss:
        Cache.instance().set_mode("READ_AND_WRITE_THROW_ON_MISS")
    else:
        Cache.instance().set_mode("READ_AND_WRITE")

    # Set up logging
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{log_name}.log"
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    rootLogger = logging.getLogger()

    fileHandler = logging.FileHandler(log_file)
    rootLogger.addHandler(fileHandler)

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
    set_tracing_export_api_key(api_key=cast(str, os.getenv("OPENAI_API_KEY")))

    # Set up tracing
    tracing_log_file = log_dir / f"{log_name}_traces.jsonl"
    nest_asyncio.apply()
    trace.get_tracer_provider().add_span_processor(LangfuseProcessor())  # type: ignore
    trace.get_tracer_provider().add_span_processor(  # type: ignore
        BatchSpanProcessor(FileSpanExporter(tracing_log_file))
    )
    OpenAIAgentsInstrumentor().instrument(exclusive_processor=False)

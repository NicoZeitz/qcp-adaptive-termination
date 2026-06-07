from datetime import datetime
import json
import os
from pathlib import Path
import logging
from typing import Literal, Unpack
from agents import set_default_openai_client, set_trace_processors
from agents.tracing.processors import BatchTraceProcessor
from dotenv import load_dotenv
import httpx
from openai._types import HttpxSendArgs
import nest_asyncio
import sys
from openai import AsyncOpenAI
from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor
from langfuse import get_client
from opentelemetry import trace
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from stages.common.agent_configuration import set_agent_models
from utils.cache import Cache
from utils.file_span_exporter import FileSpanExporter
from utils.load_prompt import LangfuseProcessor
from utils.openai_tracing import TruncatingBackendSpanExporter


class APIClient(AsyncOpenAI):
    async def _send_request(
        self,
        request: httpx.Request,
        *,
        stream: bool,
        **kwargs: Unpack[HttpxSendArgs],
    ) -> httpx.Response:
        content = json.loads(request.content.decode("utf-8"))
        if (
            "text" in content
            and "response_format" not in content
            and "GLM" in content["model"]
        ):
            content["response_format"] = content["text"]["format"]
            content["input"].append(
                {
                    "role": "system",
                    "content": f"JSON Schema: ```json\n{json.dumps(content['text']['format']['schema'], indent=2, ensure_ascii=False)}\n```",
                }
            )
            # del content["text"]
            request._content = json.dumps(content).encode("utf-8")

        logging.info(content)
        response = await super()._send_request(request, stream=stream, **kwargs)
        logging.info(response.json())
        return response


def _normalize_openai_compatible_model_name(model_name: str | None) -> str | None:
    if model_name is None or "/" not in model_name:
        return model_name

    known_prefixes = {"openai", "litellm", "any-llm"}
    prefix = model_name.split("/", 1)[0]

    if prefix in known_prefixes:
        return model_name

    # Route unknown prefixes through the OpenAI provider alias so OpenAI-compatible
    # backends can still receive namespaced model IDs like "unsloth/...".
    return f"openai/{model_name}"


def initialize(
    log_name: str = f"{datetime.now().strftime('%Y-%m-%d %H-%M-%S')}",
    log_dir: Path = Path("../.logs"),
    cache_dir: Path = Path("../.cache"),
    provider: Literal["OpenAI", "University", "Work"] = "OpenAI",
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
    if provider == "OpenAI":
        coordinator_model = _normalize_openai_compatible_model_name(
            os.getenv("OPENAI_COORDINATOR_MODEL")
        )
        data_model = _normalize_openai_compatible_model_name(
            os.getenv("OPENAI_DATA_MODEL")
        )

        custom_client = APIClient(
            base_url=os.getenv("OPENAI_BASE_URL"), api_key=os.getenv("OPENAI_API_KEY")
        )
        set_agent_models(coordinator_model, data_model)
        logging.info(f"OpenAI base URL: {os.getenv('OPENAI_BASE_URL')}")
        logging.info(
            f"OpenAI API key: {'set' if os.getenv('OPENAI_API_KEY') else 'not set'}"
        )
        logging.info(f"OpenAI coordinator model: {coordinator_model}")
        logging.info(f"OpenAI data model: {data_model}")
    elif provider == "University":
        coordinator_model = _normalize_openai_compatible_model_name(
            os.getenv("UNIVERSITY_COORDINATOR_MODEL")
        )
        data_model = _normalize_openai_compatible_model_name(
            os.getenv("UNIVERSITY_DATA_MODEL")
        )

        custom_client = APIClient(
            base_url=os.getenv("UNIVERSITY_BASE_URL"),
            api_key=os.getenv("UNIVERSITY_API_KEY"),
        )
        set_agent_models(coordinator_model, data_model)
        logging.info(f"University base URL: {os.getenv('UNIVERSITY_BASE_URL')}")
        logging.info(
            f"University API key: {'set' if os.getenv('UNIVERSITY_API_KEY') else 'not set'}"
        )
        logging.info(f"University coordinator model: {coordinator_model}")
        logging.info(f"University data model: {data_model}")
    elif provider == "Work":
        coordinator_model = _normalize_openai_compatible_model_name(
            os.getenv("WORK_COORDINATOR_MODEL")
        )
        data_model = _normalize_openai_compatible_model_name(
            os.getenv("WORK_DATA_MODEL")
        )

        custom_client = APIClient(
            base_url=os.getenv("WORK_BASE_URL"), api_key=os.getenv("WORK_API_KEY")
        )
        set_agent_models(coordinator_model, data_model)
        logging.info(f"Work base URL: {os.getenv('WORK_BASE_URL')}")
        logging.info(
            f"Work API key: {'set' if os.getenv('WORK_API_KEY') else 'not set'}"
        )
        logging.info(f"Work coordinator model: {coordinator_model}")
        logging.info(f"Work data model: {data_model}")
    else:
        raise ValueError(f"Unsupported provider: {provider}")
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

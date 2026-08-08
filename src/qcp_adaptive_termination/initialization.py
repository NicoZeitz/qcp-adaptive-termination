import json
import logging
import os
from pathlib import Path

import httpx
import nest_asyncio
import sys
from agents import (
    MultiProvider,
    OpenAIProvider,
    set_trace_processors,
    set_tracing_export_api_key,
)
from agents.models.multi_provider import MultiProviderMap
from agents.tracing.processors import BatchTraceProcessor
from dotenv import load_dotenv
from openai import AsyncOpenAI
from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor
from langfuse import get_client
from opentelemetry import trace
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from qcp_adaptive_termination.cache import Cache
from qcp_adaptive_termination.tracing import (
    FileSpanExporter,
    TruncatingBackendSpanExporter,
)

_multi_provider: MultiProvider | None = None


def set_multi_provider(provider: MultiProvider) -> None:
    global _multi_provider
    _multi_provider = provider


def get_multi_provider() -> MultiProvider:
    if _multi_provider is None:
        raise RuntimeError("MultiProvider not initialized — call initialize() first")
    return _multi_provider


async def modify_request_hook(request: httpx.Request):
    if request.method == "POST" and "application/json" in request.headers.get(
        "content-type", ""
    ):
        body_bytes = await request.aread()
        request_json = json.loads(body_bytes.decode("utf-8"))

        request_json["temperature"] = 0.0

        # thinking_mode needs to be disabled for correct output, otherwise it will return output in reasoning, which breaks openai sdk
        thinking_enabled = (
            "reasoning_effort" in request_json
            and request_json["reasoning_effort"] != "none"
        )

        request_json["chat_template_kwargs"] = {
            "thinking_mode": "enabled" if thinking_enabled else "disabled",
            "thinking": True if thinking_enabled else False,
            "reasoning_effort": request_json.get("reasoning_effort", "none"),
        }
        if "reasoning_effort" in request_json:
            del request_json["reasoning_effort"]

        # if tool choice required, tools not empty and response format exist, remove response format
        if (
            request_json.get("tool_choice") == "required"
            and request_json.get("tools")
            and request_json.get("response_format") is not None
        ):
            del request_json["response_format"]

        # Re-encode and write back to the request stream
        modified_bytes = json.dumps(request_json).encode("utf-8")
        request.stream = httpx.ByteStream(modified_bytes)
        request.headers["Content-Length"] = str(len(modified_bytes))

    return request


def add_provider(
    prefix: str, provider_map: MultiProviderMap, *, university_hooks: bool = False
):
    base_url = os.getenv(f"{prefix.upper()}_BASE_URL")
    api_key = os.getenv(f"{prefix.upper()}_API_KEY")

    if not base_url:
        logging.warning(
            f"{prefix.upper()}_BASE_URL not set; skipping provider registration."
        )
        return

    client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        http_client=httpx.AsyncClient(event_hooks={"request": [modify_request_hook]})
        if university_hooks
        else None,
    )
    provider = OpenAIProvider(
        openai_client=client,
        use_responses=False if university_hooks else None,
    )
    provider_map.add_provider(prefix.lower(), provider)


def setup_logging(log_name: str, log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{log_name}.log"
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logging.getLogger().addHandler(logging.FileHandler(log_file))
    logging.info(f"Logging initialized. Log file: {log_file}")


def setup_cache(
    cache_dir: Path,
    throw_on_cache_miss: bool = False,
    disable_cache: bool = False,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    Cache.instance().set_cache_folder(cache_dir)
    if disable_cache:
        Cache.instance().set_mode("DISABLED")
        logging.info("Cache completely disabled")
    elif throw_on_cache_miss:
        Cache.instance().set_mode("READ_AND_WRITE_THROW_ON_MISS")
    else:
        Cache.instance().set_mode("READ_AND_WRITE")


def setup_providers() -> None:
    provider_map = MultiProviderMap()
    add_provider("OPENAI", provider_map=provider_map)
    add_provider("UNIVERSITY", provider_map=provider_map, university_hooks=True)
    add_provider("WORK", provider_map=provider_map)
    set_multi_provider(MultiProvider(provider_map=provider_map))


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
    trace.get_tracer_provider().add_span_processor(  # type: ignore
        BatchSpanProcessor(FileSpanExporter(tracing_log_file))
    )
    OpenAIAgentsInstrumentor().instrument(exclusive_processor=False)
    logging.info(f"Tracing initialized. Trace file: {tracing_log_file}")


def initialize(
    log_name: str,
    log_dir: Path,
    cache_dir: Path,
    throw_on_cache_miss: bool = False,
    disable_cache: bool = False,
) -> None:
    load_dotenv(override=True)
    setup_logging(log_name, log_dir)
    setup_cache(cache_dir, throw_on_cache_miss, disable_cache)
    setup_providers()
    if not setup_langfuse():
        sys.exit(1)
    setup_tracing(log_name, log_dir)

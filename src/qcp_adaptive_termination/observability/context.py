"""
Context managers for logging, tracing, cache, and config setup.

Use individually (stacking) or via `pipeline_context` which composes all four.
`pipeline_context` is idempotent: if already active it is a no-op.
"""

import atexit
import contextlib
import logging
import os
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path

import agents.tracing.setup as _agents_tracing_setup
import nest_asyncio2
from agents import get_current_trace, set_trace_processors, set_tracing_export_api_key, trace
from agents.tracing.processors import BatchTraceProcessor
from dotenv import load_dotenv
from langfuse import get_client
from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from pydantic import JsonValue

from qcp_adaptive_termination.models import CacheMode, Config
from qcp_adaptive_termination.observability.cache import Cache
from qcp_adaptive_termination.observability.file_span_exporter import FileSpanExporter
from qcp_adaptive_termination.observability.openai_tracing import TruncatingBackendSpanExporter
from qcp_adaptive_termination.observability.tracing import serialize_data_for_tracing

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
_DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / ".logs"
_DEFAULT_CACHE_DIR = _PROJECT_ROOT / ".cache"

from dataclasses import dataclass, field

_cache_var: ContextVar[Cache | None] = ContextVar("cache", default=None)
_config_var: ContextVar[Config | None] = ContextVar("config", default=None)
_data_dir_var: ContextVar[Path | None] = ContextVar("data_dir", default=None)
_output_dir_var: ContextVar[Path | None] = ContextVar("output_dir", default=None)
_logging_active_var: ContextVar[bool] = ContextVar("logging_active", default=False)


@dataclass
class UsageAccumulator:
    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0
    duration_ms: int = 0
    embedding_calls: int = 0
    embedding_duration_ms: int = 0
    ranking_duration_ms: int = 0


_usage_accumulator_var: ContextVar[UsageAccumulator | None] = ContextVar("usage_accumulator", default=None)


def get_usage_accumulator() -> UsageAccumulator:
    acc = _usage_accumulator_var.get()
    if acc is None:
        acc = UsageAccumulator()
        _usage_accumulator_var.set(acc)
    return acc


def reset_usage_accumulator() -> UsageAccumulator:
    """Reset and return a fresh accumulator, returning the old totals."""
    old = get_usage_accumulator()
    _usage_accumulator_var.set(UsageAccumulator())
    return old


def get_cache() -> Cache | None:
    return _cache_var.get()


def get_config() -> Config:
    cfg = _config_var.get()
    if cfg is None:
        cfg = Config()
        _config_var.set(cfg)
    return cfg


def get_data_dir() -> Path | None:
    return _data_dir_var.get()


def get_output_dir() -> Path | None:
    return _output_dir_var.get()


def _is_cache_active() -> bool:
    return _cache_var.get() is not None


def _config_is_active() -> bool:
    return _config_var.get() is not None


def _is_logging_active() -> bool:
    return _logging_active_var.get()


def _is_tracing_active() -> bool:
    return get_current_trace() is not None


def _is_data_dir_active() -> bool:
    return _data_dir_var.get() is not None


def _setup_tracing(output_dir: Path) -> BatchTraceProcessor | None:
    tracing_api_key = os.getenv("OPENAI_API_KEY")
    openai_processor: BatchTraceProcessor | None = None
    if tracing_api_key:
        set_tracing_export_api_key(tracing_api_key)
        max_field_bytes = int(os.getenv("OPENAI_TRACING_MAX_FIELD_BYTES", "9500"))
        openai_exporter = TruncatingBackendSpanExporter(
            api_key=tracing_api_key,
            max_field_bytes=max_field_bytes,
        )
        openai_processor = BatchTraceProcessor(exporter=openai_exporter)
        set_trace_processors([openai_processor])
    else:
        logging.warning("No OpenAI tracing API key found; OpenAI trace export is disabled.")

    tracing_log_file = output_dir / "traces.jsonl"
    nest_asyncio2.apply()

    provider = otel_trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        provider = TracerProvider()
        otel_trace.set_tracer_provider(provider)
    provider.add_span_processor(BatchSpanProcessor(FileSpanExporter(tracing_log_file)))
    if not get_client().auth_check():
        logging.warning("Langfuse authentication failed; Langfuse tracing is disabled.")

    OpenAIAgentsInstrumentor().instrument(exclusive_processor=False)
    logging.info(f"Tracing initialized. Trace file: {tracing_log_file}")
    return openai_processor


@contextmanager
def cache_context(
    cache_dir: Path,
    mode: CacheMode = CacheMode.READ_AND_WRITE,
) -> Generator[None]:
    """Set up cache folder and mode. Stacks with existing cache context."""
    token_cache = _cache_var.set(Cache(folder=cache_dir, mode=mode))
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        if mode == CacheMode.DISABLED:
            logging.info("Cache completely disabled")

        # One-shot migration: convert legacy Embedding_*.json files to the fast
        # single-file numpy cache. Runs only when no embeddings_keys_*.npy files
        # exist yet but JSON files do, so it's a no-op on every subsequent run.
        if mode not in (CacheMode.DISABLED, CacheMode.WRITE_ONLY):
            from qcp_adaptive_termination.embedding.numpy_cache import migrate_json_to_npy  # noqa: PLC0415

            already_migrated = any(cache_dir.glob("embeddings_keys_*.npy"))
            if not already_migrated and any(cache_dir.glob("Embedding_*.json")):
                logging.info("NumpyEmbeddingCache: migrating Embedding_*.json files — this runs once")
                migrate_json_to_npy(cache_dir)

        yield
    finally:
        _cache_var.reset(token_cache)


@contextmanager
def logging_context(output_dir: Path) -> Generator[None]:
    """Set up logging to file + stderr. Stacks with existing logging context."""
    token_dir = _output_dir_var.set(output_dir)
    token_active = _logging_active_var.set(True)
    file_handler: logging.FileHandler | None = None

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        log_file = output_dir / "stdout.log"
        logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
        file_handler = logging.FileHandler(log_file)
        logging.getLogger().addHandler(file_handler)
        logging.info(f"Logging initialized. Log file: {log_file}")

        yield
    finally:
        if file_handler is not None:
            logging.getLogger().removeHandler(file_handler)
            file_handler.close()
        _output_dir_var.reset(token_dir)
        _logging_active_var.reset(token_active)


@contextmanager
def tracing_context(name: str, output_dir: Path, *, metadata: dict[str, JsonValue] | None = None) -> Generator[None]:
    """Set up OTel file exporter + agents trace. Stacks with existing tracing context."""
    load_dotenv(override=True)
    openai_processor = _setup_tracing(output_dir)

    metadata_str = {k: str(v) for k, v in (metadata or {}).items()}
    langfuse = get_client()

    try:
        with (
            langfuse.start_as_current_observation(
                name=name, input=serialize_data_for_tracing(metadata), metadata=metadata_str
            ),
            trace(name, metadata=metadata_str),
        ):
            yield
    finally:
        shutdown_timeout = float(os.getenv("TRACING_SHUTDOWN_TIMEOUT", "10.0"))

        if openai_processor is not None:
            openai_processor.shutdown(timeout=shutdown_timeout)

        provider = otel_trace.get_tracer_provider()
        if isinstance(provider, TracerProvider):
            provider.force_flush(timeout_millis=int(shutdown_timeout * 1000))
            provider.shutdown()

        langfuse.flush()
        langfuse.shutdown()

        atexit.unregister(vars(_agents_tracing_setup)["_shutdown_global_trace_provider"])


@contextmanager
def data_context(data_dir: Path) -> Generator[None]:
    """Set the data directory used to resolve agent paths."""
    token = _data_dir_var.set(data_dir)
    try:
        yield
    finally:
        _data_dir_var.reset(token)


@contextmanager
def config_context(cfg: Config) -> Generator[Config]:
    """Activate a Config for the duration of the block. Stacks with existing config context."""
    token = _config_var.set(cfg)
    try:
        yield cfg
    finally:
        _config_var.reset(token)


@contextmanager
def pipeline_context(
    name: str | None = None,
    trace_name: str | None = None,
    output_dir: Path | None = None,
    cache_dir: Path | None = None,
    data_dir: Path | None = None,
    cache_mode: CacheMode = CacheMode.READ_AND_WRITE,
    config: Config | None = None,
    *,
    trace_metadata: dict[str, JsonValue] | None = None,
) -> Generator[None]:
    """Activate logging, tracing, cache, data dir, and config in one block. Skips already-active ones."""
    load_dotenv(override=True)

    datetime_str = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    if name is None:
        name = datetime_str

    if trace_name is None:
        trace_name = name

    if output_dir is None:
        output_dir = _DEFAULT_OUTPUT_DIR / datetime_str

    if cache_dir is None:
        cache_dir = _DEFAULT_CACHE_DIR

    if config is None:
        config = Config()

    trace_metadata = {
        "name": name,
        "output_dir": str(output_dir),
        "cache_dir": str(cache_dir),
        "data_dir": str(data_dir) if data_dir else None,
        "cache_mode": cache_mode.value,
        "config": config.model_dump_json(),
        **(trace_metadata or {}),
    }

    with contextlib.ExitStack() as stack:
        if not _is_logging_active():
            stack.enter_context(logging_context(output_dir))
        if not _is_tracing_active():
            stack.enter_context(tracing_context(trace_name, output_dir, metadata=trace_metadata))
        if not _is_cache_active():
            stack.enter_context(cache_context(cache_dir, cache_mode))
        if not _config_is_active():
            stack.enter_context(config_context(config))
        if not _is_data_dir_active() and data_dir is not None:
            stack.enter_context(data_context(data_dir))

        yield

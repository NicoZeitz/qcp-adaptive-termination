"""Context managers for logging, tracing, cache, and config setup, composed via `pipeline_context`."""

import atexit
import io
import logging
import logging.config
import os
import subprocess
import sys
import time
from collections.abc import Generator
from contextlib import ExitStack, contextmanager, redirect_stdout
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import IO, Literal

import agents.tracing.setup as _agents_tracing_setup
import nest_asyncio2
from agents import get_current_trace, set_trace_processors, set_tracing_export_api_key, trace
from agents.tracing.processors import BatchTraceProcessor
from dotenv import load_dotenv
from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import SpanLimits, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from pydantic import JsonValue
lazy from langfuse import get_client

from qcp_adaptive_termination.models import CacheMode, Config
from qcp_adaptive_termination.observability.cache import Cache
from qcp_adaptive_termination.observability.file_span_exporter import FileSpanExporter
from qcp_adaptive_termination.observability.openai_tracing import TruncatingBackendSpanExporter
from qcp_adaptive_termination.observability.tracing import serialize_data_for_tracing

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
_DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / ".logs"
_DEFAULT_CACHE_DIR = _PROJECT_ROOT / ".cache"

_cache_var: ContextVar[Cache | None] = ContextVar("cache", default=None)
_config_var: ContextVar[Config | None] = ContextVar("config", default=None)
_data_dir_var: ContextVar[Path | None] = ContextVar("data_dir", default=None)
_output_dir_var: ContextVar[Path | None] = ContextVar("output_dir", default=None)
_logging_active_var: ContextVar[bool] = ContextVar("logging_active", default=False)
_tracing_providers_var: ContextVar[list[Literal["openai", "file", "langfuse"]] | None] = ContextVar(
    "tracing_providers", default=None
)


class _Tee(io.TextIOBase):
    """Write the same data to multiple streams (e.g. a file and real stdout)."""

    def __init__(self, *streams: IO[str]) -> None:
        """Initialize the tee with the streams to mirror into."""
        super().__init__()
        self.streams = streams

    def write(self, data: str) -> int:
        """Write data to all wrapped streams."""
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        """Flush all wrapped streams."""
        for stream in self.streams:
            if not getattr(stream, "closed", False):
                stream.flush()


def get_cache() -> Cache | None:
    """Return the active cache, or None if not set."""
    return _cache_var.get()


def get_config() -> Config:
    """Return the active config, creating a default one if not set."""
    cfg = _config_var.get()
    if cfg is None:
        cfg = Config()
        _config_var.set(cfg)
    return cfg


def get_data_dir() -> Path | None:
    """Return the active data directory, or None if not set."""
    return _data_dir_var.get()


def get_output_dir() -> Path | None:
    """Return the active output directory, or None if not set."""
    return _output_dir_var.get()


def get_tracing_providers() -> list[Literal["openai", "file", "langfuse"]]:
    """Return the enabled tracing providers."""
    providers = _tracing_providers_var.get()
    return providers or []


def _is_cache_active() -> bool:
    """Return whether a cache context is active."""
    return _cache_var.get() is not None


def _config_is_active() -> bool:
    """Return whether a config context is active."""
    return _config_var.get() is not None


def _is_logging_active() -> bool:
    """Return whether the logging context is active."""
    return _logging_active_var.get()


def _is_tracing_active() -> bool:
    """Return whether a trace is active."""
    return get_current_trace() is not None


def _is_data_dir_active() -> bool:
    """Return whether a data-dir context is active."""
    return _data_dir_var.get() is not None


@contextmanager
def cache_context(
    cache_dir: Path,
    mode: CacheMode = CacheMode.READ_AND_WRITE,
) -> Generator[None]:
    """Set up cache folder and mode. Stacks with existing cache context.

    Args:
        cache_dir: Directory to store the cache in
        mode: Cache read/write mode
    """
    token_cache = _cache_var.set(Cache(folder=cache_dir, mode=mode))
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        if mode == CacheMode.DISABLED:
            logging.info("Cache completely disabled")

        yield
    finally:
        _cache_var.reset(token_cache)


def configure_logging(output_dir: Path) -> Path:
    """Apply ``logging.conf`` so logging writes to ``<output_dir>/stderr.log``.

    Args:
        output_dir: Directory to write the log file into

    Returns:
        (Path): Resolved stderr log path
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stderr_log = output_dir / "stderr.log"
    logging.config.fileConfig(
        _PROJECT_ROOT / "logging.conf",
        defaults={"logfile": str(stderr_log)},
        disable_existing_loggers=False,
    )
    return stderr_log


@contextmanager
def logging_context(output_dir: Path) -> Generator[None]:
    """Set up logging and output capture in one block.

    Args:
        output_dir: Directory to write stdout and stderr logs into
    """
    token_dir = _output_dir_var.set(output_dir)
    token_active = _logging_active_var.set(True)

    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = output_dir / "stdout.log"

    root = logging.getLogger()
    previous_level = root.level
    previous_handlers = root.handlers[:]
    stderr_log = configure_logging(output_dir)

    _start = time.monotonic()
    with open(stdout_log, "a") as stdout_file, redirect_stdout(_Tee(stdout_file, sys.stdout)):
        try:
            logging.info("Logging initialized. Log file: %s", stderr_log)
            try:
                git_log = subprocess.check_output(
                    ["git", "log", "-1", "--oneline"], text=True, stderr=subprocess.DEVNULL
                ).strip()
                logging.info("Git commit: %s", git_log)
            except Exception:
                pass

            yield
        finally:
            elapsed = time.monotonic() - _start
            logging.info("Execution time: %.2fs", elapsed)
            for handler in root.handlers:
                handler.close()
            root.handlers = previous_handlers
            root.setLevel(previous_level)
            _output_dir_var.reset(token_dir)
            _logging_active_var.reset(token_active)


@contextmanager
def tracing_context(
    name: str,
    output_dir: Path,
    *,
    metadata: dict[str, JsonValue] | None = None,
    tracing_providers: list[Literal["openai", "file", "langfuse"]] | None = None,
) -> Generator[None]:
    """Set up tracing providers and an agents trace. Stacks with existing tracing context.

    Args:
        name: Name for the trace
        output_dir: Directory to write trace files into
        metadata: Metadata to attach to the trace
        tracing_providers: Tracing backends to enable
    """
    load_dotenv(override=True)

    if tracing_providers is None:
        tracing_providers = ["file"]
    token = _tracing_providers_var.set(tracing_providers)
    metadata_str = {k: str(v) for k, v in (metadata or {}).items()}
    langfuse_observation = None

    if "openai" in tracing_providers:
        tracing_api_key = os.getenv("OPENAI_API_KEY")
        if not tracing_api_key:
            _tracing_providers_var.reset(token)
            raise ValueError("OPENAI_API_KEY is required for OpenAI tracing.")

        set_tracing_export_api_key(tracing_api_key)
        max_field_bytes = int(os.getenv("OPENAI_TRACING_MAX_FIELD_BYTES", "9500"))
        openai_exporter = TruncatingBackendSpanExporter(
            api_key=tracing_api_key,
            max_field_bytes=max_field_bytes,
        )
        openai_processor = BatchTraceProcessor(exporter=openai_exporter)
        set_trace_processors([openai_processor])
    else:
        # reset default openai processor
        set_trace_processors([])

    if "file" in tracing_providers or "langfuse" in tracing_providers:
        nest_asyncio2.apply()
        OpenAIAgentsInstrumentor().instrument(exclusive_processor=False)

        provider = otel_trace.get_tracer_provider()
        if not isinstance(provider, TracerProvider):
            provider = TracerProvider(span_limits=SpanLimits(max_attributes=4096))
            otel_trace.set_tracer_provider(provider)

        if "file" in tracing_providers:
            tracing_log_file = output_dir / "traces.jsonl"
            provider.add_span_processor(BatchSpanProcessor(FileSpanExporter(tracing_log_file)))
            logging.info(f"Tracing initialized. Trace file: {tracing_log_file}")

        if "langfuse" in tracing_providers:
            client = get_client()

            if not client.auth_check():
                _tracing_providers_var.reset(token)
                raise ValueError("Langfuse client authentication failed.")

            langfuse_observation = client.start_as_current_observation(
                name=name,
                input=serialize_data_for_tracing(metadata),
                metadata=metadata_str,
            )
            langfuse_observation.__enter__()

    try:
        with trace(name, metadata=metadata_str):
            yield
    finally:
        if "openai" in tracing_providers:
            atexit.unregister(vars(_agents_tracing_setup)["_shutdown_global_trace_provider"])

        if "langfuse" in tracing_providers:
            assert langfuse_observation is not None
            langfuse_observation.__exit__(None, None, None)

            client = get_client()
            client.flush()
            client.shutdown()

        provider = otel_trace.get_tracer_provider()
        if isinstance(provider, TracerProvider):
            provider.force_flush(timeout_millis=5000)
            provider.shutdown()

        _tracing_providers_var.reset(token)


@contextmanager
def data_context(data_dir: Path) -> Generator[None]:
    """Set the data directory used to resolve agent paths.

    Args:
        data_dir: Data directory to resolve agent paths against
    """
    token = _data_dir_var.set(data_dir)
    try:
        yield
    finally:
        _data_dir_var.reset(token)


@contextmanager
def config_context(cfg: Config) -> Generator[Config]:
    """Activate a Config for the duration of the block. Stacks with existing config context.

    Args:
        cfg: Config to activate
    """
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
    tracing_providers: list[Literal["openai", "file", "langfuse"]] | None = None,
) -> Generator[None]:
    """Activate logging, tracing, cache, data dir, and config in one block. Skips already-active ones.

    Args:
        name: Name for the run
        trace_name: Name for the trace
        output_dir: Directory for run logs and traces
        cache_dir: Directory for the cache
        data_dir: Data directory used to resolve agent paths
        cache_mode: Cache read/write mode
        config: Config to activate
        trace_metadata: Metadata to attach to the trace
        tracing_providers: Tracing backends to enable
    """
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

    with ExitStack() as stack:
        if not _is_logging_active():
            stack.enter_context(logging_context(output_dir))
        if not _is_tracing_active():
            stack.enter_context(
                tracing_context(
                    trace_name,
                    output_dir,
                    metadata=trace_metadata,
                    tracing_providers=tracing_providers,
                )
            )
        if not _is_cache_active():
            stack.enter_context(cache_context(cache_dir, cache_mode))
        if not _config_is_active():
            stack.enter_context(config_context(config))
        if not _is_data_dir_active() and data_dir is not None:
            stack.enter_context(data_context(data_dir))

        yield

"""Shared orchestration for evaluation scripts.

Runs a per-query worker function across all input files, either in parallel
(process pool) or sequentially, then optionally aggregates the per-query
outputs. Replaces the old bash loop + inline `python3 -c` aggregation.
"""

import argparse
import asyncio
import contextlib
import logging
import os
from collections.abc import Awaitable, Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from qcp_adaptive_termination.models import Config
from qcp_adaptive_termination.observability import context
from qcp_adaptive_termination.pipeline import load_input


async def _run_worker(
    worker: Callable[..., Awaitable[Any]],
    input_file: Path,
    config: Config,
    dataset_dir: Path,
    output_dir: Path,
    cache_dir: Path | None,
    extra: dict[str, Any] | None,
) -> Any:
    """Load input + config for one query and run the worker.

    Each worker runs inside its own logging/tracing/data/config context scoped
    to the per-query output subfolder, so every input file gets its own
    ``traces.jsonl`` and ``stdout.log``/``stderr.log``.
    """
    data, input_extra = load_input(input_file)
    merged_extra = {**(input_extra or {}), **(extra or {})}
    query_slug = input_file.stem
    run_output_dir = output_dir / query_slug
    with (
        context.logging_context(run_output_dir),
        context.tracing_context(f"[QCP-EVAL] {query_slug}", run_output_dir),
        context.data_context(dataset_dir.parent),
        context.config_context(config),
        context.cache_context(cache_dir) if cache_dir is not None else contextlib.nullcontext(),
    ):
        return await worker(
            input_file=input_file,
            config=config,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            cache_dir=cache_dir,
            data=data,
            extra=merged_extra,
        )


async def run_parallel(
    worker: Callable[..., Awaitable[Any]],
    dataset_dir: Path,
    config: Config,
    output_dir: Path,
    cache_dir: Path | None = None,
    *,
    parallel: bool = True,
    max_workers: int | None = None,
    extra: dict[str, Any] | None = None,
    query: str | None = None,
) -> list[Any]:
    """Run `worker` for every input file in ``dataset_dir/_meta/input``.

    If ``query`` is given, only files whose stem contains it (case-insensitive)
    are run. Returns worker results in input-file order.
    """
    input_dir = dataset_dir / "_meta" / "input"
    input_files = sorted(input_dir.glob("*.json"))
    if query:
        needle = query.lower()
        input_files = [f for f in input_files if needle in f.stem.lower()]
    if not input_files:
        logging.warning("No input files found in %s", input_dir)
        return []

    if max_workers is None:
        max_workers = min(os.cpu_count() or 1, 6)
    workers = max_workers if parallel else 1
    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1) as pool:
        futures = [
            loop.run_in_executor(
                pool,
                _run_worker_sync,
                worker,
                input_file,
                config,
                dataset_dir,
                output_dir,
                cache_dir,
                extra,
            )
            for input_file in input_files
        ]
        results = await asyncio.gather(*futures, return_exceptions=True)
        for input_file, result in zip(input_files, results, strict=True):
            if isinstance(result, BaseException):
                logging.error("Worker failed for %s: %s", input_file.name, result)
        return results


def _run_worker_sync(
    worker: Callable[..., Awaitable[Any]],
    input_file: Path,
    config: Config,
    dataset_dir: Path,
    output_dir: Path,
    cache_dir: Path | None,
    extra: dict[str, Any] | None,
) -> Any:
    """Process-pool entry point: run the async worker in a fresh event loop."""
    return asyncio.run(_run_worker(worker, input_file, config, dataset_dir, output_dir, cache_dir, extra))


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add the shared CLI args used by all evaluation subcommands."""
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        required=True,
        help="Dataset folder containing _meta/input (query JSONs) and _meta/meta_mapping.yaml",
    )
    parser.add_argument(
        "--output_dir", type=Path, required=True, help="Directory to write per-query + aggregate results"
    )
    parser.add_argument("--cache_dir", type=Path, default=None, help="Directory for cache files")
    parser.add_argument("--config", type=str, default=None, help="Config as a YAML file path or inline YAML string")
    parser.add_argument("--parallel", action="store_true", default=True, help="Run queries in parallel (default)")
    parser.add_argument("--sequential", action="store_true", help="Run queries sequentially")
    parser.add_argument("--max_workers", type=int, default=None, help="Max parallel worker processes")
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Only run input queries whose filename stem contains this substring (case-insensitive).",
    )

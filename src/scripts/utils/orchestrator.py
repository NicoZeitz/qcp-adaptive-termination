"""Shared orchestration for evaluation scripts.

Runs a per-query worker function across all input files, in parallel or
sequentially.
"""

import asyncio
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
) -> Any:
    """Load input and config for one query and run the worker."""
    data, extra = load_input(input_file)
    query_slug = input_file.stem
    run_output_dir = output_dir / query_slug
    cache_dir = Path(cache_dir) if cache_dir is not None else None
    with context.pipeline_context(
        name=f"query-{query_slug}",
        trace_name=f"[QCP-EVAL] {query_slug}",
        output_dir=run_output_dir,
        cache_dir=cache_dir,
        data_dir=dataset_dir.parent,
        config=config,
    ):
        return await worker(
            input_file=input_file,
            config=config,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            cache_dir=cache_dir,
            data=data,
            extra=extra,
        )


def _run_worker_sync(
    worker: Callable[..., Awaitable[Any]],
    input_file: Path,
    config: Config,
    dataset_dir: Path,
    output_dir: Path,
    cache_dir: Path | None,
) -> Any:
    """Process-pool entry point: run the async worker in a fresh event loop."""
    return asyncio.run(_run_worker(worker, input_file, config, dataset_dir, output_dir, cache_dir))


async def run_parallel(
    worker: Callable[..., Awaitable[Any]],
    input_files: list[Path],
    config: Config,
    dataset_dir: Path,
    output_dir: Path,
    cache_dir: Path | None = None,
    *,
    parallel: bool = True,
    max_workers: int | None = None,
) -> list[Any]:
    """Run `worker` for every input file, in parallel or sequentially.

    Returns worker results in input-file order.
    """
    if not input_files:
        logging.warning("No input files to run.")
        return []

    max_workers = max_workers or os.cpu_count() or 1
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
            )
            for input_file in input_files
        ]
        results = await asyncio.gather(*futures, return_exceptions=True)
        for input_file, result in zip(input_files, results, strict=True):
            if isinstance(result, BaseException):
                logging.error(
                    "Worker failed for %s: %s",
                    input_file.name,
                    result,
                    exc_info=(type(result), result, result.__traceback__),
                )
        return results

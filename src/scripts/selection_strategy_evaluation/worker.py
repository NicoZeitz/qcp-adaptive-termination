"""Selection strategy evaluation worker.

Benchmarks every selection strategy on each multi-agent step of a query plan.
The LLM-heavy parts of the pipeline (align_schema, data_agent, completeness)
are stubbed; only the selection-strategy ranking machinery runs for real.
"""

import asyncio
import contextlib
import logging
import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

import qcp_adaptive_termination.hooks as hooks
from qcp_adaptive_termination.hooks import HookEvent
from qcp_adaptive_termination.models import (
    AlignedResultList,
    Config,
    DataAgent,
    SelectionStrategyType,
)
from qcp_adaptive_termination.models.execution import PartialQueryAnswer
from qcp_adaptive_termination.models.query_plan import (
    MultiAgentQueryPlanStep,
    QueryPlan,
)
from qcp_adaptive_termination.models.stages import CollaborativeQueryAnswering
from qcp_adaptive_termination.observability import context
from qcp_adaptive_termination.stages.execution.multi_agent import (
    answer_step_multiple_agents,
)
from scripts.selection_strategy_evaluation.metrics import (
    EvaluationCurve,
    build_curve,
    compute_greedy_oracle,
    compute_relative_auc,
)
from scripts.selection_strategy_evaluation.summary import StepSummary, write_summary
from scripts.utils.canonical import (
    load_meta_mapping,
    make_align_schema_override,
    make_data_agent_override,
    parse_canonical_table,
    query_agent_canonical,
)
from scripts.utils.overrides import (
    make_align_schema_dispatch,
    make_completeness_override,
    make_data_agent_dispatch,
)
from scripts.utils.sql import parse_select_columns

# Per-strategy dispatch maps: strategy execution_id -> step-specific handler.
# Populated/cleaned per _run_one_strategy call; supports parallel step execution.
_per_strategy_align: dict[str, Callable[..., AlignedResultList]] = {}
_per_strategy_data: dict[str, Callable[..., PartialQueryAnswer]] = {}


def _get_total_unique(
    relevant_agents: list[DataAgent],
    canonical_sql: str,
    canonical_table: str,
    key_cols: list[str],
    meta_mapping: dict[str, Any],
    data_dir: Path,
) -> int:
    with ThreadPoolExecutor() as pool:
        dfs = list(
            pool.map(
                lambda a: query_agent_canonical(a, canonical_sql, canonical_table, meta_mapping, data_dir),
                relevant_agents,
            )
        )
    frames = [df for df in dfs if df is not None and not df.empty]
    if not frames:
        return 0
    combined = pd.concat(frames, ignore_index=True)
    cols = [c for c in key_cols if c in combined.columns]
    if not cols:
        cols = list(combined.columns)
    return int(len(combined[cols].drop_duplicates()))


def _register_agent_tracking(target_execution_id: str) -> tuple[list[str], Any]:
    ordered: list[str] = []

    def _on_agent_query(
        execution_id: str,
        data_agent: DataAgent,
        query: str,
        result: PartialQueryAnswer,
    ) -> None:
        if execution_id == target_execution_id:
            ordered.append(data_agent.name)

    hooks.register_post(HookEvent.EXECUTION_QUERY_DATA_AGENT, _on_agent_query)
    return ordered, _on_agent_query


async def _run_one_strategy(
    strategy: SelectionStrategyType,
    step: MultiAgentQueryPlanStep,
    align_handler: Callable[..., AlignedResultList],
    data_handler: Callable[..., PartialQueryAnswer],
    all_agents: list[DataAgent],
    data_cache: dict[str, pd.DataFrame | None],
    total_unique: int,
    canonical_cols: list[str],
    config: Config,
) -> tuple[str, EvaluationCurve | None, str | None]:
    logging.info("Running strategy: %s / %s", step.id, strategy)
    execution_id = uuid4().hex[:12]
    strategy_config = config.model_copy(update={"selection_type": strategy})
    ordered_agents, agent_query_handler = _register_agent_tracking(execution_id)
    # Benchmark the step in isolation: dependencies dropped, canonical SQL self-contained.
    isolated_step = step.model_copy(update={"dependencies": []})

    _per_strategy_align[execution_id] = align_handler
    _per_strategy_data[execution_id] = data_handler

    try:
        with hooks.execution_context(execution_id), context.config_context(strategy_config):
            await answer_step_multiple_agents(
                data_agents=all_agents,
                query_plan_step=isolated_step,
                additional_information=[],
                step_budgets={},
            )
            logging.info("Strategy %s/%s completed", step.id, strategy)
    except Exception as exc:
        logging.warning("Strategy %s/%s failed: %s", step.id, strategy, exc)
        logging.exception(exc)
        return str(strategy), None, f"{type(exc).__name__}: {exc}"
    finally:
        _per_strategy_align.pop(execution_id, None)
        _per_strategy_data.pop(execution_id, None)
        hooks.unregister_post(HookEvent.EXECUTION_QUERY_DATA_AGENT, agent_query_handler)

    seen: set[str] = set()
    deduped: list[str] = []
    for name in ordered_agents:
        if name not in seen:
            seen.add(name)
            deduped.append(name)

    if not deduped:
        logging.warning("Strategy %s/%s: no agents queried", step.id, strategy)
        return str(strategy), None, "no agents queried"

    curve = build_curve(deduped, data_cache, total_unique, canonical_cols)
    logging.info("  %s/%s AUC=%.4f agents=%d", step.id, strategy, curve.auc, len(deduped))
    return str(strategy), curve, None


def _run_strategy_worker(
    strategy: SelectionStrategyType,
    step: MultiAgentQueryPlanStep,
    step_meta: dict[str, Any],
    canonical_sql: str,
    canonical_table: str,
    canonical_cols: list[str],
    all_agents: list[DataAgent],
    data_cache: dict[str, pd.DataFrame | None],
    total_unique: int,
    meta_mapping: dict[str, Any],
    data_dir: str,
    query: str,
    config: Config,
    cache_dir: str | None,
    output_dir: str,
) -> tuple[str, EvaluationCurve | None, str | None]:
    """Run one strategy in a fresh worker process (its own hooks + context).

    The strategy machinery is process-local: each worker registers its own
    ALIGN_SCHEMA / DATA_AGENT / COMPLETENESS overwrites and its own agent
    query-tracking post hook, so strategies can run concurrently across cores.
    """
    _per_strategy_align.clear()
    _per_strategy_data.clear()
    align_dispatch = make_align_schema_dispatch(_per_strategy_align)
    data_dispatch = make_data_agent_dispatch(_per_strategy_data)
    completeness_override = make_completeness_override()
    hooks.register_overwrite(
        HookEvent.EXECUTION_EXECUTE_QUERY_PLAN_EXECUTE_STEP_ANSWER_STEP_MULTIPLE_AGENTS_ALIGN_SCHEMA,
        align_dispatch,
    )
    hooks.register_overwrite(HookEvent.EXECUTION_QUERY_DATA_AGENT, data_dispatch)
    hooks.register_overwrite(
        HookEvent.EXECUTION_EXECUTE_QUERY_PLAN_EXECUTE_STEP_ANSWER_STEP_MULTIPLE_AGENTS_COMPLETENESS,
        completeness_override,
    )

    data_dir_path = Path(data_dir)
    align_handler = make_align_schema_override(step_meta, query)
    data_handler = make_data_agent_override(canonical_sql, canonical_table, meta_mapping, data_cache, data_dir_path)

    # Route this worker's logs to the same stderr log as the parent without the
    # redirect/tee-based `logging_context` (which corrupts shared handlers when a
    # pool worker is reused). Use a plain append FileHandler instead. In a fresh
    # worker process the root logger defaults to WARNING, so raise it to INFO.
    log_file = Path(output_dir) / "stderr.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    root_logger.addHandler(file_handler)

    with (
        context.config_context(config),
        context.data_context(data_dir_path),
        context.cache_context(Path(cache_dir)) if cache_dir is not None else contextlib.nullcontext(),
    ):
        try:
            return asyncio.run(
                _run_one_strategy(
                    strategy=strategy,
                    step=step,
                    align_handler=align_handler,
                    data_handler=data_handler,
                    all_agents=all_agents,
                    data_cache=data_cache,
                    total_unique=total_unique,
                    canonical_cols=canonical_cols,
                    config=config,
                )
            )
        except Exception as exc:
            logging.exception("Strategy %s/%s crashed in worker", step.id, strategy)
            return str(strategy), None, f"{type(exc).__name__}: {exc}"
        finally:
            logging.getLogger().removeHandler(file_handler)
            file_handler.close()
            hooks.unregister_overwrite(
                HookEvent.EXECUTION_EXECUTE_QUERY_PLAN_EXECUTE_STEP_ANSWER_STEP_MULTIPLE_AGENTS_ALIGN_SCHEMA,
                align_dispatch,
            )
            hooks.unregister_overwrite(HookEvent.EXECUTION_QUERY_DATA_AGENT, data_dispatch)
            hooks.unregister_overwrite(
                HookEvent.EXECUTION_EXECUTE_QUERY_PLAN_EXECUTE_STEP_ANSWER_STEP_MULTIPLE_AGENTS_COMPLETENESS,
                completeness_override,
            )


def _default_strategy_workers() -> int:
    """Default number of parallel strategy worker processes per step."""
    return max(1, min(os.cpu_count() or 1, 4))


async def _run_strategies(
    *,
    step: MultiAgentQueryPlanStep,
    step_meta: dict[str, Any],
    canonical_sql: str,
    canonical_table: str,
    canonical_cols: list[str],
    all_agents: list[DataAgent],
    data_cache: dict[str, pd.DataFrame | None],
    total_unique: int,
    meta_mapping: dict[str, Any],
    data_dir: Path,
    query: str,
    config: Config,
    cache_dir: Path | None,
    output_dir: Path,
    strategy_workers: int,
) -> list[tuple[str, EvaluationCurve | None, str | None]]:
    """Run all selection strategies, in parallel worker processes when enabled.

    Each strategy can run on its own core: agent-registry, sketch, and embedding
    precomputes are CPU heavy and previously serialized inside one process.
    Falls back to in-process `asyncio.gather` for ``strategy_workers <= 1``.
    """
    if strategy_workers <= 1:
        align_handler = make_align_schema_override(step_meta, query)
        data_handler = make_data_agent_override(canonical_sql, canonical_table, meta_mapping, data_cache, data_dir)
        return list(
            await asyncio.gather(
                *[
                    _run_one_strategy(
                        strategy=s,
                        step=step,
                        align_handler=align_handler,
                        data_handler=data_handler,
                        all_agents=all_agents,
                        data_cache=data_cache,
                        total_unique=total_unique,
                        canonical_cols=canonical_cols,
                        config=config,
                    )
                    for s in SelectionStrategyType
                ]
            )
        )

    loop = asyncio.get_running_loop()
    # One strategy per worker: each registers its own hooks/context/logging, so a
    # fresh process avoids stale handlers and tracing re-entry (same rationale as
    # the outer orchestrator's max_tasks_per_child=1).
    with ProcessPoolExecutor(max_workers=strategy_workers, max_tasks_per_child=1) as pool:
        futures = [
            loop.run_in_executor(
                pool,
                _run_strategy_worker,
                s,
                step,
                step_meta,
                canonical_sql,
                canonical_table,
                canonical_cols,
                all_agents,
                data_cache,
                total_unique,
                meta_mapping,
                str(data_dir),
                query,
                config,
                str(cache_dir) if cache_dir is not None else None,
                str(output_dir),
            )
            for s in SelectionStrategyType
        ]
        return list(await asyncio.gather(*futures))


async def _run_step(
    step: MultiAgentQueryPlanStep,
    step_meta: dict[str, Any],
    relevant_data_agents: list[DataAgent],
    all_agents: list[DataAgent],
    meta_mapping: dict[str, Any],
    query: str,
    query_slug: str,
    output_dir: Path,
    data_dir: Path,
    config: Config,
    cache_dir: Path | None = None,
    strategy_workers: int | None = None,
) -> StepSummary | None:
    canonical_sql: str = step_meta["sql_query"]
    schema_meta = step_meta["aligned_schema_metadata"]
    key_cols: list[str] = schema_meta["keys"]

    canonical_table = parse_canonical_table(canonical_sql)
    if canonical_table is None:
        logging.warning(
            "%s/%s: could not parse table from sql %r — skipping",
            query_slug,
            step.id,
            canonical_sql,
        )
        return None
    canonical_cols = parse_select_columns(canonical_sql)
    if not canonical_cols:
        logging.warning(
            "%s/%s: could not parse columns from sql %r — skipping",
            query_slug,
            step.id,
            canonical_sql,
        )
        return None

    logging.info(
        "Query: %s | Step: %s | table: %s | cols: %s | key_cols: %s",
        query,
        step.id,
        canonical_table,
        canonical_cols,
        key_cols,
    )

    step_agent_names = set(step.to_agent)
    step_agents = [a for a in relevant_data_agents if a.name in step_agent_names]

    total_unique = _get_total_unique(step_agents, canonical_sql, canonical_table, key_cols, meta_mapping, data_dir)
    logging.info("Total unique rows (ground truth, %d agents): %d", len(step_agents), total_unique)

    data_cache: dict[str, pd.DataFrame | None] = {}
    with ThreadPoolExecutor() as pool:
        futures = {
            agent.name: pool.submit(
                query_agent_canonical, agent, canonical_sql, canonical_table, meta_mapping, data_dir
            )
            for agent in step_agents
        }
        for name, future in futures.items():
            data_cache[name] = future.result()

    strategy_workers = strategy_workers or _default_strategy_workers()
    strategy_runs = await _run_strategies(
        step=step,
        step_meta=step_meta,
        canonical_sql=canonical_sql,
        canonical_table=canonical_table,
        canonical_cols=canonical_cols,
        all_agents=all_agents,
        data_cache=data_cache,
        total_unique=total_unique,
        meta_mapping=meta_mapping,
        data_dir=data_dir,
        query=query,
        config=config,
        cache_dir=cache_dir,
        output_dir=output_dir,
        strategy_workers=strategy_workers,
    )
    results = [(strategy, curve) for strategy, curve, _ in strategy_runs if curve is not None]
    failures = [(strategy, error or "unknown failure") for strategy, curve, error in strategy_runs if curve is None]

    oracle_curve = compute_greedy_oracle(step.to_agent, data_cache, total_unique, canonical_cols)
    logging.info("Oracle AUC=%.4f", oracle_curve.auc)
    _save_results(results, oracle_curve, query_slug, step.id, output_dir)

    curves = {strategy: curve for strategy, curve in results}
    curves["oracle"] = oracle_curve
    return StepSummary(
        query_slug=query_slug,
        step_id=step.id,
        agents_total=len(step_agents),
        oracle_auc=oracle_curve.auc,
        final_row_count=total_unique,
        curves=curves,
        failures=failures,
    )


def _save_results(
    results: list[tuple[str, EvaluationCurve]],
    oracle_curve: EvaluationCurve,
    query_slug: str,
    step_id: str,
    output_dir: Path,
) -> None:
    """Write per-step AUC/curve CSVs and plots."""
    from scripts.selection_strategy_evaluation.plotting import plot_step_curves

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{query_slug}_{step_id}"

    rows = [
        {"strategy": s, "auc": c.auc, "relative_auc": compute_relative_auc(c.auc, oracle_curve.auc)} for s, c in results
    ]
    rows.append({"strategy": "oracle", "auc": oracle_curve.auc, "relative_auc": 1.0})
    pd.DataFrame(rows).to_csv(output_dir / f"{prefix}_auc.csv", index=False)

    curve_rows = []
    for strategy, curve in results:
        total = len(curve.x) - 1
        for x, y in zip(curve.x, curve.y, strict=True):
            curve_rows.append({"strategy": strategy, "agents_queried": x * total, "agents_fraction": x, "coverage": y})
    for x, y in zip(oracle_curve.x, oracle_curve.y, strict=True):
        curve_rows.append(
            {
                "strategy": "oracle",
                "agents_queried": x * (len(oracle_curve.x) - 1),
                "agents_fraction": x,
                "coverage": y,
            }
        )
    pd.DataFrame(curve_rows).to_csv(output_dir / f"{prefix}.csv", index=False)

    plot_step_curves(results, oracle_curve, query_slug, step_id, output_dir)
    logging.info("Saved to %s", output_dir)


async def run_query(
    *,
    input_file: Path,
    config: Config,
    dataset_dir: Path,
    output_dir: Path,
    cache_dir: Path | None,
    data: CollaborativeQueryAnswering,
    extra: dict[str, Any] | None,
) -> None:
    """Evaluate all selection strategies for one input query (worker)."""
    extra = extra or {}
    steps_metadata: dict[str, Any] = extra.get("steps_metadata", {})

    raw_query_plan = extra.get("query_plan")
    if raw_query_plan is None:
        logging.warning("%s: no query_plan in extra — skipping", input_file.stem)
        return

    strategy_workers: int | None = extra.get("strategy_workers")

    query_plan = QueryPlan.model_validate(raw_query_plan)
    multi_steps = [s for s in query_plan.steps if isinstance(s, MultiAgentQueryPlanStep)]

    if not multi_steps:
        logging.warning("%s: no multi-agent steps — skipping", input_file.stem)
        return

    data_dir = dataset_dir.parent
    meta_mapping = load_meta_mapping(dataset_dir)
    all_agents = data.data_agents
    relevant_agent_names: set[str] = set(extra.get("relevant_data_agents", []))
    relevant_data_agents = [a for a in all_agents if a.name in relevant_agent_names]
    query = query_plan.query
    query_slug = input_file.stem
    run_output_dir = output_dir / query_slug

    # Single dispatch handlers for ALIGN_SCHEMA and DATA_AGENT overrides.
    # These route each call to the correct per-step handler via execution_id.
    align_dispatch = make_align_schema_dispatch(_per_strategy_align)
    data_dispatch = make_data_agent_dispatch(_per_strategy_data)
    completeness_override = make_completeness_override()
    hooks.register_overwrite(
        HookEvent.EXECUTION_EXECUTE_QUERY_PLAN_EXECUTE_STEP_ANSWER_STEP_MULTIPLE_AGENTS_ALIGN_SCHEMA,
        align_dispatch,
    )
    hooks.register_overwrite(HookEvent.EXECUTION_QUERY_DATA_AGENT, data_dispatch)
    hooks.register_overwrite(
        HookEvent.EXECUTION_EXECUTE_QUERY_PLAN_EXECUTE_STEP_ANSWER_STEP_MULTIPLE_AGENTS_COMPLETENESS,
        completeness_override,
    )

    step_tasks = []
    for step in multi_steps:
        step_meta = steps_metadata.get(step.id)
        if step_meta is None:
            logging.warning("%s/%s: no steps_metadata — skipping", input_file.stem, step.id)
            continue
        step_tasks.append(
            _run_step(
                step=step,
                step_meta=step_meta,
                relevant_data_agents=relevant_data_agents,
                all_agents=all_agents,
                meta_mapping=meta_mapping,
                query=query,
                query_slug=query_slug,
                output_dir=run_output_dir,
                data_dir=data_dir,
                config=config,
                cache_dir=cache_dir,
                strategy_workers=strategy_workers,
            )
        )

    try:
        gathered = await asyncio.gather(*step_tasks)
        step_summaries = [s for s in gathered if s is not None]
    finally:
        hooks.unregister_overwrite(
            HookEvent.EXECUTION_EXECUTE_QUERY_PLAN_EXECUTE_STEP_ANSWER_STEP_MULTIPLE_AGENTS_ALIGN_SCHEMA,
            align_dispatch,
        )
        hooks.unregister_overwrite(HookEvent.EXECUTION_QUERY_DATA_AGENT, data_dispatch)
        hooks.unregister_overwrite(
            HookEvent.EXECUTION_EXECUTE_QUERY_PLAN_EXECUTE_STEP_ANSWER_STEP_MULTIPLE_AGENTS_COMPLETENESS,
            completeness_override,
        )

    # Summary over all steps of this file.
    write_summary(step_summaries, run_output_dir, "summary_all", " — all steps")
    # Summary excluding find steps (single-row result sets, ill-posed for overlap).
    non_find = [s for s in step_summaries if not s.is_find_step]
    write_summary(non_find, run_output_dir, "summary_no_find", " — excluding find steps")

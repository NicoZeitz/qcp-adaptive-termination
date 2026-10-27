"""k-warmup worker: for one table, run the 3 orderings and build estimator trajectories.

For a single-step ``SELECT * FROM <table>`` plan, each ordering (random, posterior,
count) ranks the table's agents, their rows are folded incrementally into an
estimator, and the estimator trajectory plus flatness/error k are recorded.
"""

import asyncio
import contextlib
import logging
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
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
from scripts.k_warmup.metrics import (
    EstimatorTrajectory,
    MetaMapping,
    build_estimator_trajectory,
    canonical_key_cols,
    error_k,
    flatness_k,
    load_agent_cache,
    select_all_sql,
    true_count_from_cache,
)
from scripts.utils.canonical import (
    load_meta_mapping,
    make_align_schema_override,
    make_data_agent_override,
    parse_canonical_table,
)
from scripts.utils.overrides import (
    make_align_schema_dispatch,
    make_completeness_override,
    make_data_agent_dispatch,
)

STRATEGIES: list[SelectionStrategyType] = [
    SelectionStrategyType.RANDOM,
    SelectionStrategyType.POSTERIOR,
    SelectionStrategyType.COUNT,
]
"""The orderings benchmarked by the k-warmup study."""

FLATNESS_THRESHOLDS = [0.05, 0.10]
"""Relative one-step N_hat change thresholds defining 'flat'."""
ERROR_THRESHOLDS = [0.05, 0.10, 0.20]
"""Relative error-vs-true thresholds used for reference convergence k."""

# Per-execution dispatch maps (mirror selection_strategy_evaluation).
_per_run_align: dict[str, Callable[..., AlignedResultList]] = {}
_per_run_data: dict[str, Callable[..., PartialQueryAnswer]] = {}


@dataclass(slots=True)
class TableSummary:
    """k-warmup outcome for one table."""

    table: str
    """Canonical table name"""
    agents_total: int
    """Number of agents containing the table"""
    true_count: int
    """True distinct count across all agents"""
    trajectories: dict[str, EstimatorTrajectory]
    """Estimator trajectory per ordering"""
    failures: dict[str, str]
    """Ordering/error pairs for failed runs"""


def _register_agent_tracking(target_execution_id: str) -> tuple[list[str], Any]:
    """Register a hook that records the queried agent order for a run."""
    ordered: list[str] = []

    def _on_agent_query(
        execution_id: str,
        data_agent: DataAgent,
        query: str,
        result: PartialQueryAnswer,
    ):
        """Append the queried agent for the target execution."""
        if execution_id == target_execution_id:
            ordered.append(data_agent.name)

    hooks.register_post(HookEvent.EXECUTION_QUERY_DATA_AGENT, _on_agent_query)
    return ordered, _on_agent_query


def _isolated_step(step: MultiAgentQueryPlanStep) -> MultiAgentQueryPlanStep:
    """Return a single-step copy with dependencies dropped."""
    return step.model_copy(update={"dependencies": []})


async def _run_one_ordering(
    ordering: SelectionStrategyType,
    step: MultiAgentQueryPlanStep,
    align_handler: Any,
    data_handler: Any,
    all_agents: list[DataAgent],
    data_cache: dict[str, pd.DataFrame | None],
    key_cols: list[str],
    true_count: int,
    config: Config,
    table: str,
    output_dir: Path | None = None,
) -> tuple[str, EstimatorTrajectory | None, str | None]:
    """Run one ordering on a table and build its estimator trajectory."""
    logging.info("Running ordering: %s / %s", table, ordering)
    execution_id = uuid4().hex[:12]
    ordering_config = config.model_copy(
        update={"selection": config.selection.model_copy(update={"strategy": ordering})}
    )
    ordered_agents, agent_query_handler = _register_agent_tracking(execution_id)
    isolated = _isolated_step(step)

    _per_run_align[execution_id] = align_handler
    _per_run_data[execution_id] = data_handler

    try:
        with hooks.execution_context(execution_id), context.config_context(ordering_config):
            await answer_step_multiple_agents(
                data_agents=all_agents,
                query_plan_step=isolated,
                additional_information=[],
                step_budgets={},
            )
    except Exception as exc:  # noqa: BLE001
        logging.warning("Ordering %s/%s failed: %s", table, ordering, exc)
        return str(ordering), None, f"{type(exc).__name__}: {exc}"
    finally:
        _per_run_align.pop(execution_id, None)
        _per_run_data.pop(execution_id, None)
        hooks.unregister_post(HookEvent.EXECUTION_QUERY_DATA_AGENT, agent_query_handler)

    seen: set[str] = set()
    deduped: list[str] = []
    for n in ordered_agents:
        if n in seen:
            continue
        seen.add(n)
        deduped.append(n)

    if not deduped:
        return str(ordering), None, "no agents queried"

    traj = build_estimator_trajectory(deduped, data_cache, key_cols, true_count)

    def first(k_fn, thr):
        """Run a k-extraction helper, tolerating errors."""
        try:
            return k_fn(traj, thr)
        except Exception:  # noqa: BLE001
            return None

    logging.info(
        "  %s/%s flat_k(10%%)=%s err_k(10%%)=%s agents=%d",
        table,
        ordering,
        first(flatness_k, 0.10),
        first(error_k, 0.10),
        len(deduped),
    )
    if output_dir is not None:
        _save_trajectory(output_dir, table, str(ordering), traj)
    return str(ordering), traj, None


def _save_trajectory(output_dir: Path, table: str, ordering: str, traj: EstimatorTrajectory):
    """Persist one ordering's estimator trajectory to CSV, eagerly."""
    out = output_dir / "trajectories"
    out.mkdir(parents=True, exist_ok=True)
    frame = traj.to_frame()
    frame.insert(0, "ordering", ordering)
    frame.to_csv(out / f"{table}__{ordering}.csv", index=False)


def _run_ordering_worker(
    ordering: SelectionStrategyType,
    step: MultiAgentQueryPlanStep,
    step_meta: dict[str, Any],
    canonical_sql: str,
    canonical_table: str,
    key_cols: list[str],
    all_agents: list[DataAgent],
    data_cache: dict[str, pd.DataFrame | None],
    true_count: int,
    meta_mapping: MetaMapping,
    data_dir: str,
    config: Config,
    cache_dir: str | None,
    output_dir: str,
    table: str,
) -> tuple[str, EstimatorTrajectory | None, str | None]:
    """Run one ordering in a fresh worker process (own hooks + context)."""
    from scripts.utils.canonical import query_agent_canonical  # noqa: F401

    _per_run_align.clear()
    _per_run_data.clear()
    align_dispatch = make_align_schema_dispatch(_per_run_align)
    data_dispatch = make_data_agent_dispatch(_per_run_data)
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
    align_handler = make_align_schema_override(step_meta, table)
    data_handler = make_data_agent_override(canonical_sql, canonical_table, meta_mapping, data_cache, data_dir_path)

    context.configure_logging(Path(output_dir))

    with (
        context.config_context(config),
        context.data_context(data_dir_path),
        context.cache_context(Path(cache_dir)) if cache_dir is not None else contextlib.nullcontext(),
    ):
        try:
            return asyncio.run(
                _run_one_ordering(
                    ordering=ordering,
                    step=step,
                    align_handler=align_handler,
                    data_handler=data_handler,
                    all_agents=all_agents,
                    data_cache=data_cache,
                    key_cols=key_cols,
                    true_count=true_count,
                    config=config,
                    table=table,
                    output_dir=Path(output_dir),
                )
            )
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


def _default_workers() -> int:
    """Default number of parallel ordering worker processes per table."""
    return max(1, min(len(STRATEGIES), 3))


async def _run_orderings(
    *,
    step: MultiAgentQueryPlanStep,
    step_meta: dict[str, Any],
    canonical_sql: str,
    canonical_table: str,
    key_cols: list[str],
    relevant_agents: list[DataAgent],
    all_agents: list[DataAgent],
    data_cache: dict[str, pd.DataFrame | None],
    true_count: int,
    meta_mapping: MetaMapping,
    data_dir: Path,
    config: Config,
    cache_dir: Path | None,
    output_dir: Path,
    table: str,
    workers: int,
) -> dict[str, tuple[EstimatorTrajectory | None, str | None]]:
    """Run all orderings, in parallel worker processes when enabled."""
    if workers <= 1:
        align_handler = make_align_schema_override(step_meta, table)
        data_handler = make_data_agent_override(canonical_sql, canonical_table, meta_mapping, data_cache, data_dir)
        results = await asyncio.gather(
            *[
                _run_one_ordering(
                    ordering=s,
                    step=step,
                    align_handler=align_handler,
                    data_handler=data_handler,
                    all_agents=all_agents,
                    data_cache=data_cache,
                    key_cols=key_cols,
                    true_count=true_count,
                    config=config,
                    table=table,
                    output_dir=output_dir,
                )
                for s in STRATEGIES
            ]
        )
    else:
        loop = asyncio.get_running_loop()
        with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1) as pool:
            results = list(
                await asyncio.gather(
                    *[
                        loop.run_in_executor(
                            pool,
                            _run_ordering_worker,
                            s,
                            step,
                            step_meta,
                            canonical_sql,
                            canonical_table,
                            key_cols,
                            all_agents,
                            data_cache,
                            true_count,
                            meta_mapping,
                            str(data_dir),
                            config,
                            str(cache_dir) if cache_dir is not None else None,
                            str(output_dir),
                            table,
                        )
                        for s in STRATEGIES
                    ]
                )
            )

    return {ordering: (traj, err) for ordering, traj, err in results}


async def _run_table(
    table: str,
    step: MultiAgentQueryPlanStep,
    step_meta: dict[str, Any],
    relevant_agents: list[DataAgent],
    all_agents: list[DataAgent],
    meta_mapping: MetaMapping,
    output_dir: Path,
    data_dir: Path,
    config: Config,
    cache_dir: Path | None,
    workers: int | None,
) -> TableSummary | None:
    """Run all orderings for one table and build its summary."""
    canonical_sql = select_all_sql(table)
    canonical_table = parse_canonical_table(canonical_sql)
    if canonical_table is None:
        return None
    key_cols = canonical_key_cols(meta_mapping, table)
    if not key_cols:
        logging.warning("%s: no key columns — skipping", table)
        return None

    data_cache = load_agent_cache(relevant_agents, canonical_sql, canonical_table, meta_mapping, data_dir)
    true_count = true_count_from_cache(data_cache, key_cols)
    logging.info("Table %s: true_count=%d agents=%d", table, true_count, len(relevant_agents))

    runs = await _run_orderings(
        step=step,
        step_meta=step_meta,
        canonical_sql=canonical_sql,
        canonical_table=canonical_table,
        key_cols=key_cols,
        relevant_agents=relevant_agents,
        all_agents=all_agents,
        data_cache=data_cache,
        true_count=true_count,
        meta_mapping=meta_mapping,
        data_dir=data_dir,
        config=config,
        cache_dir=cache_dir,
        output_dir=output_dir,
        table=table,
        workers=workers or _default_workers(),
    )

    trajectories = {o: t for o, (t, _) in runs.items() if t is not None}
    failures = {o: e for o, (_, e) in runs.items() if e is not None}
    return TableSummary(
        table=table,
        agents_total=len(relevant_agents),
        true_count=true_count,
        trajectories=trajectories,
        failures=failures,
    )


async def run_query(
    *,
    input_file: Path,
    config: Config,
    dataset_dir: Path,
    output_dir: Path,
    cache_dir: Path | None,
    data: CollaborativeQueryAnswering,
    extra: dict[str, Any] | None,
):
    """Run the k-warmup study for one input (single-step SELECT * table) query."""
    extra = extra or {}
    raw_query_plan = extra.get("query_plan")
    if raw_query_plan is None:
        logging.warning("%s: no query_plan — skipping", input_file.stem)
        return
    query_plan = QueryPlan.model_validate(raw_query_plan)
    multi_steps = [s for s in query_plan.steps if isinstance(s, MultiAgentQueryPlanStep)]
    if not multi_steps:
        return

    meta_mapping: MetaMapping = cast(MetaMapping, load_meta_mapping(dataset_dir))
    all_agents = data.data_agents
    relevant_names: set[str] = set(extra.get("relevant_data_agents", []))
    relevant_agents = [a for a in all_agents if a.name in relevant_names]
    table = parse_canonical_table(extra.get("sql_query") or "") or multi_steps[0].id
    query_slug = input_file.stem
    run_output_dir = output_dir / query_slug
    data_dir = dataset_dir.parent

    align_dispatch = make_align_schema_dispatch(_per_run_align)
    data_dispatch = make_data_agent_dispatch(_per_run_data)
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

    workers: int | None = extra.get("k_warmup_workers")

    try:
        summaries = await asyncio.gather(
            *[
                _run_table(
                    table=table,
                    step=step,
                    step_meta=extra.get("steps_metadata", {}).get(step.id, {}),
                    relevant_agents=relevant_agents,
                    all_agents=all_agents,
                    meta_mapping=meta_mapping,
                    output_dir=run_output_dir,
                    data_dir=data_dir,
                    config=config,
                    cache_dir=cache_dir,
                    workers=workers,
                )
                for step in multi_steps
            ]
        )
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

    from scripts.k_warmup.summary import write_table_summary

    write_table_summary([s for s in summaries if s is not None], run_output_dir)

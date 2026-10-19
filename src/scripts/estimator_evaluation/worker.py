"""Estimator evaluation worker.

Runs the full execution pipeline (including the real completeness evaluator:
deterministic estimator stop + LLM world-knowledge stop + result producer) on
each multi-agent step and records where it stops vs ground truth.
"""

import asyncio
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
import requests

import qcp_adaptive_termination.hooks as hooks
from qcp_adaptive_termination.hooks import HookEvent
from qcp_adaptive_termination.models import (
    AlignedResultList,
    Config,
    DataAgent,
)
from qcp_adaptive_termination.models.execution import (
    FailedExecution,
    PartialQueryAnswer,
)
from qcp_adaptive_termination.models.query_plan import (
    MultiAgentQueryPlanStep,
    QueryPlan,
)
from qcp_adaptive_termination.models.stages import CollaborativeQueryAnswering
from qcp_adaptive_termination.observability import context
from qcp_adaptive_termination.stages import execute_query_plan
from scripts.estimator_evaluation.ground_truth import compute_ground_truth
from scripts.estimator_evaluation.models import BatchRecord, StepResult
from scripts.estimator_evaluation.results import save_step_results
from scripts.utils.canonical import (
    load_meta_mapping,
    make_align_schema_override,
    make_data_agent_override,
    parse_canonical_table,
    query_agent_canonical,
)
from scripts.utils.overrides import (
    make_align_schema_dispatch,
    make_data_agent_dispatch,
)
from scripts.utils.sql import parse_select_columns

# Per-step dispatch maps: step execution_id -> step-specific handler.
_per_step_align: dict[str, Callable[..., AlignedResultList]] = {}
_per_step_data: dict[str, Callable[..., PartialQueryAnswer]] = {}


def _replan_error(
    execution_id: str,
    data_agents: list[DataAgent],
    relevant_data_agents: list[DataAgent],
    query_plan: QueryPlan,
    reason: FailedExecution,
) -> QueryPlan:
    del execution_id, data_agents, relevant_data_agents, query_plan
    raise RuntimeError(f"Unexpected replan triggered in evaluator benchmark. Step: {reason.step.id!r}")


def _register_tracking(
    target_execution_id: str,
    data_cache: dict[str, pd.DataFrame | None],
    total_unique: int,
    key_cols: list[str],
) -> tuple[list[str], list[BatchRecord], Any]:
    ordered: list[str] = []
    batches: list[BatchRecord] = []
    seen_keys: set[tuple] = set()

    def _on_agent_query(
        execution_id: str,
        data_agent: DataAgent,
        query: str,
        result: PartialQueryAnswer,
    ) -> None:
        if not execution_id.startswith(target_execution_id + "/"):
            return

        name = data_agent.name
        ordered.append(name)
        df = data_cache.get(name)
        if df is not None and not df.empty:
            cols = [c for c in key_cols if c in df.columns]
            if cols:
                seen_keys.update(map(tuple, df[cols].drop_duplicates().values.tolist()))

        coverage = min(len(seen_keys) / total_unique, 1.0) if total_unique > 0 else 0.0
        batches.append(
            BatchRecord(
                batch_num=len(batches) + 1,
                cumulative_agents=len(ordered),
                decision="query_more_agents",
                sample_coverage=None,
                observed_count=len(seen_keys),
                estimated_count=None,
                observed_value_sum=None,
                estimated_value_sum=None,
                observed_value_avg=None,
                estimated_value_avg=None,
                coverage_achieved=coverage,
            )
        )

    hooks.register_post(HookEvent.EXECUTION_QUERY_DATA_AGENT, _on_agent_query)
    return ordered, batches, _on_agent_query


async def _run_step(
    step: MultiAgentQueryPlanStep,
    step_meta: dict[str, Any],
    relevant_data_agents: list[DataAgent],
    all_agents: list[DataAgent],
    raw_query_plan: Any,
    meta_mapping: dict[str, Any],
    query: str,
    query_slug: str,
    output_dir: Path,
    data_dir: Path,
    config: Config,
) -> StepResult | None:
    canonical_sql: str = step_meta["sql_query"]
    schema_meta = step_meta["aligned_schema_metadata"]
    key_cols: list[str] = schema_meta["keys"]
    value_column: str | None = schema_meta.get("value_column")

    canonical_table = parse_canonical_table(canonical_sql)
    if canonical_table is None:
        logging.warning("%s/%s: could not parse table — skipping", query_slug, step.id)
        return None
    canonical_cols = parse_select_columns(canonical_sql)
    if not canonical_cols:
        logging.warning("%s/%s: could not parse columns — skipping", query_slug, step.id)
        return None

    step_agent_names = set(step.to_agent)
    step_agents = [a for a in relevant_data_agents if a.name in step_agent_names]

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

    ground_truth = compute_ground_truth(data_cache, key_cols, value_column)
    logging.info(
        "%s/%s: true_count=%d true_sum=%s agents=%d",
        query_slug,
        step.id,
        ground_truth.count,
        f"{ground_truth.value_sum:.2f}" if ground_truth.value_sum is not None else "n/a",
        len(step_agents),
    )

    align_handler = make_align_schema_override(step_meta, query)
    data_handler = make_data_agent_override(canonical_sql, canonical_table, meta_mapping, data_cache, data_dir)

    execution_id = uuid4().hex[:12]
    ordered, batches, batch_end_handler = _register_tracking(execution_id, data_cache, ground_truth.count, key_cols)
    fresh_query_plan = QueryPlan.model_validate(raw_query_plan)
    _per_step_align[execution_id] = align_handler
    _per_step_data[execution_id] = data_handler

    try:
        with (
            hooks.execution_context(execution_id),
            context.config_context(config.model_copy(update={"continue_after_final_result": False})),
        ):
            await execute_query_plan(
                query_plan=fresh_query_plan,
                relevant_data_agents=relevant_data_agents,
                data_agents=all_agents,
            )
    except Exception as exc:
        logging.warning("Step %s/%s failed: %s", query_slug, step.id, exc)
        logging.exception(exc)
        if isinstance(exc, requests.exceptions.Timeout):
            raise SystemExit(1) from exc
    finally:
        _per_step_align.pop(execution_id, None)
        _per_step_data.pop(execution_id, None)
        hooks.unregister_post(HookEvent.EXECUTION_QUERY_DATA_AGENT, batch_end_handler)

    agents_total = len(step.to_agent)

    stop_batch = batches[-1] if batches else None

    if stop_batch is None:
        logging.warning("Step %s/%s: no batch records", query_slug, step.id)
        return None

    agents_queried_at_stop = stop_batch.cumulative_agents
    stop_fraction = agents_queried_at_stop / agents_total if agents_total > 0 else 0.0
    early_stop = agents_queried_at_stop < agents_total

    logging.info(
        "%s/%s: stop=%d/%d (%.0f%%) coverage=%.3f count_est=%s true=%d early=%s",
        query_slug,
        step.id,
        agents_queried_at_stop,
        agents_total,
        stop_fraction * 100,
        stop_batch.coverage_achieved,
        f"{stop_batch.estimated_count:.1f}" if stop_batch.estimated_count is not None else "n/a",
        ground_truth.count,
        early_stop,
    )

    result = StepResult(
        step_id=step.id,
        agents_total=agents_total,
        agents_queried_at_stop=agents_queried_at_stop,
        stop_fraction=stop_fraction,
        early_stop=early_stop,
        ground_truth=ground_truth,
        batches=batches,
    )
    save_step_results(result, query_slug, output_dir)
    return result


async def run_query(
    *,
    input_file: Path,
    config: Config,
    data_dir: Path,
    output_dir: Path,
    cache_dir: Path | None,
    data: CollaborativeQueryAnswering,
    extra: dict[str, Any] | None,
) -> None:
    """Evaluate the completeness evaluator for one input query (worker)."""
    extra = extra or {}
    steps_metadata: dict[str, Any] = extra.get("steps_metadata", {})

    raw_query_plan = extra.get("query_plan")
    if raw_query_plan is None:
        logging.warning("%s: no query_plan in extra — skipping", input_file.stem)
        return

    query_plan = QueryPlan.model_validate(raw_query_plan)
    multi_steps = [s for s in query_plan.steps if isinstance(s, MultiAgentQueryPlanStep)]

    if not multi_steps:
        logging.warning("%s: no multi-agent steps — skipping", input_file.stem)
        return

    meta_mapping = load_meta_mapping(data_dir)
    all_agents = data.data_agents
    relevant_agent_names: set[str] = set(extra.get("relevant_data_agents", []))
    relevant_data_agents = [a for a in all_agents if a.name in relevant_agent_names]
    query = query_plan.query
    query_slug = input_file.stem
    run_output_dir = output_dir / query_slug

    hooks.clear(HookEvent.EXECUTION_EXECUTE_QUERY_PLAN)
    align_dispatch = make_align_schema_dispatch(_per_step_align, prefix_match=True)
    data_dispatch = make_data_agent_dispatch(_per_step_data, prefix_match=True)
    hooks.register_overwrite(
        HookEvent.EXECUTION_EXECUTE_QUERY_PLAN_EXECUTE_STEP_ANSWER_STEP_MULTIPLE_AGENTS_ALIGN_SCHEMA,
        align_dispatch,
    )
    hooks.register_overwrite(HookEvent.EXECUTION_QUERY_DATA_AGENT, data_dispatch)
    hooks.register_overwrite(HookEvent.EXECUTION_REPLAN_DRAFT_QUERY_PLAN, _replan_error)

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
                raw_query_plan=raw_query_plan,
                meta_mapping=meta_mapping,
                query=query,
                query_slug=query_slug,
                output_dir=run_output_dir,
                data_dir=data_dir,
                config=config,
            )
        )

    try:
        await asyncio.gather(*step_tasks)
    finally:
        hooks.unregister_overwrite(
            HookEvent.EXECUTION_EXECUTE_QUERY_PLAN_EXECUTE_STEP_ANSWER_STEP_MULTIPLE_AGENTS_ALIGN_SCHEMA,
            align_dispatch,
        )
        hooks.unregister_overwrite(HookEvent.EXECUTION_QUERY_DATA_AGENT, data_dispatch)
        hooks.unregister_overwrite(HookEvent.EXECUTION_REPLAN_DRAFT_QUERY_PLAN, _replan_error)

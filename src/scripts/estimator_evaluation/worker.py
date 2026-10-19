"""Estimator evaluation worker.

Runs the full execution pipeline (including the real completeness evaluator:
deterministic estimator stop + LLM world-knowledge stop + result producer) once
per query and records, for each multi-agent step, where it stopped vs ground
truth.

The query plan is executed exactly once per input query. Post-hooks on the
data-agent query and completeness events capture, per step, the agent query
order, the estimator state, and the stop decision. Per-step results are then
built from that captured trajectory.
"""

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
import requests

import qcp_adaptive_termination.hooks as hooks
from qcp_adaptive_termination.estimator import Estimator
from qcp_adaptive_termination.hooks import HookEvent
from qcp_adaptive_termination.models import (
    AlignedResultList,
    AlignedSchemaMetadata,
    Config,
    DataAgent,
    StepBudget,
    StopReason,
)
from qcp_adaptive_termination.models.execution import (
    AdditionalInformation,
    FailedExecution,
    PartialQueryAnswer,
    StopDecision,
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


def _make_align_dispatch(
    handlers: dict[str, Callable[..., AlignedResultList]],
) -> Callable[..., AlignedResultList]:
    """Dispatch ALIGN_SCHEMA by agent name (each step has a distinct agent set)."""

    def _dispatch(
        execution_id: str,
        data_agents: list[DataAgent],
        agent_names: list[str],
        query: str,
        merge_task: str,
    ) -> AlignedResultList:
        for name in agent_names:
            handler = handlers.get(name)
            if handler is not None:
                return handler(
                    execution_id,
                    data_agents=data_agents,
                    agent_names=agent_names,
                    query=query,
                    merge_task=merge_task,
                )
        raise ValueError(f"No align_schema handler for agents={agent_names!r}")

    return _dispatch


def _make_data_dispatch(
    handlers: dict[str, Callable[..., PartialQueryAnswer]],
) -> Callable[..., PartialQueryAnswer]:
    """Dispatch DATA_AGENT by agent name (each step has a distinct agent set)."""

    def _dispatch(
        execution_id: str,
        data_agent: DataAgent,
        query: str,
        additional_information: list[AdditionalInformation],
        schema_metadata: AlignedSchemaMetadata | None,
    ) -> PartialQueryAnswer:
        handler = handlers.get(data_agent.name)
        if handler is None:
            raise ValueError(f"No data_agent handler for agent={data_agent.name!r}")
        return handler(
            execution_id,
            data_agent=data_agent,
            query=query,
            additional_information=additional_information,
            schema_metadata=schema_metadata,
        )

    return _dispatch


def _estimator_snapshot(estimator: Estimator | None) -> dict[str, Any]:
    """Extract the estimator's observable state for one completeness check."""
    if estimator is None:
        return {
            "sample_coverage": None,
            "observed_count": None,
            "estimated_count": None,
            "observed_value_sum": None,
            "estimated_value_sum": None,
            "observed_value_avg": None,
            "estimated_value_avg": None,
        }
    return {
        "sample_coverage": estimator.sample_coverage(),
        "observed_count": estimator.sample_count(),
        "estimated_count": estimator.estimated_count(),
        "observed_value_sum": estimator.sample_sum(),
        "estimated_value_sum": estimator.estimated_sum(),
        "observed_value_avg": estimator.sample_avg(),
        "estimated_value_avg": estimator.estimated_avg(),
    }


class _StepTracker:
    """Collects the per-step completeness trajectory during one plan execution."""

    def __init__(self) -> None:
        # agent name -> step id
        self._agent_to_step: dict[str, str] = {}
        # step_id -> ordered agent names
        self._ordered: dict[str, list[str]] = {}
        # step_id -> list of BatchRecord
        self._batches: dict[str, list[BatchRecord]] = {}
        self._seen_keys: dict[str, set[tuple]] = {}
        self._total_unique: dict[str, int] = {}
        self._key_cols: dict[str, list[str]] = {}

    def register_step(self, step_id: str, agent_names: list[str], key_cols: list[str], total_unique: int) -> None:
        """Declare a step's agents, key columns and ground-truth unique count."""
        for name in agent_names:
            self._agent_to_step[name] = step_id
        self._key_cols[step_id] = key_cols
        self._total_unique[step_id] = total_unique

    def _coverage(self, step_id: str) -> float:
        seen = self._seen_keys.get(step_id, set())
        total = self._total_unique.get(step_id, 0)
        return min(len(seen) / total, 1.0) if total > 0 else 0.0

    def on_agent_query(
        self,
        execution_id: str,
        data_agent: DataAgent,
        query: str,
        result: PartialQueryAnswer,
    ) -> None:
        """Record one agent query for the step it belongs to (via agent name)."""
        step_id = self._agent_to_step.get(data_agent.name)
        if step_id is None:
            return
        ordered = self._ordered.setdefault(step_id, [])
        ordered.append(data_agent.name)

        df = result.answer
        if df is not None and not df.empty:
            cols = [c for c in self._key_cols.get(step_id, []) if c in df.columns]
            if cols:
                self._seen_keys.setdefault(step_id, set()).update(
                    map(tuple, df[cols].drop_duplicates().values.tolist())
                )

        coverage = self._coverage(step_id)
        self._batches.setdefault(step_id, []).append(
            BatchRecord(
                batch_num=len(self._batches[step_id]) + 1,
                cumulative_agents=len(ordered),
                decision="query_more_agents",
                sample_coverage=None,
                observed_count=len(self._seen_keys.get(step_id, set())),
                estimated_count=None,
                observed_value_sum=None,
                estimated_value_sum=None,
                observed_value_avg=None,
                estimated_value_avg=None,
                coverage_achieved=coverage,
            )
        )

    def on_completeness(
        self,
        execution_id: str,
        estimator: Estimator | None,
        step_budget: StepBudget | None,
        agents_queried: int,
        force_result: bool,
        result: StopDecision | None,
    ) -> None:
        """Record the estimator state + stop decision at each completeness check."""
        # The completeness hook fires inside the step's execution context; the
        # step budget carries the step id.
        step_id = step_budget.step_id if step_budget is not None else None
        if step_id is None:
            return
        snapshot = _estimator_snapshot(estimator)

        if result is not None:
            decision = result.stop_reason.value
        elif force_result:
            decision = StopReason.ALL_AGENTS.value
        else:
            decision = "query_more_agents"

        batches = self._batches.setdefault(step_id, [])
        if not batches:
            return
        last = batches[-1]
        last.decision = decision
        last.sample_coverage = snapshot["sample_coverage"]
        last.observed_count = snapshot["observed_count"]
        last.estimated_count = snapshot["estimated_count"]
        last.observed_value_sum = snapshot["observed_value_sum"]
        last.estimated_value_sum = snapshot["estimated_value_sum"]
        last.observed_value_avg = snapshot["observed_value_avg"]
        last.estimated_value_avg = snapshot["estimated_value_avg"]

    def batches_for(self, step_id: str) -> list[BatchRecord]:
        """Return the trajectory for a step."""
        return self._batches.get(step_id, [])


def _build_step_result(
    step: MultiAgentQueryPlanStep,
    step_meta: dict[str, Any],
    relevant_data_agents: list[DataAgent],
    meta_mapping: dict[str, Any],
    data_dir: Path,
    query_slug: str,
    output_dir: Path,
    tracker: _StepTracker,
) -> StepResult | None:
    """Build one step's result from the captured trajectory + ground truth."""
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

    batches = tracker.batches_for(step.id)
    if not batches:
        logging.warning("%s/%s: no batch records", query_slug, step.id)
        return None

    agents_total = len(step.to_agent)
    stop_batch = batches[-1]
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
    dataset_dir: Path,
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

    data_dir = dataset_dir.parent
    meta_mapping = load_meta_mapping(dataset_dir)
    all_agents = data.data_agents
    relevant_agent_names: set[str] = set(extra.get("relevant_data_agents", []))
    relevant_data_agents = [a for a in all_agents if a.name in relevant_agent_names]
    query = query_plan.query
    query_slug = input_file.stem
    run_output_dir = output_dir / query_slug

    # Pre-build per-step data caches + ground truth (needed by the tracker and
    # the per-step result builder).
    step_infos: list[tuple[MultiAgentQueryPlanStep, dict[str, Any], dict[str, pd.DataFrame | None], str]] = []
    for step in multi_steps:
        step_meta = steps_metadata.get(step.id)
        if step_meta is None:
            logging.warning("%s/%s: no steps_metadata — skipping", input_file.stem, step.id)
            continue
        canonical_sql: str = step_meta["sql_query"]
        schema_meta = step_meta["aligned_schema_metadata"]
        canonical_table = parse_canonical_table(canonical_sql)
        if canonical_table is None:
            continue
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
        step_infos.append((step, step_meta, data_cache, canonical_table))

    # Register per-step dispatch handlers keyed by agent name (each step has a
    # distinct agent set, so agent name uniquely identifies the step).
    execution_id = uuid4().hex[:12]
    tracker = _StepTracker()
    for step, step_meta, data_cache, canonical_table in step_infos:
        align_handler = make_align_schema_override(step_meta, query)
        data_handler = make_data_agent_override(
            step_meta["sql_query"], canonical_table, meta_mapping, data_cache, data_dir
        )
        for agent_name in step.to_agent:
            _per_step_align[agent_name] = align_handler
            _per_step_data[agent_name] = data_handler
        schema_meta = step_meta["aligned_schema_metadata"]
        key_cols: list[str] = schema_meta["keys"]
        value_column: str | None = schema_meta.get("value_column")
        gt = compute_ground_truth(data_cache, key_cols, value_column)
        tracker.register_step(step.id, step.to_agent, key_cols, gt.count)

    hooks.clear(HookEvent.EXECUTION_EXECUTE_QUERY_PLAN)
    align_dispatch = _make_align_dispatch(_per_step_align)
    data_dispatch = _make_data_dispatch(_per_step_data)
    hooks.register_overwrite(
        HookEvent.EXECUTION_EXECUTE_QUERY_PLAN_EXECUTE_STEP_ANSWER_STEP_MULTIPLE_AGENTS_ALIGN_SCHEMA,
        align_dispatch,
    )
    hooks.register_overwrite(HookEvent.EXECUTION_QUERY_DATA_AGENT, data_dispatch)
    hooks.register_overwrite(HookEvent.EXECUTION_REPLAN_DRAFT_QUERY_PLAN, _replan_error)
    hooks.register_post(HookEvent.EXECUTION_QUERY_DATA_AGENT, tracker.on_agent_query)
    hooks.register_post(
        HookEvent.EXECUTION_EXECUTE_QUERY_PLAN_EXECUTE_STEP_ANSWER_STEP_MULTIPLE_AGENTS_COMPLETENESS,
        tracker.on_completeness,
    )

    try:
        with (
            hooks.execution_context(execution_id),
            context.config_context(config.model_copy(update={"continue_after_final_result": False})),
        ):
            await execute_query_plan(
                query_plan=QueryPlan.model_validate(raw_query_plan),
                relevant_data_agents=relevant_data_agents,
                data_agents=all_agents,
            )
    except Exception as exc:
        logging.warning("%s: query failed: %s", query_slug, exc)
        logging.exception(exc)
        if isinstance(exc, requests.exceptions.Timeout):
            raise SystemExit(1) from exc
    finally:
        hooks.unregister_overwrite(
            HookEvent.EXECUTION_EXECUTE_QUERY_PLAN_EXECUTE_STEP_ANSWER_STEP_MULTIPLE_AGENTS_ALIGN_SCHEMA,
            align_dispatch,
        )
        hooks.unregister_overwrite(HookEvent.EXECUTION_QUERY_DATA_AGENT, data_dispatch)
        hooks.unregister_overwrite(HookEvent.EXECUTION_REPLAN_DRAFT_QUERY_PLAN, _replan_error)
        hooks.unregister_post(HookEvent.EXECUTION_QUERY_DATA_AGENT, tracker.on_agent_query)
        hooks.unregister_post(
            HookEvent.EXECUTION_EXECUTE_QUERY_PLAN_EXECUTE_STEP_ANSWER_STEP_MULTIPLE_AGENTS_COMPLETENESS,
            tracker.on_completeness,
        )
        for step, _, _, _ in step_infos:
            for agent_name in step.to_agent:
                _per_step_align.pop(agent_name, None)
                _per_step_data.pop(agent_name, None)

    # Build per-step results from the captured trajectory.
    for step, step_meta, _, _ in step_infos:
        _build_step_result(
            step=step,
            step_meta=step_meta,
            relevant_data_agents=relevant_data_agents,
            meta_mapping=meta_mapping,
            data_dir=data_dir,
            query_slug=query_slug,
            output_dir=run_output_dir,
            tracker=tracker,
        )

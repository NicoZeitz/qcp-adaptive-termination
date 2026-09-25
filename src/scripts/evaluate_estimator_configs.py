"""Evaluate completeness evaluator with fixed config across query plan steps."""

import asyncio
import hashlib
import logging
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4

import matplotlib.pyplot as plt
import pandas as pd
import requests
import yaml
from sqlalchemy import create_engine, text

import qcp_adaptive_termination.hooks as hooks
from qcp_adaptive_termination.hooks import HookEvent
from qcp_adaptive_termination.models import (
    AggregationType,
    AlignedResultList,
    AlignedSchemaMetadata,
    Config,
    DataAgent,
    EstimatorStatistics,
    ScoredDataAgent,
)
from qcp_adaptive_termination.models.execution import (
    AdditionalInformation,
    EvaluatorDecision,
    FailedExecution,
    PartialQueryAnswer,
)
from qcp_adaptive_termination.models.query_plan import MultiAgentQueryPlanStep, QueryPlan
from qcp_adaptive_termination.models.stages import CollaborativeQueryAnswering
from qcp_adaptive_termination.observability import context
from qcp_adaptive_termination.stages import execute_query_plan
from scripts.utils import ScriptContext, run_script
from scripts.utils.sql import parse_select_columns, rewrite_sql_identifiers

_META_MAPPING_CACHE: dict[str, Any] | None = None

_per_step_align: dict[str, Callable[..., AlignedResultList]] = {}
_per_step_data: dict[str, Callable[..., PartialQueryAnswer]] = {}


@dataclass(slots=True)
class GroundTruth:
    count: int
    value_sum: float | None
    value_avg: float | None
    value_min: float | None
    value_max: float | None


@dataclass(slots=True)
class BatchRecord:
    batch_num: int
    cumulative_agents: int
    decision: str
    sample_coverage: float | None
    observed_count: int | None
    estimated_count: float | None
    observed_value_sum: float | None
    estimated_value_sum: float | None
    observed_value_avg: float | None
    estimated_value_avg: float | None
    coverage_achieved: float


@dataclass(slots=True)
class StepResult:
    step_id: str
    agents_total: int
    agents_queried_at_stop: int
    stop_fraction: float
    early_stop: bool
    ground_truth: GroundTruth
    stop_stats: EstimatorStatistics | None
    batches: list[BatchRecord]


def _load_meta_mapping(data_dir: Path) -> dict[str, Any]:
    global _META_MAPPING_CACHE
    if _META_MAPPING_CACHE is None:
        mapping_path = data_dir / "dataset" / "meta_mapping.yaml"
        with open(mapping_path) as f:
            _META_MAPPING_CACHE = cast(dict[str, Any], yaml.safe_load(f) or {})
    return _META_MAPPING_CACHE


def _slug_from_agent(agent: DataAgent) -> str:
    return Path(agent.path).stem


def _parse_canonical_table(sql: str) -> str | None:
    match = re.search(r"\bFROM\s+(\w+)", sql, re.IGNORECASE)
    return match.group(1) if match else None


def _query_agent_canonical(
    agent: DataAgent,
    canonical_sql: str,
    canonical_table: str,
    meta_mapping: dict[str, Any],
    data_dir: Path,
) -> pd.DataFrame | None:
    slug = _slug_from_agent(agent)
    table_meta = meta_mapping.get(canonical_table, {}).get(slug)
    if table_meta is None:
        return None

    col_map: dict[str, str] = table_meta.get("columns", {})
    try:
        actual_sql = rewrite_sql_identifiers(canonical_sql, canonical_table, table_meta["name"], col_map)
    except Exception as exc:
        logging.warning("Failed to rewrite SQL for %s (%s): %s", agent.name, slug, exc)
        return None

    db_path = data_dir / agent.path
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(actual_sql), conn)
        if df.empty:
            return None
        reverse_map = {v: k for k, v in col_map.items()}
        return df.rename(columns=reverse_map)
    except Exception as exc:
        logging.warning("Failed to query %s (%s): %s", agent.name, slug, exc)
        return None
    finally:
        engine.dispose()


def _compute_ground_truth(
    data_cache: dict[str, pd.DataFrame | None],
    key_cols: list[str],
    value_column: str | None,
) -> GroundTruth:
    frames = [df for df in data_cache.values() if df is not None and not df.empty]
    if not frames:
        return GroundTruth(count=0, value_sum=None, value_avg=None, value_min=None, value_max=None)

    combined = pd.concat(frames, ignore_index=True)
    dedup_cols = [c for c in key_cols if c in combined.columns] or list(combined.columns)
    deduped = combined.drop_duplicates(subset=dedup_cols)

    value_sum = value_avg = value_min = value_max = None
    if value_column is not None and value_column in deduped.columns:
        col = deduped[value_column].dropna()
        if not col.empty:
            value_sum = float(cast(Any, col.sum()))
            value_avg = float(cast(Any, col.mean()))
            value_min = float(cast(Any, col.min()))
            value_max = float(cast(Any, col.max()))

    return GroundTruth(
        count=len(deduped),
        value_sum=value_sum,
        value_avg=value_avg,
        value_min=value_min,
        value_max=value_max,
    )


def _make_align_schema_override(step_meta: dict[str, Any], query: str) -> Callable[..., AlignedResultList]:
    def _handler(
        execution_id: str,
        data_agents: list[DataAgent],
        agent_names: list[str],
        query: str,
        merge_task: str,
    ) -> AlignedResultList:
        meta = step_meta["aligned_schema_metadata"]
        source_col = f"source_agent_{hashlib.sha256(query.encode()).hexdigest()[:8]}"
        agg_types = [AggregationType(a) for a in meta["aggregation_types"]]
        schema = AlignedSchemaMetadata(
            aggregation_types=agg_types,
            keys=meta["keys"],
            value_column=meta.get("value_column"),
            columns=meta["columns"],
            source_agent_column=source_col,
        )
        return AlignedResultList(
            data=pd.DataFrame(columns=meta["columns"] + [source_col]),
            schema_metadata=schema,
        )

    return _handler


def _make_data_agent_override(
    canonical_sql: str,
    canonical_table: str,
    meta_mapping: dict[str, Any],
    data_cache: dict[str, pd.DataFrame | None],
    data_dir: Path,
) -> Callable[..., PartialQueryAnswer]:
    def _handler(
        execution_id: str,
        data_agent: DataAgent,
        query: str,
        additional_information: list[AdditionalInformation],
        schema_metadata: AlignedSchemaMetadata | None,
    ) -> PartialQueryAnswer:
        if data_agent.name not in data_cache:
            data_cache[data_agent.name] = _query_agent_canonical(
                data_agent, canonical_sql, canonical_table, meta_mapping, data_dir
            )
        df = data_cache[data_agent.name]
        partial = PartialQueryAnswer()
        partial.set_answer(df if df is not None and not df.empty else pd.DataFrame())
        return partial

    return _handler


def _replan_error(
    execution_id: str,
    data_agents: list[DataAgent],
    relevant_data_agents: list[DataAgent],
    query_plan: QueryPlan,
    reason: FailedExecution,
) -> QueryPlan:
    del execution_id, data_agents, relevant_data_agents, query_plan
    raise RuntimeError(f"Unexpected replan triggered in evaluator benchmark. Step: {reason.step.id!r}")


def _align_schema_dispatch(
    execution_id: str,
    data_agents: list[DataAgent],
    agent_names: list[str],
    query: str,
    merge_task: str,
) -> AlignedResultList:
    for step_id, handler in _per_step_align.items():
        if execution_id == step_id or execution_id.startswith(step_id + "/"):
            return handler(
                execution_id, data_agents=data_agents, agent_names=agent_names, query=query, merge_task=merge_task
            )
    raise ValueError(f"No align_schema handler for execution_id={execution_id!r}")


def _data_agent_dispatch(
    execution_id: str,
    data_agent: DataAgent,
    query: str,
    additional_information: list[AdditionalInformation],
    schema_metadata: AlignedSchemaMetadata | None,
) -> PartialQueryAnswer:
    for step_id, handler in _per_step_data.items():
        if execution_id == step_id or execution_id.startswith(step_id + "/"):
            return handler(
                execution_id,
                data_agent=data_agent,
                query=query,
                additional_information=additional_information,
                schema_metadata=schema_metadata,
            )
    raise ValueError(f"No data_agent handler for execution_id={execution_id!r}")


def _register_tracking(
    target_execution_id: str,
    target_step_id: str,
    data_cache: dict[str, pd.DataFrame | None],
    total_unique: int,
    key_cols: list[str],
) -> tuple[list[str], list[BatchRecord], list[EstimatorStatistics], Any]:
    ordered: list[str] = []
    batches: list[BatchRecord] = []
    estimator_snapshots: list[EstimatorStatistics] = []
    seen_keys: set[tuple] = set()

    def _on_batch_end(
        execution_id: str,
        query_plan_step: MultiAgentQueryPlanStep,
        additional_information: list[AdditionalInformation],
        iteration: int,
        batch: list[ScoredDataAgent],
        remaining: list[ScoredDataAgent],
        result_list: AlignedResultList,
        estimators: list[EstimatorStatistics | None],
        decision: EvaluatorDecision | Literal["need"],
    ) -> None:
        del additional_information, iteration, remaining, result_list
        if not (execution_id.startswith(target_execution_id + "/") and query_plan_step.id == target_step_id):
            return

        for sda in batch:
            name = sda.agent.name
            ordered.append(name)
            df = data_cache.get(name)
            if df is not None and not df.empty:
                cols = [c for c in key_cols if c in df.columns]
                if cols:
                    seen_keys.update(map(tuple, df[cols].drop_duplicates().values.tolist()))

        latest = next((e for e in reversed(estimators) if e is not None), None)
        if latest is not None:
            estimator_snapshots.append(latest)

        coverage = min(len(seen_keys) / total_unique, 1.0) if total_unique > 0 else 0.0
        batches.append(
            BatchRecord(
                batch_num=len(batches) + 1,
                cumulative_agents=len(ordered),
                decision=str(decision),
                sample_coverage=latest.sample_coverage if latest else None,
                observed_count=latest.observed_count if latest else None,
                estimated_count=latest.estimated_count if latest else None,
                observed_value_sum=latest.observed_value_sum if latest else None,
                estimated_value_sum=latest.estimated_value_sum if latest else None,
                observed_value_avg=latest.observed_value_avg if latest else None,
                estimated_value_avg=latest.estimated_value_avg if latest else None,
                coverage_achieved=coverage,
            )
        )

    hooks.register(HookEvent.MULTI_AGENT_STEP_BATCH_END, _on_batch_end)
    return ordered, batches, estimator_snapshots, _on_batch_end


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

    canonical_table = _parse_canonical_table(canonical_sql)
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
                _query_agent_canonical, agent, canonical_sql, canonical_table, meta_mapping, data_dir
            )
            for agent in step_agents
        }
        for name, future in futures.items():
            data_cache[name] = future.result()

    ground_truth = _compute_ground_truth(data_cache, key_cols, value_column)
    logging.info(
        "%s/%s: true_count=%d true_sum=%s agents=%d",
        query_slug,
        step.id,
        ground_truth.count,
        f"{ground_truth.value_sum:.2f}" if ground_truth.value_sum is not None else "n/a",
        len(step_agents),
    )

    align_handler = _make_align_schema_override(step_meta, query)
    data_handler = _make_data_agent_override(canonical_sql, canonical_table, meta_mapping, data_cache, data_dir)

    execution_id = uuid4().hex[:12]
    ordered, batches, estimator_snapshots, batch_end_handler = _register_tracking(
        execution_id, step.id, data_cache, ground_truth.count, key_cols
    )
    fresh_query_plan = QueryPlan.model_validate(raw_query_plan)
    _per_step_align[execution_id] = align_handler
    _per_step_data[execution_id] = data_handler

    try:
        with hooks.stage_context(execution_id), context.config_context(config):
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
        hooks.unregister(HookEvent.MULTI_AGENT_STEP_BATCH_END, batch_end_handler)

    agents_total = len(step.to_agent)

    # First final_result is the real stop point (continue_after_final_result may fire more batches).
    stop_batch = next((b for b in batches if b.decision == "final_result"), None)
    if stop_batch is None:
        stop_batch = batches[-1] if batches else None

    if stop_batch is None:
        logging.warning("Step %s/%s: no batch records", query_slug, step.id)
        return None

    stop_stats = next((s for s in estimator_snapshots if s.agents_queried == stop_batch.cumulative_agents), None) or (
        estimator_snapshots[-1] if estimator_snapshots else None
    )

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
        stop_stats=stop_stats,
        batches=batches,
    )
    _save_step_results(result, query_slug, output_dir)
    return result


def _relative_error(estimated: float | None, true: float | None) -> float | None:
    if estimated is None or true is None or true == 0:
        return None
    return (estimated - true) / abs(true)


def _save_step_results(result: StepResult, query_slug: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{query_slug}_{result.step_id}"
    gt = result.ground_truth
    ss = result.stop_stats

    coverage_at_stop = (ss.observed_count / gt.count) if (ss and gt.count > 0) else None

    summary = {
        "step_id": result.step_id,
        "agents_total": result.agents_total,
        "agents_queried_at_stop": result.agents_queried_at_stop,
        "stop_fraction": result.stop_fraction,
        "early_stop": result.early_stop,
        "true_count": gt.count,
        "observed_count_at_stop": ss.observed_count if ss else None,
        "estimated_count_at_stop": ss.estimated_count if ss else None,
        "coverage_at_stop": coverage_at_stop,
        "count_error": _relative_error(ss.estimated_count if ss else None, gt.count),
        "true_value_sum": gt.value_sum,
        "true_value_avg": gt.value_avg,
        "true_value_min": gt.value_min,
        "true_value_max": gt.value_max,
        "observed_value_sum_at_stop": ss.observed_value_sum if ss else None,
        "estimated_value_sum_at_stop": ss.estimated_value_sum if ss else None,
        "value_sum_error": _relative_error(ss.estimated_value_sum if ss else None, gt.value_sum),
        "observed_value_avg_at_stop": ss.observed_value_avg if ss else None,
        "estimated_value_avg_at_stop": ss.estimated_value_avg if ss else None,
        "value_avg_error": _relative_error(ss.estimated_value_avg if ss else None, gt.value_avg),
    }
    pd.DataFrame([summary]).to_csv(output_dir / f"{prefix}_evaluator_summary.csv", index=False)

    batch_rows = [
        {
            "batch_num": b.batch_num,
            "cumulative_agents": b.cumulative_agents,
            "decision": b.decision,
            "sample_coverage": b.sample_coverage,
            "observed_count": b.observed_count,
            "estimated_count": b.estimated_count,
            "coverage_achieved": b.coverage_achieved,
            "observed_value_sum": b.observed_value_sum,
            "estimated_value_sum": b.estimated_value_sum,
            "observed_value_avg": b.observed_value_avg,
            "estimated_value_avg": b.estimated_value_avg,
        }
        for b in result.batches
    ]
    pd.DataFrame(batch_rows).to_csv(output_dir / f"{prefix}_evaluator_batches.csv", index=False)

    _save_step_plot(result, gt, prefix, output_dir)
    logging.info("Saved evaluator results to %s", output_dir)


def _save_step_plot(result: StepResult, gt: GroundTruth, prefix: str, output_dir: Path) -> None:
    batches = result.batches
    if not batches:
        return

    xs = [b.cumulative_agents for b in batches]
    stop_x = result.agents_queried_at_stop

    has_value = gt.value_sum is not None

    fig, axes = plt.subplots(1, 2 if has_value else 1, figsize=(13 if has_value else 7, 5))
    ax_cov = axes[0] if has_value else axes

    coverages = [b.coverage_achieved for b in batches]
    ax_cov.plot(xs, coverages, marker="o", markersize=3, label="coverage (observed/true)")
    ax_cov.axvline(stop_x, color="red", linestyle="--", alpha=0.7, label=f"stop (agent {stop_x})")
    ax_cov.axhline(1.0, color="green", linestyle=":", alpha=0.5)
    ax_cov.set_xlabel("Agents queried")
    ax_cov.set_ylabel("Coverage (observed / true count)")
    ax_cov.set_ylim(0.0, 1.1)
    ax_cov.set_title(f"Coverage trajectory\n{prefix}")
    ax_cov.legend(fontsize=8)

    if has_value:
        ax_val = axes[1]
        est_counts = [b.estimated_count for b in batches]
        obs_counts = [b.observed_count for b in batches]
        ax_val.plot(xs, obs_counts, marker="o", markersize=3, label="observed count")
        ax_val.plot(xs, est_counts, marker="s", markersize=3, linestyle="--", label="estimated count")
        ax_val.axhline(gt.count, color="green", linestyle=":", alpha=0.7, label=f"true count ({gt.count})")
        ax_val.axvline(stop_x, color="red", linestyle="--", alpha=0.7, label=f"stop (agent {stop_x})")
        ax_val.set_xlabel("Agents queried")
        ax_val.set_ylabel("Count")
        ax_val.set_title(f"Count estimation trajectory\n{prefix}")
        ax_val.legend(fontsize=8)

    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"{prefix}_evaluator.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


async def _run(ctx: ScriptContext) -> None:
    extra = ctx.extra or {}
    steps_metadata: dict[str, Any] = extra.get("steps_metadata", {})

    raw_query_plan = extra.get("query_plan")
    if raw_query_plan is None:
        logging.warning("%s: no query_plan in extra — skipping", ctx.input_file.stem)
        return

    query_plan = QueryPlan.model_validate(raw_query_plan)
    multi_steps = [s for s in query_plan.steps if isinstance(s, MultiAgentQueryPlanStep)]

    if not multi_steps:
        logging.warning("%s: no multi-agent steps — skipping", ctx.input_file.stem)
        return

    meta_mapping = _load_meta_mapping(ctx.data_dir)
    all_agents = cast(CollaborativeQueryAnswering, ctx.data).data_agents
    relevant_agent_names: set[str] = set(extra.get("relevant_data_agents", []))
    relevant_data_agents = [a for a in all_agents if a.name in relevant_agent_names]
    query = query_plan.query
    query_slug = ctx.input_file.stem
    output_dir = ctx.output_dir or ctx.input_file.parent / "evaluation"

    hooks.clear(HookEvent.EXECUTE_QUERY_PLAN_END)
    hooks.register(HookEvent.ALIGN_SCHEMA_OVERRIDE, _align_schema_dispatch)
    hooks.register(HookEvent.DATA_AGENT_QUERY_OVERRIDE, _data_agent_dispatch)
    hooks.register(HookEvent.REPLAN_QUERY_PLAN_OVERRIDE, _replan_error)

    step_tasks = []
    for step in multi_steps:
        step_meta = steps_metadata.get(step.id)
        if step_meta is None:
            logging.warning("%s/%s: no steps_metadata — skipping", ctx.input_file.stem, step.id)
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
                output_dir=output_dir,
                data_dir=ctx.data_dir,
                config=ctx.config,
            )
        )

    try:
        await asyncio.gather(*step_tasks)
    finally:
        hooks.unregister(HookEvent.ALIGN_SCHEMA_OVERRIDE, _align_schema_dispatch)
        hooks.unregister(HookEvent.DATA_AGENT_QUERY_OVERRIDE, _data_agent_dispatch)
        hooks.unregister(HookEvent.REPLAN_QUERY_PLAN_OVERRIDE, _replan_error)


if __name__ == "__main__":
    run_script(
        description="Evaluate completeness evaluator on federated queries",
        run_fn=_run,
    )

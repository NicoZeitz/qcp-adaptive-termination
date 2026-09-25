"""Benchmark all selection strategies across all steps of federated queries."""

import asyncio
import hashlib
import json
import logging
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
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
    SelectionStrategyType,
)
from qcp_adaptive_termination.models.execution import (
    AdditionalInformation,
    EvaluatorDecision,
    ExecutionResult,
    FailedExecution,
    FinalResultOutcome,
    PartialQueryAnswer,
    QueryMoreAgentsOutcome,
)
from qcp_adaptive_termination.models.query_plan import MultiAgentQueryPlanStep, QueryPlan
from qcp_adaptive_termination.models.stages import CollaborativeQueryAnswering
from qcp_adaptive_termination.observability import context
from qcp_adaptive_termination.stages import execute_query_plan
from scripts.utils import ScriptContext, run_script
from scripts.utils.evaluation import EvaluationCurve, build_curve, compute_greedy_oracle
from scripts.utils.sql import parse_select_columns, rewrite_sql_identifiers

_META_MAPPING_CACHE: dict[str, Any] | None = None

# Per-strategy dispatch maps: strategy execution_id -> step-specific handler.
# Populated/cleaned per _run_one_strategy call; supports parallel step execution.
_per_strategy_align: dict[str, Callable[..., AlignedResultList]] = {}
_per_strategy_data: dict[str, Callable[..., PartialQueryAnswer]] = {}

# Execution ids for last-step strategy runs — only these get the JSON output.
_last_step_execution_ids: set[str] = set()


def _load_meta_mapping(data_dir: Path) -> dict[str, Any]:
    global _META_MAPPING_CACHE
    if _META_MAPPING_CACHE is None:
        mapping_path = data_dir / "dataset" / "meta_mapping.yaml"
        with open(mapping_path) as f:
            _META_MAPPING_CACHE = cast(dict[str, Any], yaml.safe_load(f) or {})
    return _META_MAPPING_CACHE


def _slug_from_agent(agent: DataAgent) -> str:
    return Path(agent.path).stem


def _agent_db_path(agent: DataAgent, data_dir: Path) -> Path:
    return data_dir / agent.path


def _parse_canonical_table(sql: str) -> str | None:
    """Extract first table name from FROM clause."""
    match = re.search(r"\bFROM\s+(\w+)", sql, re.IGNORECASE)
    return match.group(1) if match else None


def _query_agent_canonical(
    agent: DataAgent,
    canonical_sql: str,
    canonical_table: str,
    meta_mapping: dict[str, Any],
    data_dir: Path,
) -> pd.DataFrame | None:
    """Execute rewritten step SQL on agent DB, return df with canonical column names."""
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

    db_path = _agent_db_path(agent, data_dir)
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(actual_sql), conn)
        if df.empty:
            return None
        reverse_map = {v: k for k, v in col_map.items()}
        df = df.rename(columns=reverse_map)
        return df
    except Exception as exc:
        logging.warning("Failed to query %s (%s): %s", agent.name, slug, exc)
        return None
    finally:
        engine.dispose()


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


def _completeness_override(
    execution_id: str,
    aligned_result_list: AlignedResultList,
    estimator_statistics: EstimatorStatistics,
    query: str,
    previous_estimator_statistics: list[EstimatorStatistics],
    merge_task: str,
    force_result: bool,
    **kwargs: Any,
) -> QueryMoreAgentsOutcome | FinalResultOutcome:
    if force_result:
        return FinalResultOutcome(result=aligned_result_list.data, result_list=aligned_result_list)
    return QueryMoreAgentsOutcome(result_list=aligned_result_list)


def _replan_error(
    execution_id: str,
    data_agents: list[DataAgent],
    relevant_data_agents: list[DataAgent],
    query_plan: QueryPlan,
    reason: FailedExecution,
) -> QueryPlan:
    del execution_id, data_agents, relevant_data_agents, query_plan
    raise RuntimeError(f"Unexpected replan triggered in benchmark. Step: {reason.step.id!r}")


def _align_schema_dispatch(
    execution_id: str,
    data_agents: list[DataAgent],
    agent_names: list[str],
    query: str,
    merge_task: str,
) -> AlignedResultList:
    for strategy_id, handler in _per_strategy_align.items():
        if execution_id == strategy_id or execution_id.startswith(strategy_id + "/"):
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
    for strategy_id, handler in _per_strategy_data.items():
        if execution_id == strategy_id or execution_id.startswith(strategy_id + "/"):
            return handler(
                execution_id,
                data_agent=data_agent,
                query=query,
                additional_information=additional_information,
                schema_metadata=schema_metadata,
            )
    raise ValueError(f"No data_agent handler for execution_id={execution_id!r}")


def _write_last_step_stage_3(execution_id: str, attempt: int, result: ExecutionResult) -> None:
    for strategy_id in _last_step_execution_ids:
        if execution_id == strategy_id or execution_id.startswith(strategy_id + "/"):
            if output_dir := context.get_output_dir():
                path = output_dir / "stage_3_output_collaborative_query_answering_result.json"
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(result.model_dump(), f, sort_keys=False, ensure_ascii=False, indent=4)
            return


def _register_batch_tracking(target_execution_id: str, target_step_id: str) -> tuple[list[str], Any]:
    ordered: list[str] = []

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
        del additional_information, iteration, remaining, result_list, estimators, decision
        if execution_id.startswith(target_execution_id + "/") and query_plan_step.id == target_step_id:
            ordered.extend(sda.agent.name for sda in batch)

    hooks.register(HookEvent.MULTI_AGENT_STEP_BATCH_END, _on_batch_end)
    return ordered, _on_batch_end


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
                lambda a: _query_agent_canonical(a, canonical_sql, canonical_table, meta_mapping, data_dir),
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


def _save_results(
    results: list[tuple[str, EvaluationCurve]],
    oracle_curve: EvaluationCurve,
    query_slug: str,
    step_id: str,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{query_slug}_{step_id}"

    rows = [{"strategy": s, "auc": c.auc} for s, c in results]
    rows.append({"strategy": "oracle", "auc": oracle_curve.auc})
    pd.DataFrame(rows).to_csv(output_dir / f"{prefix}_auc.csv", index=False)

    curve_rows = []
    for strategy, curve in results:
        for x, y in zip(curve.x, curve.y, strict=True):
            curve_rows.append({"strategy": strategy, "x": x, "y": y})
    for x, y in zip(oracle_curve.x, oracle_curve.y, strict=True):
        curve_rows.append({"strategy": "oracle", "x": x, "y": y})
    pd.DataFrame(curve_rows).to_csv(output_dir / f"{prefix}_curves.csv", index=False)

    for xscale in ("linear", "log"):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            oracle_curve.x,
            oracle_curve.y,
            "--",
            color="black",
            label=f"oracle (AUC={oracle_curve.auc:.3f})",
            linewidth=2,
        )
        for strategy, curve in results:
            ax.plot(curve.x, curve.y, label=f"{strategy} (AUC={curve.auc:.3f})")
        ax.set_xlabel("Fraction of agents queried")
        ax.set_ylabel("Fraction of unique rows covered")
        ax.set_title(f"Selection strategy comparison ({xscale} scale)\n{query_slug} / {step_id}")
        ax.legend(fontsize=8)
        ax.set_xscale(xscale)
        if xscale == "linear":
            ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        fig.tight_layout()
        for ext in ("png", "pdf", "svg"):
            fig.savefig(output_dir / f"{prefix}_{xscale}.{ext}", dpi=150, bbox_inches="tight")
        plt.close(fig)

    logging.info("Saved to %s", output_dir)


async def _run_one_strategy(
    strategy: SelectionStrategyType,
    step: MultiAgentQueryPlanStep,
    align_handler: Callable[..., AlignedResultList],
    data_handler: Callable[..., PartialQueryAnswer],
    raw_query_plan: Any,
    relevant_data_agents: list[DataAgent],
    all_agents: list[DataAgent],
    data_cache: dict[str, pd.DataFrame | None],
    total_unique: int,
    canonical_cols: list[str],
    config: Config,
    is_last: bool,
) -> tuple[str, EvaluationCurve]:
    logging.info("Running strategy: %s / %s", step.id, strategy)
    execution_id = uuid4().hex[:12]
    strategy_config = config.model_copy(update={"selection_type": strategy})
    ordered_agents, batch_end_handler = _register_batch_tracking(execution_id, step.id)
    fresh_query_plan = QueryPlan.model_validate(raw_query_plan)

    _per_strategy_align[execution_id] = align_handler
    _per_strategy_data[execution_id] = data_handler
    if is_last:
        _last_step_execution_ids.add(execution_id)

    try:
        with hooks.stage_context(execution_id), context.config_context(strategy_config):
            await execute_query_plan(
                query_plan=fresh_query_plan,
                relevant_data_agents=relevant_data_agents,
                data_agents=all_agents,
            )
            logging.info("Strategy %s/%s completed", step.id, strategy)
    except Exception as exc:
        logging.warning("Strategy %s/%s failed: %s", step.id, strategy, exc)
        logging.exception(exc)
        if isinstance(exc, requests.exceptions.Timeout):
            logging.error("Timeout exception: %s", exc)
            raise SystemExit(1) from exc
    finally:
        _per_strategy_align.pop(execution_id, None)
        _per_strategy_data.pop(execution_id, None)
        _last_step_execution_ids.discard(execution_id)
        hooks.unregister(HookEvent.MULTI_AGENT_STEP_BATCH_END, batch_end_handler)

    seen: set[str] = set()
    deduped: list[str] = []
    for name in ordered_agents:
        if name not in seen:
            seen.add(name)
            deduped.append(name)

    if not deduped:
        logging.warning("Strategy %s/%s: no agents queried", step.id, strategy)
        return str(strategy), EvaluationCurve(x=[0.0, 1.0], y=[0.0, 0.0], auc=0.0)

    curve = build_curve(deduped, data_cache, total_unique, canonical_cols)
    logging.info("  %s/%s AUC=%.4f agents=%d", step.id, strategy, curve.auc, len(deduped))
    return str(strategy), curve


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
    is_last: bool,
) -> None:
    canonical_sql: str = step_meta["sql_query"]
    schema_meta = step_meta["aligned_schema_metadata"]
    key_cols: list[str] = schema_meta["keys"]

    canonical_table = _parse_canonical_table(canonical_sql)
    if canonical_table is None:
        logging.warning(
            "%s/%s: could not parse table from sql %r — skipping",
            query_slug,
            step.id,
            canonical_sql,
        )
        return
    canonical_cols = parse_select_columns(canonical_sql)
    if not canonical_cols:
        logging.warning(
            "%s/%s: could not parse columns from sql %r — skipping",
            query_slug,
            step.id,
            canonical_sql,
        )
        return

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
                _query_agent_canonical, agent, canonical_sql, canonical_table, meta_mapping, data_dir
            )
            for agent in step_agents
        }
        for name, future in futures.items():
            data_cache[name] = future.result()

    align_handler = _make_align_schema_override(step_meta, query)
    data_handler = _make_data_agent_override(canonical_sql, canonical_table, meta_mapping, data_cache, data_dir)

    results = list(
        await asyncio.gather(
            *[
                _run_one_strategy(
                    strategy=s,
                    step=step,
                    align_handler=align_handler,
                    data_handler=data_handler,
                    raw_query_plan=raw_query_plan,
                    relevant_data_agents=relevant_data_agents,
                    all_agents=all_agents,
                    data_cache=data_cache,
                    total_unique=total_unique,
                    canonical_cols=canonical_cols,
                    config=config,
                    is_last=is_last,
                )
                for s in SelectionStrategyType
            ]
        )
    )

    oracle_curve = compute_greedy_oracle(step.to_agent, data_cache, total_unique, canonical_cols)
    logging.info("Oracle AUC=%.4f", oracle_curve.auc)
    _save_results(results, oracle_curve, query_slug, step.id, output_dir)


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

    # Replace the auto-registered EXECUTE_QUERY_PLAN_END hook with one that
    # only writes the JSON output for the last multi-agent step's strategy runs.
    hooks.clear(HookEvent.EXECUTE_QUERY_PLAN_END)
    hooks.register(HookEvent.EXECUTE_QUERY_PLAN_END, _write_last_step_stage_3)
    # Single dispatch handlers for ALIGN_SCHEMA and DATA_AGENT overrides.
    # These route each call to the correct per-step handler via execution_id.
    hooks.register(HookEvent.ALIGN_SCHEMA_OVERRIDE, _align_schema_dispatch)
    hooks.register(HookEvent.DATA_AGENT_QUERY_OVERRIDE, _data_agent_dispatch)
    hooks.register(HookEvent.COMPLETENESS_EVALUATOR_OVERRIDE, _completeness_override)
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
                is_last=step is multi_steps[-1],
            )
        )

    try:
        await asyncio.gather(*step_tasks)
    finally:
        hooks.unregister(HookEvent.EXECUTE_QUERY_PLAN_END, _write_last_step_stage_3)
        hooks.unregister(HookEvent.ALIGN_SCHEMA_OVERRIDE, _align_schema_dispatch)
        hooks.unregister(HookEvent.DATA_AGENT_QUERY_OVERRIDE, _data_agent_dispatch)
        hooks.unregister(HookEvent.COMPLETENESS_EVALUATOR_OVERRIDE, _completeness_override)
        hooks.unregister(HookEvent.REPLAN_QUERY_PLAN_OVERRIDE, _replan_error)


if __name__ == "__main__":
    run_script(
        description="Benchmark selection strategies on a single-step federated query",
        run_fn=_run,
    )

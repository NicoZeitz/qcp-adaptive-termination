"""Benchmark all 8 selection strategies on a single-step federated query."""
import asyncio
import hashlib
import logging
import re
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
    DataAgent,
    EstimatorStatistics,
    ScoredDataAgent,
    SelectionStrategyType,
)
from qcp_adaptive_termination.models.execution import (
    AdditionalInformation,
    EvaluatorDecision,
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

_OVERRIDE_EVENTS = [
    HookEvent.ALIGN_SCHEMA_OVERRIDE,
    HookEvent.DATA_AGENT_QUERY_OVERRIDE,
    HookEvent.COMPLETENESS_EVALUATOR_OVERRIDE,
    HookEvent.MULTI_AGENT_STEP_BATCH_END,
]

_META_MAPPING_CACHE: dict[str, Any] | None = None


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


def _make_align_schema_override(step_meta: dict[str, Any], query: str):
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
):
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


def _register_batch_tracking(target_execution_id: str) -> tuple[list[str], Any]:
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
        del query_plan_step, additional_information, iteration, remaining, result_list, estimators, decision
        if execution_id.startswith(target_execution_id + "/"):
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
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [{"strategy": s, "auc": c.auc} for s, c in results]
    rows.append({"strategy": "oracle", "auc": oracle_curve.auc})
    pd.DataFrame(rows).to_csv(output_dir / f"{query_slug}_auc.csv", index=False)

    # save a more detailed CSV with the full curves
    curve_rows = []
    for strategy, curve in results:
        for x, y in zip(curve.x, curve.y, strict=True):
            curve_rows.append({"strategy": strategy, "x": x, "y": y})
    for x, y in zip(oracle_curve.x, oracle_curve.y, strict=True):
        curve_rows.append({"strategy": "oracle", "x": x, "y": y})
    pd.DataFrame(curve_rows).to_csv(output_dir / f"{query_slug}_curves.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    for ax, xscale in zip(axes, ["linear", "log"], strict=True):
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
        ax.set_title(f"Selection strategy comparison ({xscale} x)\n{query_slug}")
        ax.legend(fontsize=8)
        ax.set_xscale(xscale)
        if xscale == "linear":
            ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
    fig.tight_layout()
    fig.savefig(output_dir / f"{query_slug}_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved to %s", output_dir)


async def _run(ctx: ScriptContext) -> None:
    extra = ctx.extra or {}
    steps_metadata: dict[str, Any] = extra.get("steps_metadata", {})

    raw_query_plan = extra.get("query_plan")
    if raw_query_plan is None:
        logging.warning("%s: no query_plan in extra — skipping", ctx.input_file.stem)
        return

    query_plan = QueryPlan.model_validate(raw_query_plan)
    multi_steps = [s for s in query_plan.steps if isinstance(s, MultiAgentQueryPlanStep)]

    if len(multi_steps) != 1:
        logging.warning(
            "%s: expected exactly 1 multi-agent step, got %d — skipping",
            ctx.input_file.stem,
            len(multi_steps),
        )
        return

    step = multi_steps[0]
    step_meta = steps_metadata.get(step.id)
    if step_meta is None:
        raise ValueError(
            f"No steps_metadata entry for step id '{step.id}' in {ctx.input_file}. "
            f"Available keys: {list(steps_metadata.keys())}"
        )

    canonical_sql: str = step_meta["sql_query"]
    schema_meta = step_meta["aligned_schema_metadata"]
    key_cols: list[str] = schema_meta["keys"]
    query = query_plan.query
    query_slug = ctx.input_file.stem

    canonical_table = _parse_canonical_table(canonical_sql)
    if canonical_table is None:
        logging.warning("%s: could not parse table from sql %r — skipping", ctx.input_file.stem, canonical_sql)
        return
    canonical_cols = parse_select_columns(canonical_sql)
    if not canonical_cols:
        logging.warning("%s: could not parse columns from sql %r — skipping", ctx.input_file.stem, canonical_sql)
        return

    meta_mapping = _load_meta_mapping(ctx.data_dir)

    all_agents = cast(CollaborativeQueryAnswering, ctx.data).data_agents
    relevant_agent_names: set[str] = set(extra.get("relevant_data_agents", []))
    relevant_data_agents = [a for a in all_agents if a.name in relevant_agent_names]

    logging.info("Query: %s", query)
    logging.info("Step: %s, table: %s, cols: %s, key_cols: %s", step.id, canonical_table, canonical_cols, key_cols)

    total_unique = _get_total_unique(
        relevant_data_agents, canonical_sql, canonical_table, key_cols, meta_mapping, ctx.data_dir
    )
    logging.info("Total unique rows (ground truth): %d", total_unique)

    data_cache: dict[str, pd.DataFrame | None] = {}
    with ThreadPoolExecutor() as pool:
        futures = {
            agent.name: pool.submit(
                _query_agent_canonical, agent, canonical_sql, canonical_table, meta_mapping, ctx.data_dir
            )
            for agent in relevant_data_agents
        }
        for name, future in futures.items():
            data_cache[name] = future.result()

    align_schema_handler = _make_align_schema_override(step_meta, query)
    data_agent_handler = _make_data_agent_override(
        canonical_sql, canonical_table, meta_mapping, data_cache, ctx.data_dir
    )

    hooks.register(HookEvent.ALIGN_SCHEMA_OVERRIDE, align_schema_handler)
    hooks.register(HookEvent.DATA_AGENT_QUERY_OVERRIDE, data_agent_handler)
    hooks.register(HookEvent.COMPLETENESS_EVALUATOR_OVERRIDE, _completeness_override)
    hooks.register(HookEvent.REPLAN_QUERY_PLAN_OVERRIDE, _replan_error)

    async def _run_one_strategy(strategy: SelectionStrategyType) -> tuple[str, EvaluationCurve]:
        logging.info("Running strategy: %s", strategy)
        execution_id = uuid4().hex[:12]
        strategy_config = ctx.config.model_copy(update={"selection_type": strategy})
        ordered_agents, batch_end_handler = _register_batch_tracking(execution_id)
        fresh_query_plan = QueryPlan.model_validate(raw_query_plan)

        try:
            with hooks.stage_context(execution_id), context.config_context(strategy_config):
                await execute_query_plan(
                    query_plan=fresh_query_plan,
                    relevant_data_agents=relevant_data_agents,
                    data_agents=all_agents,
                )
                logging.info("Strategy %s completed successfully", strategy)
        except Exception as exc:
            logging.warning("Strategy %s failed: %s", strategy, exc)
            # print all information about the exception to the log
            logging.exception(exc)
            if isinstance(exc, requests.exceptions.Timeout):
                # log all information about the timeout exception
                logging.error("Timeout exception: %s", exc)
                logging.error("Timeout request: %s", str(exc.request))
                logging.error("Timeout response: %s", str(exc.response))
                # exit program with non-zero exit code to indicate failure
                raise SystemExit(1) from exc
        finally:
            hooks.unregister(HookEvent.MULTI_AGENT_STEP_BATCH_END, batch_end_handler)

        seen: set[str] = set()
        deduped: list[str] = []
        for name in ordered_agents:
            if name not in seen:
                seen.add(name)
                deduped.append(name)
        ordered_agents = deduped

        if not ordered_agents:
            logging.warning("Strategy %s: no agents queried", strategy)
            return str(strategy), EvaluationCurve(x=[0.0, 1.0], y=[0.0, 0.0], auc=0.0)

        curve = build_curve(ordered_agents, data_cache, total_unique, canonical_cols)
        logging.info("  %s AUC=%.4f, agents queried=%d", strategy, curve.auc, len(ordered_agents))
        return str(strategy), curve

    try:
        results = list(await asyncio.gather(*[_run_one_strategy(s) for s in SelectionStrategyType]))
    finally:
        hooks.unregister(HookEvent.ALIGN_SCHEMA_OVERRIDE, align_schema_handler)
        hooks.unregister(HookEvent.DATA_AGENT_QUERY_OVERRIDE, data_agent_handler)
        hooks.unregister(HookEvent.COMPLETENESS_EVALUATOR_OVERRIDE, _completeness_override)
        hooks.unregister(HookEvent.REPLAN_QUERY_PLAN_OVERRIDE, _replan_error)

    oracle_curve = compute_greedy_oracle(step.to_agent, data_cache, total_unique, canonical_cols)
    logging.info("Oracle AUC=%.4f", oracle_curve.auc)

    output_dir = ctx.output_dir or ctx.input_file.parent / "evaluation"
    _save_results(results, oracle_curve, query_slug, output_dir)


if __name__ == "__main__":
    run_script(
        description="Benchmark selection strategies on a single-step federated query",
        run_fn=_run,
    )

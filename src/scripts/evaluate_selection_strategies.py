"""Benchmark all 8 selection strategies on a single-step federated query."""

import hashlib
import logging
from pathlib import Path
from typing import Any, Literal, cast

import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import create_engine

import qcp_adaptive_termination.database as database
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
    FinalResultOutcome,
    PartialQueryAnswer,
    QueryMoreAgentsOutcome,
)
from qcp_adaptive_termination.models.query_plan import MultiAgentQueryPlanStep
from qcp_adaptive_termination.models.stages import DraftedQueryPlan
from qcp_adaptive_termination.observability import context
from qcp_adaptive_termination.stages import execute_query_plan
from scripts.utils import ScriptContext, run_script
from scripts.utils.evaluation import EvaluationCurve, build_curve, compute_greedy_oracle

_OVERRIDE_EVENTS = [
    HookEvent.ALIGN_SCHEMA_OVERRIDE,
    HookEvent.DATA_AGENT_QUERY_OVERRIDE,
    HookEvent.COMPLETENESS_EVALUATOR_OVERRIDE,
    HookEvent.MULTI_AGENT_STEP_BATCH_END,
]


def _make_align_schema_override(step_meta: dict[str, Any], query: str):
    def _handler(
        data_agents: list[DataAgent],
        agent_names: list[str],
        query: str,
        merge_task: str,
    ) -> AlignedResultList:
        meta = step_meta["aligned_schema_metadata"]
        source_col = f"source_agent_{hashlib.sha256(query.encode()).hexdigest()[:8]}"
        schema = AlignedSchemaMetadata(
            aggregation_types=[AggregationType(a) for a in meta["aggregation_types"]],
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


def _make_data_agent_override(canonical_sql: str, data_cache: dict[str, pd.DataFrame | None]):
    def _handler(
        data_agent: DataAgent,
        query: str,
        additional_information: list[AdditionalInformation],
        schema_metadata: AlignedSchemaMetadata | None,
    ) -> PartialQueryAnswer:
        if data_agent.name not in data_cache:
            try:
                engine = database.get_database_engine(data_agent)
                df = pd.DataFrame(pd.read_sql_query(canonical_sql, engine))
                if schema_metadata is not None and not df.empty:
                    cols = [c for c in schema_metadata.columns if c in df.columns]
                    if cols:
                        df = cast(pd.DataFrame, df[cols])
                data_cache[data_agent.name] = df if not df.empty else None
            except Exception as exc:
                logging.warning("Failed to query %s: %s", data_agent.name, exc)
                data_cache[data_agent.name] = None

        df = data_cache[data_agent.name]
        partial = PartialQueryAnswer()
        if df is not None and not df.empty:
            partial.set_answer(df)
        return partial

    return _handler


def _completeness_override(
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


def _register_batch_tracking() -> list[str]:
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
        ordered.extend(sda.agent.name for sda in batch)

    hooks.register(HookEvent.MULTI_AGENT_STEP_BATCH_END, _on_batch_end)
    return ordered


def _get_total_unique(data_dir: Path, sql_query: str, key_cols: list[str]) -> int:
    gt_db = data_dir / "ground-truth.db"
    engine = create_engine(f"sqlite:///{gt_db}")
    try:
        df = pd.read_sql_query(sql_query, engine)
    finally:
        engine.dispose()
    if df.empty:
        return 0
    cols = [c for c in key_cols if c in df.columns]
    return int(len(df[cols].drop_duplicates())) if cols else int(len(df))


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
    ax.set_title(f"Selection strategy comparison\n{query_slug}")
    ax.legend(fontsize=8)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    fig.tight_layout()
    fig.savefig(output_dir / f"{query_slug}_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {output_dir}")


async def _run(ctx: ScriptContext) -> None:
    extra = ctx.extra or {}
    steps_metadata: dict[str, Any] = extra.get("steps_metadata", {})

    data = cast(DraftedQueryPlan, ctx.data)
    multi_steps = [s for s in data.query_plan.steps if isinstance(s, MultiAgentQueryPlanStep)]

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
    query = data.query_plan.query
    query_slug = ctx.input_file.stem

    print(f"Query: {query}")
    print(f"Step: {step.id}, agents: {len(step.to_agent)}, key_cols: {key_cols}")

    total_unique = _get_total_unique(ctx.data_dir, canonical_sql, key_cols)
    print(f"Total unique rows (ground truth): {total_unique}")

    data_cache: dict[str, pd.DataFrame | None] = {}
    results: list[tuple[str, EvaluationCurve]] = []

    align_schema_handler = _make_align_schema_override(step_meta, query)
    data_agent_handler = _make_data_agent_override(canonical_sql, data_cache)

    for strategy in SelectionStrategyType:
        print(f"Running strategy: {strategy}")
        strategy_config = ctx.config.model_copy(update={"selection_type": strategy})
        ordered_agents = _register_batch_tracking()

        hooks.register(HookEvent.ALIGN_SCHEMA_OVERRIDE, align_schema_handler)
        hooks.register(HookEvent.DATA_AGENT_QUERY_OVERRIDE, data_agent_handler)
        hooks.register(HookEvent.COMPLETENESS_EVALUATOR_OVERRIDE, _completeness_override)

        try:
            with context.config_context(strategy_config):
                await execute_query_plan(
                    query_plan=data.query_plan,
                    relevant_data_agents=data.relevant_data_agents,
                    data_agents=data.data_agents,
                )
        except Exception as exc:
            logging.warning("Strategy %s failed: %s", strategy, exc)
        finally:
            for event in _OVERRIDE_EVENTS:
                hooks.clear(event)

        if not ordered_agents:
            logging.warning("Strategy %s: no agents queried", strategy)
            results.append((str(strategy), EvaluationCurve(x=[0.0, 1.0], y=[0.0, 0.0], auc=0.0)))
            continue

        curve = build_curve(ordered_agents, data_cache, total_unique, key_cols)
        print(f"  AUC={curve.auc:.4f}, agents queried={len(ordered_agents)}")
        results.append((str(strategy), curve))

    oracle_curve = compute_greedy_oracle(step.to_agent, data_cache, total_unique, key_cols)
    print(f"Oracle AUC={oracle_curve.auc:.4f}")

    output_dir = ctx.output_dir or ctx.input_file.parent / "evaluation"
    _save_results(results, oracle_curve, query_slug, output_dir)


if __name__ == "__main__":
    run_script(
        description="Benchmark selection strategies on a single-step federated query",
        run_fn=_run,
    )

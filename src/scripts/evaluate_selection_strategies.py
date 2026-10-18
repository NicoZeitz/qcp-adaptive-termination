"""Benchmark all selection strategies across all steps of federated queries."""

import asyncio
import hashlib
import json
import logging
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import yaml
from sqlalchemy import create_engine, text

import qcp_adaptive_termination.hooks as hooks
from qcp_adaptive_termination.estimator import Estimator
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
    FinalResultOutcome,
    PartialQueryAnswer,
    QueryMoreAgentsOutcome,
)
from qcp_adaptive_termination.models.query_plan import (
    MultiAgentQueryPlanStep,
    QueryPlan,
)
from qcp_adaptive_termination.models.stages import CollaborativeQueryAnswering
from qcp_adaptive_termination.observability import context
from qcp_adaptive_termination.stages.execution.multi_agent import answer_step_multiple_agents
from scripts.utils import ScriptContext, run_script
from scripts.utils.evaluation import EvaluationCurve, build_curve, compute_greedy_oracle, compute_relative_auc
from scripts.utils.sql import parse_select_columns, rewrite_sql_identifiers

_META_MAPPING_CACHE: dict[str, Any] | None = None

# Per-strategy dispatch maps: strategy execution_id -> step-specific handler.
# Populated/cleaned per _run_one_strategy call; supports parallel step execution.
_per_strategy_align: dict[str, Callable[..., AlignedResultList]] = {}
_per_strategy_data: dict[str, Callable[..., PartialQueryAnswer]] = {}


@dataclass(slots=True)
class StepSummary:
    """Benchmark outcome for one step, used for summary aggregation."""

    query_slug: str
    step_id: str
    agents_total: int
    oracle_auc: float
    final_row_count: int
    curves: dict[str, EvaluationCurve] = field(default_factory=dict)
    failures: list[tuple[str, str]] = field(default_factory=list)

    @property
    def is_find_step(self) -> bool:
        """Find step: full result set is a single row (plus oracle AUC = 1)."""
        return self.final_row_count == 1


def _load_meta_mapping(data_dir: Path) -> dict[str, Any]:
    global _META_MAPPING_CACHE
    if _META_MAPPING_CACHE is None:
        mapping_path = data_dir / "dataset" / "_meta" / "meta_mapping.yaml"
        with open(mapping_path) as f:
            _META_MAPPING_CACHE = cast(dict[str, Any], yaml.safe_load(f) or {})
    return _META_MAPPING_CACHE


def _slug_from_agent(agent: DataAgent) -> str:
    return Path(agent.path).parent.name


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


def _estimator_statistics_override(
    execution_id: str,
    result_list: AlignedResultList,
    agents_queried: int,
    agents_total: int,
    agents_current: list[str],
    previous_estimator: Estimator | None,
) -> tuple[EstimatorStatistics, Estimator | None]:
    return (
        EstimatorStatistics(
            sample_coverage=None,
            new_unique_count=0,
            agents_queried=agents_queried,
            agents_total=agents_total,
            aggregation_types=result_list.schema_metadata.aggregation_types,
            observed_count=0,
        ),
        None,
    )


def _align_schema_dispatch(
    execution_id: str,
    data_agents: list[DataAgent],
    agent_names: list[str],
    query: str,
    merge_task: str,
) -> AlignedResultList:
    for strategy_id, handler in _per_strategy_align.items():
        if execution_id == strategy_id:
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
        if execution_id == strategy_id:
            return handler(
                execution_id,
                data_agent=data_agent,
                query=query,
                additional_information=additional_information,
                schema_metadata=schema_metadata,
            )
    raise ValueError(f"No data_agent handler for execution_id={execution_id!r}")


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
        if execution_id == target_execution_id and query_plan_step.id == target_step_id:
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
    pd.DataFrame(curve_rows).to_csv(output_dir / f"{prefix}_curves.csv", index=False)

    for xscale in ("linear", "log"):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            range(len(oracle_curve.x)),
            oracle_curve.y,
            "--",
            color="black",
            label=f"oracle (AUC={oracle_curve.auc:.3f})",
            linewidth=2,
            drawstyle="steps-post",
        )
        for strategy, curve in results:
            ax.plot(
                range(len(curve.x)),
                curve.y,
                label=f"{strategy} (AUC={curve.auc:.3f})",
                drawstyle="steps-post",
            )
        ax.set_xlabel("Agents queried")
        ax.set_ylabel("Fraction of unique rows covered")
        ax.set_title(f"Selection strategy comparison ({xscale} scale)\n{query_slug} / {step_id}")
        ax.legend(fontsize=8)
        ax.set_xscale(xscale)
        x_max = max(len(oracle_curve.x) - 1, max((len(c.x) - 1 for _, c in results), default=0))
        if xscale == "linear":
            ax.set_xlim(0.0, x_max)
        else:
            ax.set_xlim(1.0, x_max)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.set_ylim(0.0, 1.05)
        fig.tight_layout()
        for ext in ("png", "pdf", "svg"):
            fig.savefig(output_dir / f"{prefix}_{xscale}.{ext}", dpi=150, bbox_inches="tight")
        plt.close(fig)

    logging.info("Saved to %s", output_dir)


def _average_curves(summaries: list[StepSummary]) -> dict[str, list[tuple[float, float]]]:
    """Average coverage over steps at a common fraction grid, per strategy (incl. oracle)."""
    if not summaries:
        return {}
    grid = [i / 100 for i in range(101)]
    all_names: list[str] = ["oracle"]
    for summary in summaries:
        for name in summary.curves:
            if name not in all_names:
                all_names.append(name)

    averaged: dict[str, list[tuple[float, float]]] = {}
    for name in all_names:
        values: list[list[float]] = []
        for summary in summaries:
            curve = summary.curves.get(name)
            if curve is None:
                continue
            series = pd.Series(curve.y, index=curve.x).reindex(grid, method="ffill").fillna(0.0)
            values.append(series.tolist())
        if not values:
            continue
        means = [sum(col) / len(values) for col in zip(*values, strict=True)]
        averaged[name] = list(zip(grid, means, strict=True))
    return averaged


def _write_summary(
    summaries: list[StepSummary],
    output_dir: Path,
    filename_prefix: str,
    title_suffix: str,
) -> None:
    """Write summary CSVs and averaged fraction plots for a set of step summaries."""
    if not summaries:
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    auc_rows = []
    failure_rows = []
    for summary in summaries:
        for strategy, curve in summary.curves.items():
            auc_rows.append(
                {
                    "query": summary.query_slug,
                    "step": summary.step_id,
                    "strategy": strategy,
                    "auc": curve.auc,
                    "relative_auc": compute_relative_auc(curve.auc, summary.oracle_auc),
                    "oracle_auc": summary.oracle_auc,
                    "agents_total": summary.agents_total,
                    "final_row_count": summary.final_row_count,
                }
            )
        failure_rows.extend(
            {"query": summary.query_slug, "step": summary.step_id, "strategy": strategy, "error": error}
            for strategy, error in summary.failures
        )
    per_step_auc = pd.DataFrame(auc_rows)
    per_step_auc.to_csv(output_dir / f"{filename_prefix}_per_step_auc.csv", index=False)
    pd.DataFrame(failure_rows, columns=["query", "step", "strategy", "error"]).to_csv(
        output_dir / f"{filename_prefix}_failures.csv", index=False
    )

    mean_auc = (
        per_step_auc.groupby("strategy")["auc"]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    mean_auc.to_csv(output_dir / f"{filename_prefix}_mean_auc.csv", index=False)

    averaged = _average_curves(summaries)
    curve_rows = [
        {"strategy": name, "agents_fraction": x, "coverage": y} for name, points in averaged.items() for x, y in points
    ]
    pd.DataFrame(curve_rows).to_csv(output_dir / f"{filename_prefix}_curves.csv", index=False)

    for xscale in ("linear", "log"):
        fig, ax = plt.subplots(figsize=(10, 6))
        oracle_points = averaged.get("oracle")
        if oracle_points is not None:
            oracle_mean_auc = mean_auc.loc[mean_auc["strategy"] == "oracle", "mean"].iloc[0]
            ax.plot(
                [x for x, _ in oracle_points],
                [y for _, y in oracle_points],
                "--",
                color="black",
                label=f"oracle (mean AUC={oracle_mean_auc:.3f})",
                linewidth=2,
                drawstyle="steps-post",
            )
        for name, points in averaged.items():
            if name == "oracle":
                continue
            mean_auc_value = mean_auc.loc[mean_auc["strategy"] == name, "mean"].iloc[0]
            ax.plot(
                [x for x, _ in points],
                [y for _, y in points],
                label=f"{name} (mean AUC={mean_auc_value:.3f})",
                drawstyle="steps-post",
            )
        ax.set_xlabel("Fraction of agents queried")
        ax.set_ylabel("Fraction of unique rows covered")
        ax.set_title(f"Selection strategy comparison ({xscale} scale){title_suffix}")
        ax.legend(fontsize=8)
        ax.set_xscale(xscale)
        if xscale == "linear":
            ax.set_xlim(0.0, 1.0)
        else:
            ax.set_xlim(0.01, 1.0)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.set_ylim(0.0, 1.05)
        fig.tight_layout()
        for ext in ("png", "pdf", "svg"):
            fig.savefig(output_dir / f"{filename_prefix}_{xscale}.{ext}", dpi=150, bbox_inches="tight")
        plt.close(fig)

    logging.info("Summary saved to %s (%s)", output_dir, filename_prefix)


def aggregate_selection_results(results_dir: Path, input_dir: Path) -> Path:
    """Aggregate latest per-query curves into all-step and no-find summaries."""
    aggregate_dir = results_dir / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    step_summaries: list[StepSummary] = []
    status_rows: list[dict[str, Any]] = []
    known_strategies = {str(strategy) for strategy in SelectionStrategyType}

    for input_file in sorted(input_dir.glob("*.json")):
        query_slug = input_file.stem
        query_dir = results_dir / query_slug
        run_dirs = [path for path in query_dir.iterdir() if path.is_dir()] if query_dir.is_dir() else []
        if not run_dirs:
            status_rows.append({"query": query_slug, "status": "missing_output", "details": "no run directory"})
            continue

        run_dir = max(run_dirs, key=lambda path: path.stat().st_mtime_ns)
        summary_path = run_dir / "summary_all_per_step_auc.csv"
        if not summary_path.is_file():
            status_rows.append({"query": query_slug, "status": "missing_summary", "details": str(summary_path)})
            continue

        with input_file.open(encoding="utf-8") as stream:
            raw_input = json.load(stream)
        raw_plan = (raw_input.get("extra") or {}).get("query_plan")
        expected_steps = (
            {step.id for step in QueryPlan.model_validate(raw_plan).steps if isinstance(step, MultiAgentQueryPlanStep)}
            if raw_plan is not None
            else set()
        )
        summary_frame = pd.read_csv(summary_path)
        summary_by_step = {
            str(step_id): rows.to_dict("records") for step_id, rows in summary_frame.groupby("step", sort=False)
        }

        logs = ""
        for log_name in ("stderr.log", "stdout.log"):
            log_path = run_dir / log_name
            if log_path.is_file():
                logs += log_path.read_text(errors="replace") + "\n"

        failures: dict[tuple[str, str], str] = {}
        for match in re.finditer(r"Strategy (s\d+)/(\w+): no agents queried", logs):
            failures.setdefault((match.group(1), match.group(2)), "no agents queried")
        for match in re.finditer(r"Strategy (s\d+)/(\w+) failed: (.+)", logs):
            failures[(match.group(1), match.group(2))] = match.group(3).strip()

        failure_path = run_dir / "summary_all_failures.csv"
        if failure_path.is_file():
            for row in pd.read_csv(failure_path).to_dict("records"):
                failures[(str(row["step"]), str(row["strategy"]))] = str(row["error"])

        missing_steps = expected_steps - set(summary_by_step)
        query_failure_count = 0
        missing_curves: list[str] = []
        for step_id, summary_rows in summary_by_step.items():
            curve_path = run_dir / f"{query_slug}_{step_id}_curves.csv"
            if not curve_path.is_file():
                missing_curves.append(step_id)
                continue

            metadata_by_strategy = {str(row["strategy"]): row for row in summary_rows}
            curve_frame = pd.read_csv(curve_path)
            curves: dict[str, EvaluationCurve] = {}
            oracle_auc: float | None = None
            oracle_agents = 0
            final_row_count = 0

            for strategy, rows in curve_frame.groupby("strategy", sort=False):
                strategy = str(strategy)
                rows = rows.sort_values("agents_fraction")
                x = [float(value) for value in rows["agents_fraction"]]
                y = [float(value) for value in rows["coverage"]]
                auc = sum(y[1:]) / max(1, len(y) - 1)

                if strategy == "oracle":
                    oracle_auc = auc
                    oracle_agents = int(metadata_by_strategy.get("oracle", {}).get("agents_total", len(y) - 1))
                    final_row_count = int(metadata_by_strategy.get("oracle", {}).get("final_row_count", 0))
                    curves[strategy] = EvaluationCurve(x=x, y=y, auc=auc)
                    continue

                error = failures.get((step_id, strategy))
                if error is not None:
                    query_failure_count += 1
                    continue
                curves[strategy] = EvaluationCurve(x=x, y=y, auc=auc)

            for strategy in known_strategies - curves.keys() - {"oracle"}:
                if (step_id, strategy) not in failures:
                    failures[(step_id, strategy)] = "missing strategy curve"
                    query_failure_count += 1

            if oracle_auc is None:
                missing_curves.append(f"{step_id}:oracle")
                continue

            step_summaries.append(
                StepSummary(
                    query_slug=query_slug,
                    step_id=step_id,
                    agents_total=oracle_agents,
                    oracle_auc=oracle_auc,
                    final_row_count=final_row_count,
                    curves=curves,
                    failures=[
                        (strategy, error)
                        for (failed_step, strategy), error in failures.items()
                        if failed_step == step_id
                    ],
                )
            )

        status = "complete_with_failures" if query_failure_count else "complete"
        details = []
        if missing_steps:
            status = "partial"
            details.append(f"missing steps: {','.join(sorted(missing_steps))}")
        if missing_curves:
            status = "partial"
            details.append(f"missing curves: {','.join(missing_curves)}")
        status_rows.append(
            {
                "query": query_slug,
                "status": status,
                "expected_steps": len(expected_steps),
                "completed_steps": len(summary_by_step),
                "failed_strategies": query_failure_count,
                "details": "; ".join(details),
            }
        )

    _write_summary(step_summaries, aggregate_dir, "summary_all_queries", " — all queries and steps")
    non_find_steps = [summary for summary in step_summaries if not summary.is_find_step]
    _write_summary(non_find_steps, aggregate_dir, "summary_no_find_queries", " — all queries, excluding find steps")
    pd.DataFrame(status_rows).to_csv(aggregate_dir / "aggregate_run_status.csv", index=False)
    logging.info(
        "Aggregated %d/%d input queries to %s", len(status_rows), len(list(input_dir.glob("*.json"))), aggregate_dir
    )
    return aggregate_dir


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
    ordered_agents, batch_end_handler = _register_batch_tracking(execution_id, step.id)
    # Benchmark the step in isolation: dependencies dropped, canonical SQL self-contained.
    isolated_step = step.model_copy(update={"dependencies": []})

    _per_strategy_align[execution_id] = align_handler
    _per_strategy_data[execution_id] = data_handler

    try:
        with hooks.stage_context(execution_id), context.config_context(strategy_config):
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
        hooks.unregister(HookEvent.MULTI_AGENT_STEP_BATCH_END, batch_end_handler)

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
) -> StepSummary | None:
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
                _query_agent_canonical, agent, canonical_sql, canonical_table, meta_mapping, data_dir
            )
            for agent in step_agents
        }
        for name, future in futures.items():
            data_cache[name] = future.result()

    align_handler = _make_align_schema_override(step_meta, query)
    data_handler = _make_data_agent_override(canonical_sql, canonical_table, meta_mapping, data_cache, data_dir)

    strategy_runs = list(
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

    # Single dispatch handlers for ALIGN_SCHEMA and DATA_AGENT overrides.
    # These route each call to the correct per-step handler via execution_id.
    hooks.register(HookEvent.ALIGN_SCHEMA_OVERRIDE, _align_schema_dispatch)
    hooks.register(HookEvent.DATA_AGENT_QUERY_OVERRIDE, _data_agent_dispatch)
    hooks.register(HookEvent.COMPLETENESS_EVALUATOR_OVERRIDE, _completeness_override)
    hooks.register(HookEvent.ESTIMATOR_STATISTICS_OVERRIDE, _estimator_statistics_override)

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
                meta_mapping=meta_mapping,
                query=query,
                query_slug=query_slug,
                output_dir=output_dir,
                data_dir=ctx.data_dir,
                config=ctx.config,
            )
        )

    try:
        gathered = await asyncio.gather(*step_tasks)
        step_summaries = [s for s in gathered if s is not None]
    finally:
        hooks.unregister(HookEvent.ALIGN_SCHEMA_OVERRIDE, _align_schema_dispatch)
        hooks.unregister(HookEvent.DATA_AGENT_QUERY_OVERRIDE, _data_agent_dispatch)
        hooks.unregister(HookEvent.COMPLETENESS_EVALUATOR_OVERRIDE, _completeness_override)
        hooks.unregister(HookEvent.ESTIMATOR_STATISTICS_OVERRIDE, _estimator_statistics_override)

    # Summary over all steps of this file.
    _write_summary(step_summaries, output_dir, "summary_all", " — all steps")
    # Summary excluding find steps (single-row result sets, ill-posed for overlap).
    non_find = [s for s in step_summaries if not s.is_find_step]
    _write_summary(non_find, output_dir, "summary_no_find", " — excluding find steps")


if __name__ == "__main__":
    run_script(
        description="Benchmark selection strategies on a single-step federated query",
        run_fn=_run,
    )

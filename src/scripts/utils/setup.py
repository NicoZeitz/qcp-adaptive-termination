"""Shared setup utilities for pipeline and evaluation runs."""

import argparse
import csv
import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import yaml

from qcp_adaptive_termination import hooks, run_stages
from qcp_adaptive_termination.hooks import HookEvent
from qcp_adaptive_termination.models import Config, DataAgent
from qcp_adaptive_termination.models.execution import ExecutionResult
from qcp_adaptive_termination.models.query_plan import QueryPlan
from qcp_adaptive_termination.models.stages import (
    CollaborativeQueryAnswering,
    CollaborativeQueryAnsweringResult,
    DraftedQueryPlan,
    FilteredRelevantAgents,
    Stage,
)
from qcp_adaptive_termination.observability import context
from qcp_adaptive_termination.observability.usage import CSV_COLUMNS, UsageReport
from qcp_adaptive_termination.pipeline import load_config, load_input


def register_usage_hooks(usage_report: UsageReport) -> None:
    """Register usage/cost hooks (LLM and embedding)."""

    @hooks.post(HookEvent.AGENT_COMPLETED)
    def _write_llm_usage(execution_id: str, role, agent_name: str | None, label: str | None, model: str, usage) -> None:
        """Record one LLM run's usage into the report."""
        del execution_id
        usage_report.record_llm(role, agent_name, label, model, usage)

    @hooks.post(HookEvent.EMBEDDING_COMPLETED)
    def _write_embedding_usage(execution_id: str, text: str, model: str, prompt_tokens: int) -> None:
        """Record one embedding computation's usage into the report."""
        del execution_id
        usage_report.record_embedding(text, model, prompt_tokens)


def register_stage_output_hooks() -> None:
    """Register stage input/output JSON dump hooks for full-pipeline runs."""

    def _write_output_file(filename: str, data: dict) -> None:
        if output_dir := context.get_output_dir():
            path = output_dir / filename
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, sort_keys=False, ensure_ascii=False, indent=4)

    @hooks.pre(HookEvent.SETUP_LOAD_INPUT)
    def _write_input_file(
        execution_id: str,
        input_file: Path,
        data: CollaborativeQueryAnswering
        | CollaborativeQueryAnsweringResult
        | DraftedQueryPlan
        | FilteredRelevantAgents,
        extra: dict | None,
    ) -> None:
        if output_dir := context.get_output_dir():
            shutil.copy(input_file, output_dir / input_file.name)

    @hooks.pre(HookEvent.SETUP_LOAD_CONFIG)
    def _write_config_file(execution_id: str, config: Config) -> None:
        if output_dir := context.get_output_dir():
            with open(output_dir / "config.yaml", "w", encoding="utf-8") as f:
                yaml.dump(config.model_dump(), f, sort_keys=False, allow_unicode=True, default_flow_style=False)
            logging.info(f"Config: {config.model_dump_json(ensure_ascii=False)}")

    @hooks.pre(HookEvent.DISCOVERY_FILTER_RELEVANT_AGENTS)
    def _write_stage_1_input(execution_id: str, query: str, data_agents: list[DataAgent]) -> None:  # noqa: ARG001
        _write_output_file(
            "stage_1_input.json",
            {"query": query, "data_agents": [a.model_dump() for a in data_agents]},
        )

    @hooks.post(HookEvent.DISCOVERY_FILTER_RELEVANT_AGENTS)
    def _write_stage_1(execution_id: str, result: FilteredRelevantAgents) -> None:
        _write_output_file(
            "stage_1_output_filtered_relevant_agents.json",
            result.model_dump(),
        )

    @hooks.pre(HookEvent.PLANNING_STAGE)
    def _write_stage_2_input(
        execution_id: str, query: str, data_agents: list[DataAgent], relevant_data_agents: list[DataAgent]
    ) -> None:
        _write_output_file(
            "stage_2_input_filtered_relevant_agents.json",
            {
                "query": query,
                "data_agents": [a.model_dump() for a in data_agents],
                "relevant_data_agents": [a.model_dump() for a in relevant_data_agents],
            },
        )

    @hooks.post(HookEvent.PLANNING_STAGE)
    def _write_stage_2(execution_id: str, result: DraftedQueryPlan) -> None:
        _write_output_file(
            "stage_2_output_drafted_query_plan.json",
            result.model_dump(),
        )

    @hooks.pre(HookEvent.EXECUTION_EXECUTE_QUERY_PLAN)
    def _write_stage_3_input(
        execution_id: str, attempt: int, query_plan: QueryPlan, current_data_agents: list[DataAgent]
    ) -> None:
        _write_output_file(
            "stage_3_input_drafted_query_plan.json",
            {
                "attempt": attempt,
                "query_plan": query_plan.model_dump(),
                "current_data_agents": [a.model_dump() for a in current_data_agents],
            },
        )

    @hooks.post(HookEvent.EXECUTION_EXECUTE_QUERY_PLAN)
    def _write_stage_3(execution_id: str, result: ExecutionResult) -> None:
        _write_output_file(
            "stage_3_output_collaborative_query_answering_result.json",
            result.model_dump(),
        )


def _write_usage_reports(output_dir: Path, usage_report: UsageReport, config: Config) -> None:
    """Write the per-call usage CSV and JSON next to the run outputs."""
    (output_dir / "usage.json").write_text(json.dumps(usage_report.summarize(config), indent=2), encoding="utf-8")
    with (output_dir / "usage.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(usage_report.call_rows(config))


@dataclass(slots=True)
class ScriptContext:
    args: argparse.Namespace
    datetime_str: str
    config: Config
    input_file: Path
    output_dir: Path | None
    cache_dir: Path | None
    data_dir: Path
    dataset_dir: Path | None
    start_stage: Stage
    end_stage: Stage
    name: str
    trace_name: str
    data: CollaborativeQueryAnswering | FilteredRelevantAgents | DraftedQueryPlan | CollaborativeQueryAnsweringResult
    extra: dict[str, Any] | None

    @property
    def data_agents(self) -> list[DataAgent]:
        if isinstance(self.data, (CollaborativeQueryAnswering, FilteredRelevantAgents, DraftedQueryPlan)):
            return self.data.data_agents
        raise TypeError(f"Input type {type(self.data).__name__} has no data_agents")


async def run_script(
    *,
    args: argparse.Namespace,
    run_fn=None,
    before_fn=None,
    after_fn=None,
    config: Config | None = None,
):
    """Run the pipeline (or the provided ``run_fn``) for one explicit args set.

    Args:
        args: parsed CLI arguments for this run
        run_fn: async function to run instead of the pipeline
        before_fn: async function run before the run with the script context
        after_fn: async function run after the run with the context and result
        config: config to use, loaded from args when omitted
    """
    datetime_str = datetime.now().strftime("%Y-%m-%d %H-%M-%S")

    config = config or load_config(args.config)
    input_file = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else None
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else input_file.parent.parent.parent
    data_dir = dataset_dir.parent

    start_stage = cast(Stage, args.start_stage)
    end_stage = cast(Stage, args.end_stage)
    name = f"{args.name} - {datetime_str}" if args.name else datetime_str
    cache_mode = config.cache_mode

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    usage_report = UsageReport()
    register_usage_hooks(usage_report)

    if start_stage.order > end_stage.order:
        raise ValueError("start_stage must be <= end_stage")

    max_trace_label_len = 50
    name_part = f"{name.strip()[:max_trace_label_len]} - " if name else ""
    trace_name = f"[QCP] {name_part}{input_file.stem[:max_trace_label_len]}: Stage {start_stage}-{end_stage}"

    data, extra = load_input(input_file)

    async def _runner():
        with context.pipeline_context(
            name=name,
            trace_name=trace_name,
            output_dir=output_dir,
            cache_dir=cache_dir,
            data_dir=data_dir,
            cache_mode=cache_mode,
            config=config,
            trace_metadata={
                "start_stage": start_stage.value,
                "end_stage": end_stage.value,
                "input_file": str(input_file),
            },
        ):
            ctx = ScriptContext(
                args=args,
                datetime_str=datetime_str,
                config=config,
                input_file=input_file,
                output_dir=output_dir,
                cache_dir=cache_dir,
                data_dir=data_dir,
                dataset_dir=dataset_dir,
                start_stage=start_stage,
                end_stage=end_stage,
                name=name,
                trace_name=trace_name,
                data=data,
                extra=extra,
            )

            if before_fn is not None:
                await before_fn(ctx)

            if run_fn is not None:
                result = await run_fn(ctx)
            else:
                result = await run_stages(
                    input_file=input_file,
                    start_stage=start_stage,
                    end_stage=end_stage,
                )

            if after_fn is not None:
                await after_fn(ctx, result)

            return result

    result = await _runner()
    if output_dir:
        _write_usage_reports(output_dir, usage_report, config)
    return result

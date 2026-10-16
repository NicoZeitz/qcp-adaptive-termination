import asyncio
import json
import logging
import shutil
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from qcp_adaptive_termination import hooks
from qcp_adaptive_termination.hooks import HookEvent
from qcp_adaptive_termination.models import result_type
from qcp_adaptive_termination.observability import context
from qcp_adaptive_termination.pipeline import load_config, run_pipeline

if TYPE_CHECKING:
    from qcp_adaptive_termination.models import Config, DataAgent
    from qcp_adaptive_termination.models.execution import ExecutionResult
    from qcp_adaptive_termination.models.query_plan import QueryPlan
    from qcp_adaptive_termination.models.stages import (
        CollaborativeQueryAnswering,
        CollaborativeQueryAnsweringResult,
        DraftedQueryPlan,
        FilteredRelevantAgents,
    )


def register_default_output_hooks() -> None:
    def _write_output_file(filename: str, data: dict) -> None:
        if output_dir := context.get_output_dir():
            path = output_dir / filename
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, sort_keys=False, ensure_ascii=False, indent=4)

    @hooks.on(HookEvent.LOAD_INPUT)
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

    @hooks.on(HookEvent.LOAD_CONFIG)
    def _write_config_file(execution_id: str, config: Config) -> None:
        if output_dir := context.get_output_dir():
            with open(output_dir / "config.yaml", "w", encoding="utf-8") as f:
                yaml.dump(config.model_dump(), f, sort_keys=False, allow_unicode=True, default_flow_style=False)
            logging.info(f"Config: {config.model_dump_json(ensure_ascii=False)}")

    @hooks.on(HookEvent.FILTER_RELEVANT_AGENTS_START)
    def _write_stage_1_input(execution_id: str, query: str, data_agents: list[DataAgent]) -> None:  # noqa: ARG001
        _write_output_file(
            "stage_1_input.json",
            {"query": query, "data_agents": [a.model_dump() for a in data_agents]},
        )

    @hooks.on(HookEvent.FILTER_RELEVANT_AGENTS_END)
    def _write_stage_1(execution_id: str, filtered_relevant_agents: FilteredRelevantAgents) -> None:
        _write_output_file(
            "stage_1_output_filtered_relevant_agents.json",
            filtered_relevant_agents.model_dump(),
        )

    @hooks.on(HookEvent.FILTER_RELEVANT_AGENTS_END)
    def _write_stage_1_llm_stats(execution_id: str, filtered_relevant_agents: FilteredRelevantAgents) -> None:  # noqa: ARG001
        acc = context.reset_usage_accumulator()
        _write_output_file(
            "stage_1_output_llm_stats.json",
            {
                "llm_calls": acc.llm_calls,
                "input_tokens": acc.input_tokens,
                "output_tokens": acc.output_tokens,
                "total_tokens": acc.input_tokens + acc.output_tokens,
                "llm_duration_ms": acc.duration_ms,
                "embedding_calls": acc.embedding_calls,
                "embedding_duration_ms": acc.embedding_duration_ms,
                "ranking_duration_ms": acc.ranking_duration_ms,
            },
        )

    @hooks.on(HookEvent.EMBEDDING_PRE_FILTER_END)
    def _write_embedding_pre_filter(
        execution_id: str,  # noqa: ARG001
        query: str,
        embedding_filtered_agents: list[DataAgent],
        total_agents: int,
        l2_scores: dict[str, float],
        best_columns: dict[str, str] | None = None,
        after_kneedle_agents: list[str] | None = None,
        after_llm_filter_agents: list[str] | None = None,
    ) -> None:
        _write_output_file(
            "stage_1_output_embedding_pre_filter.json",
            {
                "query": query,
                "embedding_filtered_agents": [a.name for a in embedding_filtered_agents],
                "total_agents": total_agents,
                "l2_scores": l2_scores,
                **({"best_columns": best_columns} if best_columns is not None else {}),
                **({"after_kneedle_agents": after_kneedle_agents} if after_kneedle_agents is not None else {}),
                **({"after_llm_filter_agents": after_llm_filter_agents} if after_llm_filter_agents is not None else {}),
            },
        )

    @hooks.on(HookEvent.DRAFT_QUERY_PLAN_START)
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

    @hooks.on(HookEvent.DRAFT_QUERY_PLAN_END)
    def _write_stage_2(execution_id: str, drafted_query_plan: DraftedQueryPlan) -> None:
        _write_output_file(
            "stage_2_output_drafted_query_plan.json",
            drafted_query_plan.model_dump(),
        )

    @hooks.on(HookEvent.EXECUTE_QUERY_PLAN_START)
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

    @hooks.on(HookEvent.EXECUTE_QUERY_PLAN_END)
    def _write_stage_3(execution_id: str, attempt: int, result: ExecutionResult) -> None:
        _write_output_file(
            "stage_3_output_collaborative_query_answering_result.json",
            result.model_dump(),
        )


def argument_parser(
    prog: str = "qcp-adaptive-termination", description="Collaborative Query Processing with Adaptive Termination"
) -> ArgumentParser:
    parser = ArgumentParser(
        prog="qcp-adaptive-termination",
        description="Collaborative Query Processing with Adaptive Termination",
    )
    parser.add_argument("--name", type=str, default="", help="Name of the run")
    parser.add_argument("--input", type=str, help="Input file path")
    parser.add_argument(
        "--start-stage",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="Stage to start the pipeline from (1, 2, or 3)",
    )
    parser.add_argument(
        "--end-stage",
        type=int,
        choices=[1, 2, 3],
        default=3,
        help="Stage after which to end the pipeline at (1, 2, or 3)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config as a YAML file path or inline YAML string",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for logs and traces (default: .logs/ at project root)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory for cache files (default: .cache/ at project root)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Root directory for agent data files (default: parent of input file)",
    )
    return parser


def main() -> None:
    args = argument_parser().parse_args()

    datetime_str = datetime.now().strftime("%Y-%m-%d %H-%M-%S")

    config = load_config(args.config)
    input_file = Path(args.input)
    output_dir = Path(args.output_dir) / datetime_str if args.output_dir else None
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    data_dir = Path(args.data_dir) if args.data_dir else input_file.parent

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    register_default_output_hooks()

    result = asyncio.run(
        run_pipeline(
            name=f"{args.name} - {datetime_str}" if args.name else datetime_str,
            input_file=input_file,
            start_stage=args.start_stage,
            end_stage=args.end_stage,
            output_dir=output_dir,
            cache_dir=cache_dir,
            data_dir=data_dir,
            cache_mode=config.cache_mode,
            config=config,
        )
    )

    if result_type.is_result_type(result):
        print("Pipeline completed. Result preview:")
        print(json.dumps(result_type.serialize(result, context="preview"), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

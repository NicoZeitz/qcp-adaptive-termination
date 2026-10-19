import asyncio
import json
import logging
import shutil
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

import yaml

from qcp_adaptive_termination import hooks
from qcp_adaptive_termination.hooks import HookEvent
from qcp_adaptive_termination.models import result_type
from qcp_adaptive_termination.models.stages import Stage, SynthesizedResult
from qcp_adaptive_termination.observability import context
from qcp_adaptive_termination.pipeline import load_config, run_pipeline

if TYPE_CHECKING:
    from qcp_adaptive_termination.models import Config, DataAgent, ResultType
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
        type=Stage,
        choices=list(Stage),
        default=Stage.DISCOVERY,
        help="Stage to start the pipeline from",
    )
    parser.add_argument(
        "--end-stage",
        type=Stage,
        choices=list(Stage),
        default=Stage.EXECUTION,
        help="Stage after which to end the pipeline at",
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
        "--dataset_dir",
        type=str,
        default=None,
        help="Dataset folder containing _meta/input (default: derived from input file)",
    )
    return parser


def main() -> None:
    args = argument_parser().parse_args()

    datetime_str = datetime.now().strftime("%Y-%m-%d %H-%M-%S")

    config = load_config(args.config)
    input_file = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else None
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else input_file.parent.parent.parent
    data_dir = dataset_dir.parent

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
        print(
            json.dumps(
                result_type.serialize(cast(ResultType, result), context="preview"),
                indent=2,
                ensure_ascii=False,
            )
        )
    elif isinstance(result, SynthesizedResult):
        print("Pipeline completed. Result summary:")
        print(result.summary)
        print("Result preview:")
        print(
            json.dumps(
                result_type.serialize(result.result, context="preview"),
                indent=2,
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()

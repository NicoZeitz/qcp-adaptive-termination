import logging
import json
from pathlib import Path
from typing import cast
from datetime import datetime

from agents import trace
from langfuse import get_client

from common.model.execution_config import ExecutionConfig
from common.preview import result_to_json_preview
from stages.planning import draft_query_plan, STAGE_NAME as PLANNING_STAGE_NAME
from common.model.stages.drafted_query_plan import DraftedQueryPlan
from stages.execution import execute_query_plan, STAGE_NAME as EXECUTION_STAGE_NAME
from common.model.stages.collaborative_query_answering_result import (
    CollaborativeQueryAnsweringResult,
)
from stages.discovery import filter_relevant_agents, STAGE_NAME as DISCOVERY_STAGE_NAME
from common.model.stages.collaborative_query_answering import (
    CollaborativeQueryAnswering,
)
from common.model.stages.filtered_relevant_agents import FilteredRelevantAgents
from common.cache import load_cache_file
from common.initialization import initialize


stages_mapping = {
    1: DISCOVERY_STAGE_NAME,
    2: PLANNING_STAGE_NAME,
    3: EXECUTION_STAGE_NAME,
    99: "DEBUG",
}


async def main(
    name: str,
    input_file: Path,
    cache_dir: Path,
    log_dir: Path,
    start_stage: int,
    end_stage: int,
    throw_on_cache_miss: bool = False,
    disable_cache: bool = False,
    execution_config: ExecutionConfig | None = None,
):
    if start_stage > end_stage:
        raise Exception

    max_trace_label_len = 50

    log_filename = f"Stage {start_stage}-{end_stage} - {datetime.now().strftime('%Y-%m-%d %H-%M-%S')}"

    initialize(
        log_filename,
        log_dir,
        cache_dir,
        throw_on_cache_miss,
        disable_cache,
    )

    if execution_config is not None:
        ExecutionConfig.set(**execution_config.model_dump())
    config = ExecutionConfig.get()

    logging.info(f"ExecutionConfig: {config.model_dump_json()}")
    config_output_path = log_dir / f"{log_filename} - execution_config.json"
    with open(config_output_path, "w") as f:
        f.write(config.model_dump_json(indent=4))

    name_prefix = f"{name.strip()[:max_trace_label_len]} - " if name else ""
    input_file_prefix = input_file.stem[:max_trace_label_len]
    trace_label = (
        f"[NZ] {name_prefix}{input_file_prefix}: Stage {start_stage}-{end_stage}"
    )

    logging.info(
        f"NZ: Running {stages_mapping[start_stage]} - {stages_mapping[end_stage]}: for input path {input_file}"
    )

    trace_metadata = {
        "start_stage": str(start_stage),
        "end_stage": str(end_stage),
        "name": name,
        "input_file": str(input_file),
        "cache_dir": str(cache_dir),
        "log_dir": str(log_dir),
        "execution_config": config.model_dump_json(),
    }
    langfuse_input = {
        **trace_metadata,
        "execution_config": config.model_dump(),
    }

    langfuse = get_client()
    with (
        langfuse.start_as_current_observation(
            name=trace_label, input=langfuse_input, metadata=trace_metadata
        ),
        trace(trace_label, metadata=trace_metadata) as trace_ctx,
    ):
        data, extra = load_cache_file(
            input_file,
            deserializable_classes=[
                CollaborativeQueryAnswering,
                FilteredRelevantAgents,
                DraftedQueryPlan,
                CollaborativeQueryAnsweringResult,
            ],
        )

        logging.info("Successfully loaded input file. Starting execution.")
        logging.info(f"Trace ID: {trace_ctx.trace_id}")
        logging.info("Extra entries in cache_file:")
        if extra:
            for k, v in extra.items():
                logging.info(f"  {k}: {v}")
        else:
            logging.info("  No extra entries found.")

        if start_stage == 99:
            logging.info(f"Starting stage {start_stage}, ENTERING DEBUG CODE")
            from stages.execution.selection import AgentSelector
            from common.model.selection.selection_type import SelectionType

            with ExecutionConfig.override(
                selection_type=SelectionType.THETA_SKETCH_UCB
            ):
                selector = await AgentSelector.create(
                    data.data_agents,
                    data.query,
                )
                ranked = selector.rank(selector.selections)
                for i, ranked_item in enumerate(ranked):
                    logging.info(
                        f"Ranked {i}: {ranked_item.agent.name} - {ranked_item.score}"
                    )

            return

        if start_stage == 1:
            logging.info(f"Starting {stages_mapping[start_stage]}")
            logging.info(f"Data: {data}")

            data = cast(CollaborativeQueryAnswering, data)
            data = await filter_relevant_agents(data.query, data.data_agents)
            data = cast(FilteredRelevantAgents, data)

            logging.info("Finished stage 1")
            logging.info(f"Result: {data}")

            start_stage += 1
            if end_stage < start_stage:
                return

        if start_stage == 2:
            logging.info(f"Starting {stages_mapping[start_stage]}")
            logging.info(f"Data: {data}")

            data = cast(FilteredRelevantAgents, data)
            data = await draft_query_plan(
                data.query, data.data_agents, data.relevant_data_agents
            )
            data = cast(DraftedQueryPlan, data)

            logging.info("Finished stage 2")
            logging.info(f"Result: {data}")

            start_stage += 1
            if end_stage < start_stage:
                return

        if start_stage == 3:
            logging.info(f"Starting {stages_mapping[start_stage]}")
            logging.info(f"Data: {data}")

            data = cast(DraftedQueryPlan, data)
            data = await execute_query_plan(
                data.query_plan,
                data.relevant_data_agents,
                data.data_agents,
            )
            data = cast(CollaborativeQueryAnsweringResult, data)

            logging.info("Finished stage 3")
            logging.info(f"Result: {data}")
            logging.info(result_to_json_preview(data.result))
            output_path = log_dir / f"{log_filename} - final_result.json"
            with open(output_path, "w") as f:
                f.write(data.model_dump_json(indent=4, ensure_ascii=False))

            # TODO: remove iteration records
            from stages.execution.answer_step_multiple_agents import (
                MULTI_AGENT_ITERATION_RECORDS,
            )

            logging.info(
                f"Iteration counts per execution_id: { {k: len(v) for k, v in MULTI_AGENT_ITERATION_RECORDS.items()} }"
            )
            for execution_id, records in MULTI_AGENT_ITERATION_RECORDS.items():
                records_output_path = (
                    log_dir
                    / f"{log_filename} - iteration_records - {execution_id}.json"
                )
                with open(records_output_path, "w", encoding="utf-8") as f:
                    json.dump(
                        [r.model_dump() for r in records],
                        f,
                        indent=4,
                        ensure_ascii=False,
                        default=str,
                    )

                run_prefix = str(log_dir / log_filename)
                try:
                    from scripts.plot_iteration_records import (
                        extract_series,
                        load_final_result_value,
                        load_reasoning_by_agents_queried,
                        load_records,
                        plot,
                        save_reasoning_text,
                    )

                    plot_records = load_records(str(records_output_path))
                    series = extract_series(plot_records)
                    count_gt = series["last_observed_count"] or 0
                    value_gt = (
                        (extra.get("full_answer") if extra else None)
                        or load_final_result_value(run_prefix)
                        or series["last_observed_value"]
                        or 0
                    )

                    traces_file = log_dir / f"{log_filename}_traces.jsonl"
                    reasoning_by_n = None
                    if traces_file.exists():
                        reasoning_by_n = load_reasoning_by_agents_queried(
                            str(traces_file)
                        )
                        if reasoning_by_n:
                            save_reasoning_text(
                                series,
                                reasoning_by_n,
                                run_prefix + " - completeness reasoning.txt",
                            )

                    plot(
                        series,
                        count_gt,
                        value_gt,
                        run_prefix,
                        include_confidence_intervals=config.include_confidence_intervals,
                    )
                    logging.info(f"Plots saved to {run_prefix}.png/.svg/.pdf")
                except Exception as e:
                    logging.warning(f"Plotting failed for {execution_id}: {e}")


if __name__ == "__main__":
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(prog="Collaborative Query Answering")
    parser.add_argument(
        "--name",
        type=str,
        default="",
        help="Name of the execution (default: empty string)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="../data/imdb/queries/Who played Rocket in the movie called 'Guardians of the Galaxy'?.json",
        help="Name of the input file",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="../.cache",
        help="Directory for cache files (default: ../.cache)",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="../.logs",
        help="Directory for log files (default: ../.logs)",
    )
    parser.add_argument(
        "--start_stage",
        type=int,
        choices=[1, 2, 3, 99],
        default=1,
        help="Stage to start execution from (default: 1)",
    )
    parser.add_argument(
        "--end_stage",
        type=int,
        choices=[1, 2, 3, 99],
        default=3,
        help="Stage after which to end execution (default: 3)",
    )
    parser.add_argument(
        "--throw-on-cache-miss",
        action="store_true",
        help="Whether to throw an error if cache miss occurs",
    )
    parser.add_argument(
        "--disable-cache",
        action="store_true",
        help="Disable caching completely (WRITE_ONLY mode)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Execution config as JSON, e.g. '{\"include_confidence_intervals\": false}'",
    )
    args = parser.parse_args()

    execution_config = (
        ExecutionConfig.model_validate_json(args.config) if args.config else None
    )

    asyncio.run(
        main(
            args.name,
            Path(args.input),
            Path(args.cache_dir),
            Path(args.log_dir),
            args.start_stage,
            args.end_stage,
            args.throw_on_cache_miss,
            args.disable_cache,
            execution_config,
        )
    )

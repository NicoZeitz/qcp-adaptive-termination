import logging
import json
from dataclasses import asdict
from pathlib import Path
from typing import Literal, cast
from datetime import datetime

from agents import trace
from opentelemetry import trace as otel_trace

from utils.preview import result_to_json_preview
from stages.planning import draft_query_plan
from qtypes.drafted_query_plan import DraftedQueryPlan
from stages.execution import execute_query_plan
from qtypes.collaborative_query_answering_result import (
    CollaborativeQueryAnsweringResult,
)
from stages.discovery import filter_relevant_agents
from qtypes.collaborative_query_answering import CollaborativeQueryAnswering
from qtypes.filtered_relevant_agents import FilteredRelevantAgents
from utils.cache import load_cache_file
from utils.initialization import initialize


stages_mapping = {
    1: "Discovery",
    2: "Planning",
    3: "Execution",
}


async def main(
    name: str,
    input_file: Path,
    cache_dir: Path,
    log_dir: Path,
    start_stage: int,
    end_stage: int,
    provider: Literal["OpenAI", "University", "Work"],
    throw_on_cache_miss: bool = False,
    disable_cache: bool = False,
):
    if start_stage > end_stage:
        raise Exception

    max_trace_label_len = 50

    log_filename = f"Stage {start_stage}-{end_stage} - {datetime.now().strftime('%Y-%m-%d %H-%M-%S')}"

    name_prefix = f"{name.strip()[:max_trace_label_len]} - " if name else ""
    input_file_prefix = input_file.stem[:max_trace_label_len]
    trace_label = (
        f"[NZ] {name_prefix}{input_file_prefix}: Stage {start_stage}-{end_stage}"
    )

    initialize(
        log_filename,
        log_dir,
        cache_dir,
        provider,
        throw_on_cache_miss,
        disable_cache,
    )

    logging.info(
        f"NZ: Running stage {start_stage} {stages_mapping[start_stage]} - {end_stage} {stages_mapping[end_stage]}: for input path {input_file} with provider {provider}"
    )

    tracer = otel_trace.get_tracer(__name__)
    with (
        trace(
            trace_label,
            metadata={
                "provider": provider,
                "start_stage": str(start_stage),
                "end_stage": str(end_stage),
                "name": name,
                "input_file": str(input_file),
                "cache_dir": str(cache_dir),
                "log_dir": str(log_dir),
            },
        ) as trace_ctx,
        tracer.start_as_current_span(trace_label),
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

        if start_stage == 1:
            logging.info(f"Starting stage 1: {stages_mapping[1]}")
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
            logging.info(f"Starting stage 2: {stages_mapping[2]}")
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
            logging.info(f"Starting stage 3: {stages_mapping[3]}")
            logging.info(f"Data: {data}")

            data = cast(DraftedQueryPlan, data)
            data = await execute_query_plan(
                data.query_plan,
                data.relevant_data_agents,
                data.data_agents,
            )
            data = cast(CollaborativeQueryAnsweringResult, data)

            from stages.execution import last_iteration_records

            logging.info("Finished stage 3")
            logging.info(f"Result: {data}")
            logging.info(f"Iteration records: {len(last_iteration_records)}")
            logging.info(result_to_json_preview(data.result))
            output_path = log_dir / f"{log_filename} - final_result.json"
            with open(output_path, "w") as f:
                f.write(data.model_dump_json(indent=4, ensure_ascii=False))

            if last_iteration_records:
                records_output_path = (
                    log_dir / f"{log_filename} - iteration_records.json"
                )
                serialized = [
                    {
                        "step_id": r.step_id,
                        "agent_name": r.agent_name,
                        "result_list": r.result_list.to_dict(orient="records"),
                        "estimator_result": asdict(r.estimator_result)
                        if r.estimator_result is not None
                        else None,
                        "evaluator_decision": r.evaluator_decision,
                        "evaluator_next_agents": r.evaluator_next_agents,
                    }
                    for r in last_iteration_records
                ]
                with open(records_output_path, "w", encoding="utf-8") as f:
                    json.dump(serialized, f, indent=4, ensure_ascii=False, default=str)


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
        choices=[1, 2, 3],
        default=1,
        help="Stage to start execution from (default: 1)",
    )
    parser.add_argument(
        "--end_stage",
        type=int,
        choices=[1, 2, 3],
        default=3,
        help="Stage after which to end execution (default: 3)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["OpenAI", "University", "Work"],
        default="OpenAI",
        help="Provider to use for the agents (default: OpenAI)",
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
    args = parser.parse_args()

    asyncio.run(
        main(
            args.name,
            Path(args.input),
            Path(args.cache_dir),
            Path(args.log_dir),
            args.start_stage,
            args.end_stage,
            args.provider,
            args.throw_on_cache_miss,
            args.disable_cache,
        )
    )

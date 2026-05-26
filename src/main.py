import logging
import os
from pathlib import Path
from typing import Any, Literal, cast
from datetime import datetime

from agents import trace

from stages.draft_query_plan import draft_query_plan
from qtypes.drafted_query_plan import DraftedQueryPlan
from stages.execute_query_plan import execute_query_plan
from qtypes.collaborative_query_answering_result import (
    CollaborativeQueryAnsweringResult,
)
from stages.filter_relevant_agents import filter_relevant_agents
from qtypes.collaborative_query_answering import CollaborativeQueryAnswering
from qtypes.filtered_relevant_agents import FilteredRelevantAgents
from stages.utils.cache import load_cache_file
from stages.utils.initialization import initialize


stages_mapping = {
    1: "Filter Relevant Agents",
    2: "Draft Query Plan",
    3: "Execute Query Plan",
}


async def main(
    input_file: Path,
    cache_dir: Path,
    log_dir: Path,
    start_stage: int,
    end_stage: int,
    provider: Literal["University", "Work"],
    throw_on_cache_miss: bool = False,
):
    if start_stage > end_stage:
        raise Exception

    user = "NZ" if os.getlogin() == "I550817" else "FW"
    log_filename = f"Stage {start_stage}-{end_stage} - {datetime.now().strftime('%Y-%m-%d %H-%M-%S')}"

    initialize(log_filename, log_dir, cache_dir, provider, throw_on_cache_miss)
    logging.info(
        f"{user}: Running stage {start_stage} {stages_mapping[start_stage]} - {end_stage} {stages_mapping[end_stage]}: for input path {input_file} with provider {provider}"
    )

    data, extra = load_cache_file(
        input_file,
        deserializable_classes=[
            CollaborativeQueryAnswering,
            FilteredRelevantAgents,
            DraftedQueryPlan,
            CollaborativeQueryAnsweringResult,
        ],
    )

    logging.info(f"{user}: Successfully loaded input file. Starting execution.")
    logging.info(f"Extra entries in cache_file:")
    if extra:
        for k, v in extra.items():
            logging.info(f"  {k}: {v}")
    else:
        logging.info("  No extra entries found.")


    with trace(
        f"{user}: Collaborative Query Answering", metadata={"provider": provider}
    ):
        if start_stage == 1:
            logging.info(f"{user}: Starting stage 1: Filter Relevant Agents")
            logging.info(f"Data: {data}")

            data = cast(CollaborativeQueryAnswering, data)
            data = await filter_relevant_agents(data.query, data.data_agents)
            data = cast(FilteredRelevantAgents, data)

            logging.info(f"{user}: Finished stage 1")
            logging.info(f"Result: {data}")

            start_stage += 1
            if end_stage < start_stage:
                return

        if start_stage == 2:
            logging.info(f"{user}: Starting stage 2: Draft Query Plan")
            logging.info(f"Data: {data}")

            data = cast(FilteredRelevantAgents, data)
            data = await draft_query_plan(
                data.query, data.data_agents, data.relevant_data_agents
            )
            data = cast(DraftedQueryPlan, data)

            logging.info(f"{user}: Finished stage 2")
            logging.info(f"Result: {data}")

            start_stage += 1
            if end_stage < start_stage:
                return

        if start_stage == 3:
            logging.info(f"{user}: Starting stage 3: Execute Query Plan")
            logging.info(f"Data: {data}")

            data = cast(DraftedQueryPlan, data)
            data = await execute_query_plan(
                data.query_plan,
                data.relevant_data_agents,
                data.data_agents,
            )
            data = cast(CollaborativeQueryAnsweringResult, data)

            logging.info(f"{user}: Finished stage 3")
            logging.info(f"Result: {data}")


if __name__ == "__main__":
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(prog="Collaborative Query Answering")
    parser.add_argument("--throw-on-cache-miss", action="store_true", help="Whether to throw an error if cache miss occurs")
    parser.add_argument("--start_stage", type=int, choices=[1, 2, 3], default=1, help="Stage to start execution from (default: 1)")
    parser.add_argument("--end_stage", type=int, choices=[1, 2, 3], default=3, help="Stage after which to end execution (default: 3)")
    parser.add_argument("--input", type=str, default="../data/imdb/queries/Who played Rocket in the movie called 'Guardians of the Galaxy'?.json", help="Name of the input file")
    parser.add_argument("--cache_dir", type=str, default="../.cache", help="Directory for cache files (default: ../.cache)")
    parser.add_argument("--log_dir", type=str, default="../.logs", help="Directory for log files (default: ../.logs)")
    parser.add_argument(
        "--provider", type=str, choices=["University", "Work"], default="University", help="Provider to use for the agents (default: University)"
    )
    args = parser.parse_args()

    asyncio.run(
        main(
            Path(args.input),
            Path(args.cache_dir),
            Path(args.log_dir),
            args.start_stage,
            args.end_stage,
            args.provider,
            args.throw_on_cache_miss,
        )
    )

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
    cache_filename: str,
    cache_dir: Path,
    log_dir: Path,
    start_stage: int,
    end_stage: int,
    provider: Literal["University", "Work"],
):
    if start_stage > end_stage:
        raise Exception

    user = "NZ" if os.getlogin() == "I550817" else "FW"
    log_filename = f"Stage {start_stage}-{end_stage} - {datetime.now().strftime('%Y-%m-%d %H-%M-%S')}"

    initialize(log_filename, log_dir, cache_dir, provider)
    logging.info(
        f"{user}: Running stage {start_stage} {stages_mapping[start_stage]} - {end_stage} {stages_mapping[end_stage]}: for cache path {cache_dir / cache_filename} with provider {provider}"
    )

    data: Any = load_cache_file(
        cache_dir / cache_filename,
        deserializable_classes=[
            CollaborativeQueryAnswering,
            FilteredRelevantAgents,
            DraftedQueryPlan,
            CollaborativeQueryAnsweringResult,
        ],
    )

    with trace(
        f"{user}: Collaborative Query Answering", metadata={"provider": provider}
    ):
        if start_stage == 1:
            data = cast(CollaborativeQueryAnswering, data)
            data = await filter_relevant_agents(data.query, data.data_agents)
            data = cast(FilteredRelevantAgents, data)

            start_stage += 1
            if end_stage < start_stage:
                return

        if start_stage == 2:
            data = cast(FilteredRelevantAgents, data)
            data = await draft_query_plan(
                data.query, data.data_agents, data.relevant_data_agents
            )
            data = cast(DraftedQueryPlan, data)

            start_stage += 1
            if end_stage < start_stage:
                return

        if start_stage == 3:
            data = cast(DraftedQueryPlan, data)
            data = await execute_query_plan(
                data.query_plan,
                data.relevant_data_agents,
                data.data_agents,
            )
            data = cast(CollaborativeQueryAnsweringResult, data)

            logging.info(data)


if __name__ == "__main__":
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(prog="Collaborative Query Answering")
    # TODO: continue running later stages + exit on cache miss
    parser.add_argument("--start_stage", type=int, choices=[1, 2, 3, 4], default=1)
    parser.add_argument("--end_stage", type=int, choices=[1, 2, 3, 4], default=3)
    parser.add_argument("--name", type=str, default="cache.json")
    parser.add_argument("--cache_dir", type=str, default="../.cache")
    parser.add_argument("--log_dir", type=str, default="../.logs")
    parser.add_argument(
        "--provider", type=str, choices=["University", "Work"], default="University"
    )
    args = parser.parse_args()

    asyncio.run(
        main(
            args.name,
            Path(args.cache_dir),
            Path(args.log_dir),
            args.start_stage,
            args.end_stage,
            args.provider,
        )
    )

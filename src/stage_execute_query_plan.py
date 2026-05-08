import os
from typing import Literal
import json

from agents import trace

from data_agent.data_agent import DataAgent
from qtypes.query_plan.query_plan import QueryPlan
from query_coordinator.default_query_coordinator import DefaultQueryCoordinator
from query_coordinator.query_coordinator import QueryCoordinator
from utils.initialization import initialize, langfuse
from utils.load_agents import (
    create_sql_data_agents_from_folder,
    create_text_agents_from_folder,
)

stages_mapping = [
    "Filter relevant agents by Query Coordinator",
    "Filter relevant agents by asking the agents directly",
    "Draft a Query Plan",
    "Execute a Query Plan",
]


async def main(
    sql_data_agents_path: str,
    text_data_agents_path: str,
    test_file: str,
    stage: int,
    provider: Literal["University", "Work"],
):
    user = "NZ" if os.getlogin() == "I550817" else "FW"

    with open(test_file, "r") as f:
        test_data = json.load(f)

    query = test_data["query"]
    answer = test_data["answer"]
    sql_query = test_data["sql_query"]
    relevant_agents = test_data["relevant_data_agents"]
    query_plan = QueryPlan.load_from_json(test_data["query_plan"])

    with trace(
        f"{user}: Stage {stage} - {stages_mapping[stage - 1]}",
        metadata={"test_file": test_file, "query": query, "provider": provider},
    ):
        data_agents: list[DataAgent] = []
        data_agents.extend(create_sql_data_agents_from_folder(sql_data_agents_path))
        data_agents.extend(create_text_agents_from_folder(text_data_agents_path))
        query_coordinator: QueryCoordinator = DefaultQueryCoordinator()

        data_agents = [agent for agent in data_agents if agent.name in relevant_agents]

        print(f"QUERY: {query}")
        result = await query_coordinator.execute_query_plan(data_agents, query_plan)

        print(f"RESULT: {result}")
        print(f"EXPECTED ANSWER: {answer}")

    langfuse.flush()


if __name__ == "__main__":
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(prog="Collaborative Query Answering")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4], default=1)
    parser.add_argument(
        "--test-file",
        type=str,
        default="../test/01.json",
    )
    parser.add_argument("--sql-data-agents", type=str, default="../data/handmade")
    parser.add_argument("--text-data-agents", type=str, default="../data/text")
    parser.add_argument(
        "--provider", type=str, choices=["University", "Work"], default="University"
    )
    args = parser.parse_args()

    initialize(args.provider)
    asyncio.run(
        main(
            args.sql_data_agents,
            args.text_data_agents,
            args.test_file,
            args.stage,
            args.provider,
        )
    )

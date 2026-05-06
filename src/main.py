from typing import Literal
import os

from agents import trace

from query_coordinator.default_query_coordinator import DefaultQueryCoordinator
from query_coordinator.query_coordinator import QueryCoordinator
from utils.initialization import initialize, langfuse
from data_agent.data_agent import DataAgent
from utils.load_agents import (
    create_sql_data_agents_from_folder,
    create_text_agents_from_folder,
)


async def main(
    sql_data_agents_path: str,
    text_data_agents_path: str,
    query: str,
    provider: Literal["University", "Work"],
):
    user = "NZ" if os.getlogin() == "I550817" else "FW"

    with trace(
        f"{user}: Collaborative Query Answering",
        metadata={"query": query, "provider": provider},
    ):
        data_agents: list[DataAgent] = []
        data_agents.extend(create_sql_data_agents_from_folder(sql_data_agents_path))
        data_agents.extend(create_text_agents_from_folder(text_data_agents_path))

        query_coordinator: QueryCoordinator = DefaultQueryCoordinator()

        print(query)
        result = await query_coordinator.collaborative_query_answering(
            query, data_agents
        )
        print(result)

    langfuse.flush()


if __name__ == "__main__":
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(prog="Collaborative Query Answering")
    parser.add_argument(
        "--query",
        type=str,
        default="Who played Rocket in the movie called 'Guardians of the Galaxy'?",
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
            args.query,
            args.provider,
        )
    )

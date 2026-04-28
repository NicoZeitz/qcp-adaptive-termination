import json
import os
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, text
from agents import Agent, ModelSettings, trace, set_default_openai_client
from openai import AsyncOpenAI
from openai.types.shared import Reasoning

from modules.cache import Cache
from modules.model import DataAgent, AgentContext
from modules.query_coordinator import QueryCoordinator
from modules.sql_data_agent import SqlDataAgent


def create_data_agents_from_folder(folder_path: str) -> list[DataAgent]:
    data_agents: list[DataAgent] = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".db"):
            agent_name = os.path.splitext(file_name)[0]
            db_engine = create_engine(
                f"sqlite:///{os.path.abspath(os.path.join(folder_path, file_name))}"
            )

            with db_engine.connect() as conn:
                conn.execute(text("PRAGMA journal_mode=WAL;"))
                conn.execute(text("PRAGMA foreign_keys=OFF;"))
                conn.execute(text("PRAGMA journal_mode=MEMORY;"))
                conn.execute(text("PRAGMA temp_store=MEMORY;"))
                conn.execute(text("PRAGMA synchronous=OFF;"))
                conn.execute(text("PRAGMA query_only = ON;"))
                conn.commit()

            data_agent = SqlDataAgent(
                agent=Agent[AgentContext](
                    name=agent_name,
                    model="gpt-5-mini",
                    model_settings=ModelSettings(
                        reasoning=Reasoning(effort="low"),
                        verbosity="low",
                        parallel_tool_calls=True,
                    ),
                ),
                db_engine=db_engine,
            )
            data_agents.append(data_agent)

    return data_agents


def load_query(query_path: str) -> tuple[str, str, Any]:
    user_query = Path(query_path).stem
    with open(query_path) as f:
        content = json.load(f)
        sql_query = list(content.keys())[0]
        expected_result = content[sql_query]

    return user_query, sql_query, expected_result


async def main(data_agents_path: str, query_path: str):
    user_query, _sql_query, expected_result = load_query(query_path)

    with trace("NZ: Collaborative Query Answering", metadata={"query": user_query}):
        print(
            "================================================== USER QUERY =================================================="
        )
        print(user_query)

        data_agents = create_data_agents_from_folder(data_agents_path)

        queryCoordinator = QueryCoordinator(
            agent=Agent[AgentContext](
                name="Query Coordinator",
                # https://developers.openai.com/api/docs/pricing
                model="gpt-5",
                model_settings=ModelSettings(
                    reasoning=Reasoning(effort="high"),
                    verbosity="low",
                    parallel_tool_calls=True,
                ),
            )
        )

        for data_agent in data_agents:
            queryCoordinator.register_data_agent(data_agent)

        result = await queryCoordinator.collaborative_query_answering(user_query)
        print(
            "================================================== RESULT =================================================="
        )
        print(result)
        print(
            "================================================== EXPECTED RESULT =================================================="
        )
        print(expected_result)


if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv
    import argparse

    load_dotenv()

    # custom_client = AsyncOpenAI(base_url=os.getenv("BASE_URL"), api_key=os.getenv("API_KEY"))
    custom_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    set_default_openai_client(custom_client)

    parser = argparse.ArgumentParser(prog="Collaborative Query Answering")
    parser.add_argument("--data-agents", type=str, default="../data/handmade")
    parser.add_argument(
        "--query",
        type=str,
        default="../test/Who played Rocket in the movie called 'Guardians of the Galaxy'?.json",
    )
    parser.add_argument("--cache", action="store_true", default=False)
    args = parser.parse_args()

    if args.cache:
        Cache.instance().enable()
    else:
        Cache.instance().disable()

    asyncio.run(main(args.data_agents, args.query))

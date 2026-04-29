import json
import os
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, text
from agents import Agent, ModelSettings, trace, set_default_openai_client
from openai import AsyncOpenAI
from openai.types.shared import Reasoning

from modules.cache import Cache
from modules.model import DataAgent, AgentContext, ReplanMode
from modules.query_coordinator import QueryCoordinator
from modules.sql_data_agent import SqlDataAgent
from modules.text_data_agent import TextDataAgent


def get_default_data_agent_config(agent_name: str) -> Agent[AgentContext]:
    return Agent[AgentContext](
        name=agent_name,
        model="gpt-5",
        model_settings=ModelSettings(
            reasoning=Reasoning(effort="medium"),
            verbosity="medium",
            parallel_tool_calls=True,
        ),
    )


def get_default_query_coordinator_agent_config(agent_name: str) -> Agent[AgentContext]:
    # https://developers.openai.com/api/docs/pricing
    return Agent[AgentContext](
        name=agent_name,
        model="gpt-5",
        model_settings=ModelSettings(
            reasoning=Reasoning(effort="high"),
            verbosity="medium",
            parallel_tool_calls=True,
        ),
    )


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
                agent=get_default_data_agent_config(agent_name),
                db_engine=db_engine,
            )
            data_agents.append(data_agent)

    return data_agents


def create_text_agents_from_folder(folder_path: str) -> list[DataAgent]:
    data_agents: list[DataAgent] = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            agent_name = os.path.splitext(file_name)[0]

            with open(os.path.join(folder_path, file_name)) as f:
                data_agent = TextDataAgent(
                    agent=get_default_data_agent_config(agent_name), text=f.read()
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


async def main(
    sql_data_agents_path: str,
    text_data_agents_path: str,
    query_path: str,
    replan_mode: ReplanMode,
):
    user_query, _sql_query, expected_result = load_query(query_path)

    with trace("NZ: Collaborative Query Answering", metadata={"query": user_query}):
        print(
            "================================================== REGISTERING =================================================="
        )
        queryCoordinator = QueryCoordinator(
            agent=get_default_query_coordinator_agent_config("Query Coordinator"),
            replan_mode=replan_mode,
        )

        sql_data_agents = create_data_agents_from_folder(sql_data_agents_path)
        for data_agent in sql_data_agents:
            queryCoordinator.register_data_agent(data_agent)

        text_data_agents = create_text_agents_from_folder(text_data_agents_path)
        for data_agent in text_data_agents:
            queryCoordinator.register_data_agent(data_agent)

        print(
            "================================================== USER QUERY =================================================="
        )
        print(user_query)
        print(
            "================================================== ANSWERING =================================================="
        )
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
    parser.add_argument("--sql-data-agents", type=str, default="../data/handmade")
    parser.add_argument("--text-data-agents", type=str, default="../data/text")
    parser.add_argument(
        "--query",
        type=str,
        default="../test/Who was the cinematographer of Donald Trump's favorite movie?.json",
    )
    parser.add_argument(
        "--cache",
        type=str,
        choices=["READ_AND_WRITE", "READ_ONLY", "WRITE_ONLY", "DISABLED"],
        default="DISABLED",
    )
    parser.add_argument(
        "--replan-mode",
        type=str,
        choices=["global", "local"],
        default="local",
    )
    args = parser.parse_args()

    Cache.instance().set_mode(args.cache)
    asyncio.run(
        main(
            args.sql_data_agents,
            args.text_data_agents,
            args.query,
            ReplanMode(args.replan_mode),
        )
    )

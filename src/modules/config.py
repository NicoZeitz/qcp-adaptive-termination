import json
import os
from pathlib import Path
from typing import Any

from agents import Agent, ModelSettings
from openai.types.shared import Reasoning
from sqlalchemy import create_engine, text

from modules.model import AgentContext, DataAgent
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


def create_sql_data_agents_from_folder(folder_path: str) -> list[DataAgent]:
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

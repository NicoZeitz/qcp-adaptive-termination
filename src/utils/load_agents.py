import os

from sqlalchemy import create_engine, text

from data_agent.data_agent import DataAgent
from data_agent.sql_data_agent.sql_data_agent import SqlDataAgent
from data_agent.text_data_agent.text_data_agent import TextDataAgent


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
                name=agent_name,
                db_engine=db_engine,
            )

            data_agents.append(data_agent)

    data_agents.sort(key=lambda agent: agent.name)
    return data_agents


def create_text_agents_from_folder(folder_path: str) -> list[DataAgent]:
    data_agents: list[DataAgent] = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            agent_name = os.path.splitext(file_name)[0]

            with open(os.path.join(folder_path, file_name)) as f:
                data_agent = TextDataAgent(name=agent_name, text=f.read())
                data_agents.append(data_agent)

    data_agents.sort(key=lambda agent: agent.name)
    return data_agents

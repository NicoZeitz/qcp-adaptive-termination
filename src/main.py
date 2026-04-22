import os

from sqlalchemy import create_engine
from agents import Agent, ModelSettings, trace, set_default_openai_client
from openai import AsyncOpenAI
from openai.types.shared import Reasoning
from dotenv import load_dotenv

from modules.model import DataAgent, AgentContext
from modules.query_coordinator import QueryCoordinator
from modules.sql_data_agent import SqlDataAgent

load_dotenv()

# custom_client = AsyncOpenAI(base_url=os.getenv("BASE_URL"), api_key=os.getenv("API_KEY"))
custom_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
set_default_openai_client(custom_client)


def create_data_agents_from_folder(folder_path: str) -> list[DataAgent]:
    data_agents: list[DataAgent] = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".db"):
            agent_name = os.path.splitext(file_name)[0]
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
                db_engine=create_engine(
                    f"sqlite:///{os.path.abspath(os.path.join(folder_path, file_name))}"
                ),
            )
            data_agents.append(data_agent)

    return data_agents


def create_data_agents(amount: int) -> list[DataAgent]:
    data_agents: list[DataAgent] = []

    for i in range(1, amount + 1):
        data_agent = SqlDataAgent(
            agent=Agent[AgentContext](
                name=f"A{i}",
                model="gpt-5-mini",
                model_settings=ModelSettings(
                    reasoning=Reasoning(effort="low"),
                    verbosity="low",
                    parallel_tool_calls=True,
                ),
            ),
            db_engine=create_engine(
                f"sqlite:///{os.path.abspath(f'../data/agent{i}.db')}"
            ),
        )
        data_agents.append(data_agent)

    return data_agents


async def main():
    user_query = "What's the actors name who played Rocket in Guardians of the Galaxy?"

    with trace("NZ: Collaborative Query Answering", metadata={"query": user_query}):
        # data_agents = create_data_agents(5)
        data_agents = create_data_agents_from_folder("../data/handmade")

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
        print(result)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

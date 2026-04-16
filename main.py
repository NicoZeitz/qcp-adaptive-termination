import os
from agents import Agent, ModelSettings, trace
from sqlalchemy import create_engine
from modules.agent_context import AgentContext
from modules.data_agent import DataAgent
from modules.query_coordinator import QueryCoordinator
from modules.sql_data_agent import SqlDataAgent
from openai import AsyncOpenAI
from openai.types.shared import Reasoning
from agents import set_default_openai_client
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

# custom_client = AsyncOpenAI(base_url=os.getenv("BASE_URL"), api_key=os.getenv("API_KEY"))
custom_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
set_default_openai_client(custom_client)

def create_data_agents(amount: int) -> list[DataAgent]:
    data_agents: list[DataAgent] = []

    for i in range(1, amount + 1):
        data_agent = SqlDataAgent(
            agent=Agent[AgentContext](
                name=f"A{i}",
                model="gpt-5-nano"
                # model="gpt-4.1-nano-2025-04-14",
                # model_settings=ModelSettings(reasoning=Reasoning(effort="high"), verbosity="low")
            ),
            db_engine=create_engine(f'sqlite:///{os.path.abspath(f"./data/agent{i}.db")}'),
        )
        data_agents.append(data_agent)

    return data_agents

async def main():
    with trace("Collaborative Query Answering"):
        data_agents = create_data_agents(5)
        queryCoordinator = QueryCoordinator(
            agent=Agent[AgentContext](
                name="Query Coordinator",
                # https://developers.openai.com/api/docs/pricing
                model="gpt-5-mini",
                model_settings=ModelSettings(reasoning=Reasoning(effort="high"), verbosity="low")
            )
        )

        for data_agent in data_agents:
            queryCoordinator.register_data_agent(data_agent)

        result = await queryCoordinator.collaborative_query_answering("What's the actors name who played Rocket in Guardians of the Galaxy?")
        print(result)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
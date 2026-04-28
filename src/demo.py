import os

from sqlalchemy import create_engine, text
from agents import Agent, ModelSettings, trace, set_default_openai_client
from openai import AsyncOpenAI
from openai.types.shared import Reasoning
from dotenv import load_dotenv

from modules.cache import Cache
from modules.model import AgentContext
from modules.sql_data_agent import SqlDataAgent
from modules.text_data_agent import TextDataAgent

load_dotenv()

# custom_client = AsyncOpenAI(base_url=os.getenv("BASE_URL"), api_key=os.getenv("API_KEY"))
custom_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
set_default_openai_client(custom_client)


def create_data_agents_from_folder(folder_path: str) -> list[SqlDataAgent]:
    data_agents: list[SqlDataAgent] = []

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


async def main():
    with trace("NZ: Text Demo"), open("../data/text.txt") as f:
        data_agent = TextDataAgent(
            agent=Agent[AgentContext](
                name="T1",
                model="gpt-5-mini",
                model_settings=ModelSettings(
                    reasoning=Reasoning(effort="low"),
                    verbosity="low",
                    parallel_tool_calls=True,
                ),
            ),
            text=f.read(),
        )
        print("==============================================================")
        print(await data_agent.get_data_and_query_capabilities())
        print("==============================================================")
        print(
            await data_agent.get_schema_for_query(
                "Who was the director of Donald Trump's favorite movie?"
            )
        )
        print("==============================================================")
        print(
            await data_agent.answer_query(
                "Who was the director of Donald Trump's favorite movie?", []
            )
        )

    # with trace("NZ: Code Demo"):
    #     data_agents = create_data_agents_from_folder("../data/handmade")
    #     agent = [
    #         data_agent for data_agent in data_agents if data_agent.agent.name == "A1"
    #     ][0]  # people
    #     result = await agent.answer_query_code(
    #         # "What is the person id of Bradley Cooper", []
    #         "List the first 100 person ids when ordered by firstname and lastname",
    #         # "List all person ids",
    #         [],
    #     )
    #     print(result)


if __name__ == "__main__":
    import asyncio

    Cache.instance().disable()
    asyncio.run(main())

# Director Orson Welles https://www.imdb.com/title/tt0033467/
# Who was the director of Donald Trump's favorite movie?

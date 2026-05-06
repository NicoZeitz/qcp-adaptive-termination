from data_agent.data_agent import DataAgent
from utils.prompt import load_prompt
from utils.run_agent import run_agent


async def is_relevant(data_agent: DataAgent, query: str) -> bool:
    is_relevant = await run_agent(
        name_postfix="Relevance",
        agent_configuration=data_agent.agent_configuration,
        instructions=load_prompt(
            "SQL Data Agent/Relevance/Instructions",
            {"m_schema": await data_agent.data_description()},
        ),
        prompt=load_prompt(
            "SQL Data Agent/Relevance/Prompt",
            {"query": query},
        ),
        output_type=bool,
    )
    return is_relevant

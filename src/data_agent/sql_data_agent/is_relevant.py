from data_agent.data_agent import DataAgent
from utils.prompt import load_prompt
from utils.run_agent import run_agent


async def is_relevant(data_agent: DataAgent, query: str, reasoning: str) -> bool:
    is_relevant, _ = await run_agent(
        name_postfix="Relevance",
        agent_configuration=data_agent.agent_configuration,
        prompt_name="SQL Data Agent/Relevance",
        prompt_parameters={
            "m_schema": await data_agent.data_description(),
            "query": query,
            "reasoning": reasoning,
        },
        output_type=bool,
    )
    return is_relevant

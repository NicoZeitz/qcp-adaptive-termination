from data_agent.data_agent import DataAgent
from utils.prompt import load_prompt
from utils.preview import CHAR_LIMIT_LONG_LLM, preview_text
from utils.run_agent import run_agent


async def is_relevant(
    data_agent: DataAgent, text: str, query: str, reasoning: str
) -> bool:
    result, _ = await run_agent(
        name_postfix="Relevance",
        agent_configuration=data_agent.agent_configuration,
        prompt_name="Text Data Agent/Relevance",
        prompt_parameters={
            "summary": await data_agent.data_description(),
            "preview": preview_text(text, CHAR_LIMIT_LONG_LLM),
            "query": query,
            "reasoning": reasoning,
        },
        output_type=bool,
    )
    return result

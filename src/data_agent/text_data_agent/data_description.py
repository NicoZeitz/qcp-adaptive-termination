from async_lru import alru_cache

from data_agent.data_agent import DataAgent
from utils.prompt import load_prompt
from utils.preview import CHAR_LIMIT_LONG_LLM, preview_text
from utils.run_agent import run_agent


@alru_cache
async def data_description(data_agent: DataAgent, text: str) -> str:
    return await run_agent(
        name_postfix="Summary",
        agent_configuration=data_agent.agent_configuration,
        instructions=load_prompt("Text Data Agent/Summary/Instructions", {}),
        prompt=load_prompt(
            "Text Data Agent/Summary/Prompt",
            {"preview": preview_text(text, CHAR_LIMIT_LONG_LLM)},
        ),
    )

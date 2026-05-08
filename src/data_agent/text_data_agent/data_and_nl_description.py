from async_lru import alru_cache

from data_agent.data_agent import DataAgent
from utils.preview import CHAR_LIMIT_LONG_LLM, preview_text
from utils.run_agent import run_agent

CACHED_DESCRIPTION = {
    "T1": "A short entertainment article from MovieMaker Magazine listing Donald Trump’s five favorite films (including Citizen Kane, The Good, the Bad and the Ugly, Goodfellas, The Godfather, and Gone With the Wind) plus an action-movie “guilty pleasure,” with brief quotes and context. It also mentions Kamala Harris praising Black Panther and Joe Biden naming Chariots of Fire as his favorite. The piece is a light, list-style pop-culture write-up."
}


async def natural_language_description(data_agent: DataAgent, text: str) -> str:
    return await data_and_nl_description(data_agent, text)


async def data_description(data_agent: DataAgent, text: str) -> str:
    return await data_and_nl_description(data_agent, text)


@alru_cache
async def data_and_nl_description(data_agent: DataAgent, text: str) -> str:
    if data_agent.name in CACHED_DESCRIPTION:
        print("Using cached description for agent:", data_agent.name)
        return CACHED_DESCRIPTION[data_agent.name]  # TODO:REMOVE

    result, _ = await run_agent(
        name_postfix="Summary",
        agent_configuration=data_agent.agent_configuration,
        prompt_name="Text Data Agent/Summary",
        prompt_parameters={
            "preview": preview_text(text, CHAR_LIMIT_LONG_LLM),
        },
    )
    return result

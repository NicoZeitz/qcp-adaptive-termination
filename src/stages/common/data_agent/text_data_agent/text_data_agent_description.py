from stages.common.agent_configuration import get_data_agent_agent_configuration
from stages.common.data_agent.text_data_agent.get_text import get_text
from stages.utils.run_agent import run_agent
from stages.utils.preview import CHAR_LIMIT_LONG_LLM, preview_text
from qtypes.data_agent import DataAgent


async def text_data_agent_query_capabilities(data_agent: DataAgent) -> list[str]:
    return ["Natural Language Question Answering", "Reading Comprehension"]


async def text_data_and_nl_description(data_agent: DataAgent) -> tuple[str, str]:
    text = get_text(data_agent)
    result = await run_agent(
        name_postfix="Summary",
        agent_configuration=get_data_agent_agent_configuration(data_agent),
        prompt_name="Text Data Agent/Summary",
        prompt_parameters={
            "preview": preview_text(text, CHAR_LIMIT_LONG_LLM),
        },
    )
    return result, result

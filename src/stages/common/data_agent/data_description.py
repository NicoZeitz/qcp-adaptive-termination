from qtypes.data_agent import DataAgent
from stages.common.data_agent.sql_data_agent.sql_data_agent_description import (
    sql_data_agent_query_capabilities,
    sql_data_and_nl_description,
)

mapping = {
    "sql": {
        "query_capabilities": sql_data_agent_query_capabilities,
        "data_and_nl_description": sql_data_and_nl_description,
    },
}


async def data_agent_query_capabilities(data_agent: DataAgent) -> list[str]:
    if data_agent.type not in mapping:
        raise ValueError(f"Unsupported data agent type: {data_agent.type}")

    return await mapping[data_agent.type]["query_capabilities"](data_agent)


async def data_agent_natural_language_description(data_agent: DataAgent) -> str:
    return (await data_and_nl_description(data_agent))[1]


async def data_agent_data_description(data_agent: DataAgent) -> str:
    return (await data_and_nl_description(data_agent))[0]


async def data_and_nl_description(data_agent: DataAgent) -> tuple[str, str]:
    if data_agent.type not in mapping:
        raise ValueError(f"Unsupported data agent type: {data_agent.type}")

    return await mapping[data_agent.type]["data_and_nl_description"](data_agent)

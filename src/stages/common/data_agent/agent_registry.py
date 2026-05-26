import asyncio
from enum import Enum

from qtypes.data_agent import DataAgent
from stages.common.data_agent.data_description import (
    data_agent_data_description,
    data_agent_natural_language_description,
    data_agent_query_capabilities,
)


class AgentRegistryType(Enum):
    NATURAL_LANGUAGE = "natural_language"
    TECHNICAL = "technical"


async def return_agent_info(
    data_agent: DataAgent,
) -> tuple[str, str, list[str]]:
    return data_agent.name, *await asyncio.gather(
        data_agent_data_description(data_agent),
        data_agent_query_capabilities(data_agent),
    )


async def return_nl_agent_info(
    data_agent: DataAgent,
) -> tuple[str, str, list[str]]:
    return data_agent.name, *await asyncio.gather(
        data_agent_natural_language_description(data_agent),
        data_agent_query_capabilities(data_agent),
    )


async def build_agent_registry(
    data_agents: list[DataAgent], agent_registry_type: AgentRegistryType
) -> str:
    if agent_registry_type == AgentRegistryType.NATURAL_LANGUAGE:
        return await build_nl_agent_registry(data_agents)
    elif agent_registry_type == AgentRegistryType.TECHNICAL:
        return await build_m_schema_agent_registry(data_agents)
    else:
        raise ValueError(f"Unsupported agent registry type: {agent_registry_type}")


async def build_nl_agent_registry(data_agents: list[DataAgent]) -> str:
    agent_infos = await asyncio.gather(
        *(return_nl_agent_info(agent) for agent in data_agents)
    )

    return "\n\n".join(
        f"【Agent Name】 {name}\n{description}\n【Capabilities】 {', '.join(capabilities)}"
        for name, description, capabilities in agent_infos
    )


async def build_m_schema_agent_registry(data_agents: list[DataAgent]) -> str:
    agent_infos = await asyncio.gather(
        *(return_agent_info(agent) for agent in data_agents)
    )

    return "\n\n".join(
        f"【Agent Name】 {name}\n{description}\n【Capabilities】 {', '.join(capabilities)}"
        for name, description, capabilities in agent_infos
    )

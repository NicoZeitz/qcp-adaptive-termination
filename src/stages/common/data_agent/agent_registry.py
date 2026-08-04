import asyncio
from enum import Enum
from typing import Callable, overload

from qtypes.agent_data import AgentData
from qtypes.data_agent import DataAgent
from stages.common.data_agent.sql_data_agent.sql_data_agent_description import (
    sql_load_agent_data,
)


class AgentRegistryType(Enum):
    NATURAL_LANGUAGE = "natural_language"
    TECHNICAL = "technical"


async def load_agent_data(data_agent: DataAgent) -> AgentData:
    if data_agent.type == "sql":
        return await sql_load_agent_data(data_agent)
    raise ValueError(f"Unsupported data agent type: {data_agent.type}")


@overload
def map_entries[T](
    data_agents: list[AgentData], mapping: Callable[[AgentData], T]
) -> list[T]: ...


@overload
def map_entries[T](data_agents: AgentData, mapping: Callable[[AgentData], T]) -> T: ...


def map_entries(data_agents, mapping):
    if isinstance(data_agents, list):
        return [mapping(agent) for agent in data_agents]
    return mapping(data_agents)


class AgentRegistry:
    @staticmethod
    @overload
    async def load(agent: DataAgent) -> AgentData: ...

    @staticmethod
    @overload
    async def load(agent: list[DataAgent]) -> list[AgentData]: ...

    @staticmethod
    async def load(agent):
        if isinstance(agent, list):
            return list(await asyncio.gather(*[load_agent_data(a) for a in agent]))
        return await load_agent_data(agent)

    @staticmethod
    @overload
    async def natural_language_info(agent: DataAgent) -> str: ...

    @staticmethod
    @overload
    async def natural_language_info(agent: list[DataAgent]) -> list[str]: ...

    @staticmethod
    async def natural_language_info(agent):
        return map_entries(
            await AgentRegistry.load(agent), lambda e: e.natural_language_description
        )

    @staticmethod
    @overload
    async def query_capabilities(agent: DataAgent) -> list[str]: ...

    @staticmethod
    @overload
    async def query_capabilities(agent: list[DataAgent]) -> list[list[str]]: ...

    @staticmethod
    async def query_capabilities(agent):
        return map_entries(
            await AgentRegistry.load(agent), lambda e: e.query_capabilities
        )

    @staticmethod
    @overload
    async def technical_description(agent: DataAgent) -> str: ...

    @staticmethod
    @overload
    async def technical_description(agent: list[DataAgent]) -> list[str]: ...

    @staticmethod
    async def technical_description(agent):
        return map_entries(
            await AgentRegistry.load(agent), lambda e: e.technical_description
        )

    @staticmethod
    @overload
    async def embedding_descriptions(agent: DataAgent) -> list[str]: ...

    @staticmethod
    @overload
    async def embedding_descriptions(agent: list[DataAgent]) -> list[list[str]]: ...

    @staticmethod
    async def embedding_descriptions(agent):
        return map_entries(
            await AgentRegistry.load(agent), lambda e: e.embedding_descriptions
        )

    @staticmethod
    @overload
    async def build_registry(
        agent: DataAgent,
        registry_type: AgentRegistryType,
        include_capabilities: bool = True,
    ) -> str: ...

    @staticmethod
    @overload
    async def build_registry(
        agent: list[DataAgent],
        registry_type: AgentRegistryType,
        include_capabilities: bool = True,
    ) -> list[str]: ...

    @staticmethod
    async def build_registry(
        agent, registry_type: AgentRegistryType, include_capabilities: bool = True
    ):
        def format_entry(a: DataAgent, entry: AgentData) -> str:
            desc = (
                entry.natural_language_description
                if registry_type == AgentRegistryType.NATURAL_LANGUAGE
                else entry.technical_description
            )
            text = f"【Agent Name】 {a.name}\n{desc}"
            if include_capabilities:
                text += f"\n【Capabilities】 {', '.join(entry.query_capabilities)}"
            return text

        entries = await AgentRegistry.load(agent)
        if isinstance(entries, list):
            return [format_entry(a, e) for a, e in zip(agent, entries)]
        return format_entry(agent, entries)

    @staticmethod
    async def build_registry_string(
        agent: DataAgent | list[DataAgent],
        registry_type: AgentRegistryType,
        include_capabilities: bool = True,
    ) -> str:
        texts = await AgentRegistry.build_registry(
            agent, registry_type, include_capabilities
        )
        if isinstance(texts, list):
            return "\n\n".join(texts)
        return texts

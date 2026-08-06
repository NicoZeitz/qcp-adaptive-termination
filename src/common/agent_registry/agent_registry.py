import asyncio
from enum import Enum
from typing import Callable, overload

from common.data_agent.data_agent import load_agent_registry_data
from common.embedding.embeddings import create_embedding
from model.agent.agent_registry_data import AgentRegistryData
from model.agent.data_agent import DataAgent


class AgentRegistryType(Enum):
    NATURAL_LANGUAGE = "natural_language"
    TECHNICAL = "technical"


@overload
def map_entries[T](
    data_agents: list[AgentRegistryData], mapping: Callable[[AgentRegistryData], T]
) -> list[T]: ...
@overload
def map_entries[T](
    data_agents: AgentRegistryData, mapping: Callable[[AgentRegistryData], T]
) -> T: ...
def map_entries(data_agents, mapping):
    if isinstance(data_agents, list):
        return [mapping(agent) for agent in data_agents]
    return mapping(data_agents)


class AgentRegistry:
    @staticmethod
    @overload
    async def load(agent: DataAgent) -> AgentRegistryData: ...
    @staticmethod
    @overload
    async def load(agent: list[DataAgent]) -> dict[str, AgentRegistryData]: ...
    @staticmethod
    async def load(agent):
        if isinstance(agent, list):
            results = await asyncio.gather(
                *[load_agent_registry_data(a) for a in agent]
            )
            return {a.name: r for a, r in zip(agent, results)}

        return await load_agent_registry_data(agent)

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
        def format_entry(a: DataAgent, entry: AgentRegistryData) -> str:
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
        if isinstance(agent, list):
            return [format_entry(a, entries[a.name]) for a in agent]
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

    @staticmethod
    @overload
    async def embedding_vector_for_column(
        agents: DataAgent,
        table_per_agent: str,
        column_per_agent: str,
    ) -> list[float]: ...
    @staticmethod
    @overload
    async def embedding_vector_for_column(
        agents: list[DataAgent],
        table_per_agent: dict[str, str],
        column_per_agent: dict[str, str],
    ) -> dict[str, list[float]]: ...
    @staticmethod
    async def embedding_vector_for_column(agents, table_per_agent, column_per_agent):
        single = isinstance(agents, DataAgent)
        agent_list = [agents] if single else agents
        table_map = {agents.name: table_per_agent} if single else table_per_agent
        column_map = {agents.name: column_per_agent} if single else column_per_agent

        agent_data = await AgentRegistry.load(agent_list)

        embedding_descriptions: list[str] = []
        for agent in agent_list:
            table = table_map[agent.name]
            column = column_map[agent.name]
            data = agent_data[agent.name]

            embedding_descriptions.append(
                data.individual_descriptions[table]
                .columns[column]
                .technical_description
            )

        embeddings = await create_embedding(embedding_descriptions)

        if single:
            return embeddings[0]

        return {agent.name: embeddings[i] for i, agent in enumerate(agent_list)}

    @staticmethod
    @overload
    async def embedding_vectors_for_all_columns(
        agents: DataAgent,
        table_per_agent: str,
    ) -> dict[str, list[float]]: ...
    @staticmethod
    @overload
    async def embedding_vectors_for_all_columns(
        agents: list[DataAgent],
        table_per_agent: dict[str, str],
    ) -> dict[str, dict[str, list[float]]]: ...
    @staticmethod
    async def embedding_vectors_for_all_columns(agents, table_per_agent):
        single = isinstance(agents, DataAgent)
        agent_list = [agents] if single else agents
        table_map = {agents.name: table_per_agent} if single else table_per_agent

        agent_data = await AgentRegistry.load(agent_list)

        result: dict[str, dict[str, list[float]]] = {}
        for agent in agent_list:
            table = table_map[agent.name]
            data = agent_data[agent.name]
            column_names = list(data.individual_descriptions[table].columns.keys())

            embedding_descriptions: list[str] = [
                data.individual_descriptions[table].columns[col].technical_description
                for col in column_names
            ]

            embeddings = await create_embedding(embedding_descriptions)

            result[agent.name] = {
                col: embeddings[i] for i, col in enumerate(column_names)
            }

        if single:
            return result[agents.name]

        return result

    @staticmethod
    @overload
    async def embedding_vector_for_table(
        agents: DataAgent,
        table_per_agent: str,
    ) -> list[float]: ...
    @staticmethod
    @overload
    async def embedding_vector_for_table(
        agents: list[DataAgent],
        table_per_agent: dict[str, str],
    ) -> dict[str, list[float]]: ...
    @staticmethod
    async def embedding_vector_for_table(agents, table_per_agent):
        single = isinstance(agents, DataAgent)
        agent_list = [agents] if single else agents
        table_map = {agents.name: table_per_agent} if single else table_per_agent

        agent_data = await AgentRegistry.load(agent_list)

        embedding_descriptions: list[str] = []
        for agent in agents:
            table = table_map[agent.name]
            data = agent_data[agent.name]

            embedding_descriptions.append(
                data.individual_descriptions[table].technical_description
            )

        embeddings = await create_embedding(embedding_descriptions)

        if single:
            return embeddings[0]

        return {agent.name: embeddings[i] for i, agent in enumerate(agent_list)}

    @staticmethod
    @overload
    async def embedding_vectors_for_all_tables(
        agents: DataAgent,
    ) -> dict[str, list[float]]: ...
    @staticmethod
    @overload
    async def embedding_vectors_for_all_tables(
        agents: list[DataAgent],
    ) -> dict[str, dict[str, list[float]]]: ...
    @staticmethod
    async def embedding_vectors_for_all_tables(agents):
        single = isinstance(agents, DataAgent)
        agent_list = [agents] if single else agents

        agent_data = await AgentRegistry.load(agent_list)

        result: dict[str, dict[str, list[float]]] = {}
        for agent in agent_list:
            data = agent_data[agent.name]
            table_names = list(data.individual_descriptions.keys())

            embedding_descriptions: list[str] = [
                data.individual_descriptions[table].technical_description
                for table in table_names
            ]

            embeddings = await create_embedding(embedding_descriptions)

            result[agent.name] = {
                table: embeddings[i] for i, table in enumerate(table_names)
            }

        if single:
            return result[agents.name]

        return result

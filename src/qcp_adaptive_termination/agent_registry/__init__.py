"""
Agent registry utilities.

Provides a registry of available agents.
"""

import asyncio
from typing import overload

import numpy as np

import qcp_adaptive_termination.agent_registry.utils as utils
import qcp_adaptive_termination.embedding as embedding
from qcp_adaptive_termination.models import (
    AgentRegistryData,
    AgentRegistryType,
    DataAgent,
)


@overload
async def load(agent: DataAgent) -> AgentRegistryData: ...
@overload
async def load(agent: list[DataAgent]) -> dict[str, AgentRegistryData]: ...
async def load(agent):
    if isinstance(agent, list):
        results = await asyncio.gather(*[utils.load_registry_data(a) for a in agent])
        return {a.name: r for a, r in zip(agent, results, strict=True)}
    return await utils.load_registry_data(agent)


@overload
async def natural_language_info(agent: DataAgent) -> str: ...
@overload
async def natural_language_info(agent: list[DataAgent]) -> list[str]: ...
async def natural_language_info(agent):
    return utils.map_entries(await load(agent), lambda e: e.natural_language_description)


@overload
async def query_capabilities(agent: DataAgent) -> list[str]: ...
@overload
async def query_capabilities(agent: list[DataAgent]) -> list[list[str]]: ...
async def query_capabilities(agent):
    return utils.map_entries(await load(agent), lambda e: e.query_capabilities)


@overload
async def technical_description(agent: DataAgent) -> str: ...
@overload
async def technical_description(agent: list[DataAgent]) -> list[str]: ...
async def technical_description(agent):
    return utils.map_entries(await load(agent), lambda e: e.technical_description)


@overload
async def build_registry(
    agent: DataAgent,
    registry_type: AgentRegistryType,
    include_capabilities: bool = True,
) -> str: ...
@overload
async def build_registry(
    agent: list[DataAgent],
    registry_type: AgentRegistryType,
    include_capabilities: bool = True,
) -> list[str]: ...
async def build_registry(agent, registry_type: AgentRegistryType, include_capabilities: bool = True):
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

    entries = await load(agent)
    if isinstance(agent, list):
        return [format_entry(a, entries[a.name]) for a in agent]
    return format_entry(agent, entries)


async def build_registry_string(
    agent: DataAgent | list[DataAgent],
    registry_type: AgentRegistryType,
    include_capabilities: bool = True,
) -> str:
    texts = await build_registry(agent, registry_type, include_capabilities)
    if isinstance(texts, list):
        return "\n\n".join(texts)
    return texts


@overload
async def embedding_vector_for_column(
    agents: DataAgent,
    table_per_agent: str,
    column_per_agent: str,
) -> np.ndarray: ...
@overload
async def embedding_vector_for_column(
    agents: list[DataAgent],
    table_per_agent: dict[str, str],
    column_per_agent: dict[str, str],
) -> dict[str, np.ndarray]: ...
async def embedding_vector_for_column(agents, table_per_agent, column_per_agent):
    single = isinstance(agents, DataAgent)
    agent_list = [agents] if single else agents
    table_map = {agents.name: table_per_agent} if single else table_per_agent
    column_map = {agents.name: column_per_agent} if single else column_per_agent

    agent_data = await load(agent_list)

    embedding_descriptions: list[str] = []
    for agent in agent_list:
        table = table_map[agent.name]
        column = column_map[agent.name]
        data = agent_data[agent.name]
        embedding_descriptions.append(data.individual_descriptions[table].columns[column].technical_description)

    embeddings = await embedding.create(embedding_descriptions)

    if single:
        return embeddings[0]
    return {agent.name: embeddings[i] for i, agent in enumerate(agent_list)}


@overload
async def embedding_vectors_for_all_columns(
    agents: DataAgent,
    table_per_agent: str,
) -> dict[str, np.ndarray]: ...
@overload
async def embedding_vectors_for_all_columns(
    agents: list[DataAgent],
    table_per_agent: dict[str, str],
) -> dict[str, dict[str, np.ndarray]]: ...
async def embedding_vectors_for_all_columns(agents, table_per_agent):
    single = isinstance(agents, DataAgent)
    agent_list = [agents] if single else agents
    table_map = {agents.name: table_per_agent} if single else table_per_agent

    agent_data = await load(agent_list)

    all_descriptions: list[str] = []
    offsets: list[tuple[str, list[str]]] = []
    for agent in agent_list:
        table = table_map[agent.name]
        data = agent_data[agent.name]
        column_names = list(data.individual_descriptions[table].columns.keys())
        all_descriptions.extend(
            data.individual_descriptions[table].columns[col].technical_description for col in column_names
        )
        offsets.append((agent.name, column_names))

    all_embeddings = await embedding.create(all_descriptions)

    result: dict[str, dict[str, np.ndarray]] = {}
    pos = 0
    for agent_name, column_names in offsets:
        n = len(column_names)
        result[agent_name] = {col: all_embeddings[pos + i] for i, col in enumerate(column_names)}
        pos += n

    if single:
        return result[agents.name]
    return result


@overload
async def embedding_vector_for_table(
    agents: DataAgent,
    table_per_agent: str,
    registry_type: AgentRegistryType = AgentRegistryType.TECHNICAL,
) -> np.ndarray: ...
@overload
async def embedding_vector_for_table(
    agents: list[DataAgent],
    table_per_agent: dict[str, str],
    registry_type: AgentRegistryType = AgentRegistryType.TECHNICAL,
) -> dict[str, np.ndarray]: ...
async def embedding_vector_for_table(
    agents, table_per_agent, registry_type: AgentRegistryType = AgentRegistryType.TECHNICAL
):
    single = isinstance(agents, DataAgent)
    agent_list = [agents] if single else agents
    table_map = {agents.name: table_per_agent} if single else table_per_agent

    agent_data = await load(agent_list)

    embedding_descriptions: list[str] = []
    for agent in agent_list:
        table = table_map[agent.name]
        data = agent_data[agent.name]
        embedding_descriptions.append(
            data.individual_descriptions[table].natural_language_description
            if registry_type == AgentRegistryType.NATURAL_LANGUAGE
            else data.individual_descriptions[table].technical_description
        )

    embeddings = await embedding.create(embedding_descriptions)

    if single:
        return embeddings[0]
    return {agent.name: embeddings[i] for i, agent in enumerate(agent_list)}


@overload
async def embedding_vectors_for_all_tables(
    agents: DataAgent,
    registry_type: AgentRegistryType = AgentRegistryType.TECHNICAL,
) -> dict[str, np.ndarray]: ...
@overload
async def embedding_vectors_for_all_tables(
    agents: list[DataAgent],
    registry_type: AgentRegistryType = AgentRegistryType.TECHNICAL,
) -> dict[str, dict[str, np.ndarray]]: ...
async def embedding_vectors_for_all_tables(agents, registry_type: AgentRegistryType = AgentRegistryType.TECHNICAL):
    single = isinstance(agents, DataAgent)
    agent_list = [agents] if single else agents

    agent_data = await load(agent_list)

    all_descriptions: list[str] = []
    offsets: list[tuple[str, list[str]]] = []
    for agent in agent_list:
        data = agent_data[agent.name]
        table_names = list(data.individual_descriptions.keys())
        all_descriptions.extend(
            data.individual_descriptions[table].natural_language_description
            if registry_type == AgentRegistryType.NATURAL_LANGUAGE
            else data.individual_descriptions[table].technical_description
            for table in table_names
        )
        offsets.append((agent.name, table_names))

    all_embeddings = await embedding.create(all_descriptions)

    result: dict[str, dict[str, np.ndarray]] = {}
    pos = 0
    for agent_name, table_names in offsets:
        n = len(table_names)
        result[agent_name] = {table: all_embeddings[pos + i] for i, table in enumerate(table_names)}
        pos += n

    if single:
        return result[agents.name]
    return result


__all__ = [
    "build_registry",
    "build_registry_string",
    "embedding_vector_for_column",
    "embedding_vector_for_table",
    "embedding_vectors_for_all_columns",
    "embedding_vectors_for_all_tables",
    "load",
    "natural_language_info",
    "query_capabilities",
    "technical_description",
]

from model.agent.data_agent import DataAgent
from common.agent_registry.agent_registry import AgentRegistry
from common.embedding.embeddings import create_embedding
from common.embedding.vector_search import build_index, search_index_top_k


async def resolve_table(agents: list[DataAgent], query: str) -> dict[str, str]:
    agent_data = await AgentRegistry.load(agents)
    query_vec = await create_embedding(query)

    result: dict[str, str] = {}
    for agent, data in zip(agents, agent_data):
        table_names = list(data.table_column_names.keys())
        descriptions = data.embedding_descriptions
        if not table_names:
            continue
        vectors: list[list[float]] = await create_embedding(descriptions)
        index = build_index(list(zip(table_names, vectors)))
        ranked = search_index_top_k(index, query_vec, k=1)
        result[agent.name] = ranked[0][0]

    return result


async def resolve_column(
    agents: list[DataAgent],
    table_per_agent: dict[str, str],
    query: str,
) -> dict[str, str]:
    agent_data = await AgentRegistry.load(agents)
    query_vec = await create_embedding(query)

    result: dict[str, str] = {}
    for agent, data in zip(agents, agent_data):
        table_name = table_per_agent.get(agent.name)
        if not table_name:
            continue
        col_map = data.table_column_names.get(table_name, {})
        if not col_map:
            raise ValueError(
                f"No columns found for table '{table_name}' on agent '{agent.name}'"
            )
        col_names = list(col_map.keys())
        texts = [f"{col}: {desc}" if desc else col for col, desc in col_map.items()]
        vectors: list[list[float]] = await create_embedding(texts)
        index = build_index(list(zip(col_names, vectors)))
        ranked = search_index_top_k(index, query_vec, k=1)
        result[agent.name] = ranked[0][0]

    return result

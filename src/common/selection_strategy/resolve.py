from model.agent.data_agent import DataAgent
from common.agent_registry.agent_registry import AgentRegistry
from common.embedding.embeddings import create_embedding
from common.embedding.vector_search import build_index, search_index_top_k


async def resolve_table(agents: list[DataAgent], query: str) -> dict[str, str]:
    agents_vectors = await AgentRegistry.embedding_vectors_for_all_tables(agents)
    query_vector = await create_embedding(query)

    # for each agent find the table with the closest embedding to the query
    result: dict[str, str] = {}
    for agent, table_vectors in agents_vectors.items():
        index = build_index(list(table_vectors.items()))
        ranked = search_index_top_k(index, query_vector, k=1)
        result[agent] = ranked[0][0]

    return result


async def resolve_column(
    agents: list[DataAgent],
    table_per_agent: dict[str, str],
    query: str,
) -> dict[str, str]:
    agents_vectors = await AgentRegistry.embedding_vectors_for_all_columns(
        agents, table_per_agent
    )
    query_vector = await create_embedding(query)

    result: dict[str, str] = {}
    for agent, column_vectors in agents_vectors.items():
        index = build_index(list(column_vectors.items()))
        ranked = search_index_top_k(index, query_vector, k=1)
        result[agent] = ranked[0][0]

    return result

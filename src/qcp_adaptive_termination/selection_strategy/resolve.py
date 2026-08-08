import qcp_adaptive_termination.agent_registry as agent_registry
import qcp_adaptive_termination.embedding as embedding
from qcp_adaptive_termination.embedding import VectorIndex
from qcp_adaptive_termination.models import DataAgent


async def resolve_table(agents: list[DataAgent], query: str) -> dict[str, str]:
    agents_vectors = await agent_registry.embedding_vectors_for_all_tables(agents)
    query_vector = await embedding.create(query)

    # for each agent find the table with the closest embedding to the query
    result: dict[str, str] = {}
    for agent, table_vectors in agents_vectors.items():
        vector_index = VectorIndex.build(list(table_vectors.items()))
        ranked = vector_index.search_top_k(query_vector, k=1)
        result[agent] = ranked[0][0]

    return result


async def resolve_column(
    agents: list[DataAgent],
    table_per_agent: dict[str, str],
    query: str,
) -> dict[str, str]:
    agents_vectors = await agent_registry.embedding_vectors_for_all_columns(
        agents, table_per_agent
    )
    query_vector = await embedding.create(query)

    result: dict[str, str] = {}
    for agent, column_vectors in agents_vectors.items():
        vector_index = VectorIndex.build(list(column_vectors.items()))
        ranked = vector_index.search_top_k(query_vector, k=1)
        result[agent] = ranked[0][0]

    return result

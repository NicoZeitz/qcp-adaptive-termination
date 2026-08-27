"""Embedding-based pre-filter for data agent selection."""

import logging

import qcp_adaptive_termination.agent_registry as agent_registry
import qcp_adaptive_termination.embedding as embedding
import qcp_adaptive_termination.hooks as hooks
from qcp_adaptive_termination.embedding import VectorIndex
from qcp_adaptive_termination.hooks import HookEvent
from qcp_adaptive_termination.models import AgentRegistryType, DataAgent
from qcp_adaptive_termination.observability import tracing


class QueryDecomposition(BaseModel):
    topics: list[str]


async def decompose_query(query: str) -> list[str]:
    result = await run_llm(
        AgentRole.QUERY_DECOMPOSER,
        {"query": query},
        output_type=QueryDecomposition,
    )
    return result.topics


def find_l2_clusters(scores: list[float], min_cluster_size: int = 1) -> int:
    """Find cutoff by detecting the largest gap in ascending L2 distance scores."""
    if len(scores) <= min_cluster_size:
        return len(scores)

    gaps = [(scores[i + 1] - scores[i], i + 1) for i in range(len(scores) - 1)]

    if not gaps:
        return len(scores)

    max_gap, cutoff_idx = max(gaps, key=lambda x: x[0])

    if cutoff_idx < min_cluster_size:
        logging.info(
            "Largest gap at position %d is below min_cluster_size, using min_cluster_size=%d",
            cutoff_idx,
            min_cluster_size,
        )
        return min_cluster_size

    logging.info(
        "Largest gap of %.4f at position %d (L2: %.3f to %.3f)",
        max_gap,
        cutoff_idx,
        scores[cutoff_idx - 1],
        scores[cutoff_idx],
    )
    return cutoff_idx


async def _filter_embeddings_internal(
    query: str,
    data_agents: list[DataAgent],
    registry_type: AgentRegistryType,
) -> list[DataAgent]:
    table_vectors: dict[str, dict[str, list[float]]] = await agent_registry.embedding_vectors_for_all_tables(
        data_agents, registry_type
    )

    index_data: list[tuple[tuple[str, str], list[float]]] = [
        ((agent.name, table_name), vector)
        for agent in data_agents
        for table_name, vector in table_vectors.get(agent.name, {}).items()
    ]

    if not index_data:
        logging.warning("No table embeddings found, returning all agents unfiltered")
        return data_agents

    vector_index: VectorIndex[tuple[str, str]] = VectorIndex.build(index_data)

    needs = await decompose_query(query)
    logging.info("Query decomposed into %d topic(s): %s", len(needs), needs)

    all_query_texts = [query] + needs
    all_embeddings = await embedding.create(all_query_texts)

    agent_best_dist: dict[str, tuple[float, str]] = {}
    for (agent_name, table_name), dist in results:
        if agent_name not in agent_best_dist or dist < agent_best_dist[agent_name][0]:
            agent_best_dist[agent_name] = (dist, table_name)

    sorted_agents = sorted(agent_best_dist.items(), key=lambda x: x[1][0])

    logging.info("Embedding similarity per agent (L2 distance, ascending):")
    for agent_name, (dist, table_name) in sorted_agents:
        logging.info("  %s: L2=%.4f (table: %s)", agent_name, dist, table_name)

    l2_scores = [dist for _, (dist, _) in sorted_agents]
    cutoff_idx = find_l2_clusters(l2_scores + [2.0], min_cluster_size=1)
    sorted_agents = sorted_agents[:cutoff_idx]

    logging.info("Embedding pre-filter: kept %d / %d agents", len(sorted_agents), len(data_agents))

    agent_map = {a.name: a for a in data_agents}
    filtered = [agent_map[name] for name, _ in sorted_agents if name in agent_map]

    tracing.append_to_span_output(
        {
            "filtered_agents": [a.name for a in filtered],
            "l2_scores": {name: round(dist, 4) for name, (dist, _) in agent_best_dist.items()},
            "best_tables": {name: table for name, (_, table) in agent_best_dist.items()},
        }
    )

    hooks.emit(
        HookEvent.EMBEDDING_PRE_FILTER_END,
        query=query,
        topics=needs,
        embedding_filtered_agents=filtered,
        total_agents=len(data_agents),
    )

    return filtered


async def filter_embeddings(
    query: str,
    data_agents: list[DataAgent],
    registry_type: AgentRegistryType = AgentRegistryType.NATURAL_LANGUAGE,
) -> list[DataAgent]:
    return await hooks.call_override(
        HookEvent.EMBEDDING_PRE_FILTER_OVERRIDE,
        default=lambda: _filter_embeddings_internal(query, data_agents, registry_type),
        query=query,
        data_agents=data_agents,
    )

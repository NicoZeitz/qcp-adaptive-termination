"""Embedding-based pre-filter for data agent selection."""

import logging

import numpy as np

import qcp_adaptive_termination.agent_registry as agent_registry
import qcp_adaptive_termination.embedding as embedding
import qcp_adaptive_termination.hooks as hooks
from qcp_adaptive_termination.embedding import VectorIndex
from qcp_adaptive_termination.hooks import HookEvent
from qcp_adaptive_termination.models import AgentRegistryType, DataAgent
from qcp_adaptive_termination.observability import tracing


def find_similarity_clusters(scores: list[float], min_cluster_size: int = 1) -> int:
    """Find cutoff by detecting the largest gap in descending similarity scores."""
    if len(scores) <= min_cluster_size:
        return len(scores)

    gaps = [(scores[i] - scores[i + 1], i + 1) for i in range(len(scores) - 1)]

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
        "Largest gap of %.4f at position %d (score: %.3f to %.3f)",
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
    table_vectors: dict[str, dict[str, np.ndarray]] = await agent_registry.embedding_vectors_for_all_tables(
        data_agents, registry_type
    )

    index_data: list[tuple[tuple[str, str], np.ndarray]] = [
        ((agent.name, table_name), vector)
        for agent in data_agents
        for table_name, vector in table_vectors.get(agent.name, {}).items()
    ]

    if not index_data:
        logging.warning("No table embeddings found, returning all agents unfiltered")
        return data_agents

    vector_index: VectorIndex[tuple[str, str]] = VectorIndex.build(index_data)
    query_embedding = await embedding.create(query)
    results = vector_index.search_all(query_embedding)

    agent_best_dist: dict[str, tuple[float, str]] = {}
    for (agent_name, table_name), dist in results:
        if agent_name not in agent_best_dist or dist < agent_best_dist[agent_name][0]:
            agent_best_dist[agent_name] = (dist, table_name)

    sorted_agents = sorted(agent_best_dist.items(), key=lambda x: x[1][0])

    logging.info("Embedding similarity per agent (L2, ascending):")
    for agent_name, (dist, table_name) in sorted_agents:
        logging.info("  %s: L2=%.4f (table: %s)", agent_name, dist, table_name)

    similarity_scores = [1.0 / (1.0 + dist) for _, (dist, _) in sorted_agents]
    cutoff_idx = find_similarity_clusters(similarity_scores + [0.0], min_cluster_size=1)
    sorted_agents = sorted_agents[:cutoff_idx]

    logging.info("Embedding pre-filter: kept %d / %d agents", len(sorted_agents), len(data_agents))

    agent_map = {a.name: a for a in data_agents}
    filtered = [agent_map[name] for name, _ in sorted_agents if name in agent_map]

    tracing.append_to_span_output(
        {
            "filtered_agents": [a.name for a in filtered],
            "similarity_scores": {name: round(1.0 / (1.0 + dist), 4) for name, (dist, _) in agent_best_dist.items()},
            "best_tables": {name: table for name, (_, table) in agent_best_dist.items()},
        }
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

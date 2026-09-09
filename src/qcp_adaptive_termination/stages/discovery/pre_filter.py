"""Embedding-based pre-filter for data agent selection."""

import logging

import numpy as np
from sklearn.cluster import DBSCAN

import qcp_adaptive_termination.agent_registry as agent_registry
import qcp_adaptive_termination.embedding as embedding
import qcp_adaptive_termination.hooks as hooks
from qcp_adaptive_termination.embedding import VectorIndex
from qcp_adaptive_termination.hooks import HookEvent
from qcp_adaptive_termination.models import AgentRegistryType, DataAgent
from qcp_adaptive_termination.observability import tracing

# DBSCAN: neighborhood radius — smaller = tighter clusters, more noise points
DBSCAN_EPS = 0.5
# DBSCAN: minimum points to form a cluster core
DBSCAN_MIN_SAMPLES = 2


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


def find_knee(scores: list[float], min_cluster_size: int = 1) -> int:
    """Find cutoff using the Kneedle algorithm on ascending L2 distance scores.

    Normalizes the curve to [0,1]x[0,1], then finds the point of maximum
    distance from the diagonal — the knee of the curve.
    """
    n = len(scores)
    if n <= min_cluster_size:
        return n

    score_min, score_max = scores[0], scores[-1]
    score_range = score_max - score_min

    if score_range == 0:
        return min_cluster_size

    # normalized curve: x in [0,1], y in [0,1]
    xs = [i / (n - 1) for i in range(n)]
    ys = [(s - score_min) / score_range for s in scores]

    # distance of each point from the diagonal y=x
    distances = [ys[i] - xs[i] for i in range(n)]

    knee_idx = max(range(n), key=lambda i: distances[i])

    knee_idx = max(knee_idx, min_cluster_size)

    logging.info(
        "Knee at position %d (L2: %.3f, normalized distance from diagonal: %.4f)",
        knee_idx,
        scores[knee_idx],
        distances[knee_idx],
    )
    return knee_idx


def filter_by_cluster(
    agent_vectors: dict[str, list[float]],
    query_vector: list[float],
) -> list[str]:
    """Cluster table embeddings with DBSCAN, return agents from clusters closest to the query."""
    agent_names = list(agent_vectors.keys())
    matrix = np.array([agent_vectors[name] for name in agent_names], dtype=np.float32)
    query = np.array(query_vector, dtype=np.float32)

    labels = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, metric="euclidean", n_jobs=-1).fit_predict(matrix)

    # compute centroid per cluster and distance to query
    unique_labels = set(labels)
    cluster_dists: dict[int, float] = {}
    for label in unique_labels:
        if label == -1:
            continue  # noise points handled separately
        mask = labels == label
        centroid = matrix[mask].mean(axis=0)
        cluster_dists[label] = float(np.linalg.norm(centroid - query))

    if not cluster_dists:
        # all noise — fall back to returning all agents
        logging.warning("DBSCAN: all points are noise, returning all agents")
        return agent_names

    # find the closest cluster and its distance
    best_label, best_dist = min(cluster_dists.items(), key=lambda x: x[1])

    # keep all clusters whose centroid is within largest_gap of the best cluster
    cluster_dists_sorted = sorted(cluster_dists.values())
    gaps = [cluster_dists_sorted[i + 1] - cluster_dists_sorted[i] for i in range(len(cluster_dists_sorted) - 1)]
    cutoff_dist = cluster_dists_sorted[find_l2_clusters(cluster_dists_sorted)] if gaps else best_dist * 1.5

    relevant_labels = {label for label, dist in cluster_dists.items() if dist <= cutoff_dist}

    # noise points: include if closer to query than the cutoff
    noise_agents = [
        agent_names[i] for i, label in enumerate(labels)
        if label == -1 and float(np.linalg.norm(matrix[i] - query)) <= cutoff_dist
    ]

    kept = [agent_names[i] for i, label in enumerate(labels) if label in relevant_labels] + noise_agents

    logging.info(
        "DBSCAN: %d clusters, keeping %d relevant clusters, %d noise agents included, total %d / %d agents",
        len(cluster_dists),
        len(relevant_labels),
        len(noise_agents),
        len(kept),
        len(agent_names),
    )
    return kept


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

    [query_embedding] = await embedding.create([query])

    # best vector per agent (closest table to query)
    index_lookup: dict[tuple[str, str], list[float]] = {key: vec for key, vec in index_data}
    agent_best_vec: dict[str, list[float]] = {}
    agent_best_dist: dict[str, tuple[float, str]] = {}
    vector_index: VectorIndex[tuple[str, str]] = VectorIndex.build(index_data)
    for (agent_name, table_name), dist in vector_index.search_all(query_embedding):
        if agent_name not in agent_best_dist or dist < agent_best_dist[agent_name][0]:
            agent_best_dist[agent_name] = (dist, table_name)
            agent_best_vec[agent_name] = index_lookup[agent_name, table_name]

    kept_names = filter_by_cluster(agent_best_vec, query_embedding)

    agent_map = {a.name: a for a in data_agents}
    filtered = [agent_map[name] for name in kept_names if name in agent_map]

    logging.info("Embedding pre-filter: kept %d / %d agents", len(filtered), len(data_agents))

    tracing.append_to_span_output(
        {
            "filtered_agents": [a.name for a in filtered],
            "l2_scores": {name: round(float(dist), 4) for name, (dist, _) in agent_best_dist.items()},
            "best_tables": {name: table for name, (_, table) in agent_best_dist.items()},
        }
    )

    hooks.emit(
        HookEvent.EMBEDDING_PRE_FILTER_END,
        query=query,
        embedding_filtered_agents=filtered,
        total_agents=len(data_agents),
        l2_scores={name: round(float(dist), 4) for name, (dist, _) in agent_best_dist.items()},
    )

    return filtered


async def filter_embeddings(
    query: str,
    data_agents: list[DataAgent],
    registry_type: AgentRegistryType = AgentRegistryType.TECHNICAL,
) -> list[DataAgent]:
    return await hooks.call_override(
        HookEvent.EMBEDDING_PRE_FILTER_OVERRIDE,
        default=lambda: _filter_embeddings_internal(query, data_agents, registry_type),
        query=query,
        data_agents=data_agents,
    )


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


def find_knee(scores: list[float], min_cluster_size: int = 1) -> int:
    """Find cutoff using the Kneedle algorithm on ascending L2 distance scores.

    Normalizes the curve to [0,1]x[0,1], then finds the point of maximum
    distance from the diagonal — the knee of the curve.
    """
    n = len(scores)
    if n <= min_cluster_size:
        return n

    score_min, score_max = scores[0], scores[-1]
    score_range = score_max - score_min

    if score_range == 0:
        return min_cluster_size

    # normalized curve: x in [0,1], y in [0,1]
    xs = [i / (n - 1) for i in range(n)]
    ys = [(s - score_min) / score_range for s in scores]

    # distance of each point from the diagonal y=x
    distances = [ys[i] - xs[i] for i in range(n)]

    knee_idx = max(range(n), key=lambda i: distances[i])

    knee_idx = max(knee_idx, min_cluster_size)

    logging.info(
        "Knee at position %d (L2: %.3f, normalized distance from diagonal: %.4f)",
        knee_idx,
        scores[knee_idx],
        distances[knee_idx],
    )
    return knee_idx


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

    [query_embedding] = await embedding.create([query])

    agent_best_dist: dict[str, tuple[float, str]] = {}
    for (agent_name, table_name), dist in vector_index.search_all(query_embedding):
        if agent_name not in agent_best_dist or dist < agent_best_dist[agent_name][0]:
            agent_best_dist[agent_name] = (dist, table_name)

    sorted_agents = sorted(agent_best_dist.items(), key=lambda x: x[1][0])

    logging.info("Embedding similarity per agent (L2 distance, ascending):")
    for agent_name, (dist, table_name) in sorted_agents:
        logging.info("  %s: L2=%.4f (table: %s)", agent_name, dist, table_name)

    l2_scores = [dist for _, (dist, _) in sorted_agents]
    cutoff_idx = find_knee(l2_scores, min_cluster_size=1)
    sorted_agents = sorted_agents[:cutoff_idx]

    logging.info("Embedding pre-filter: kept %d / %d agents", len(sorted_agents), len(data_agents))

    agent_map = {a.name: a for a in data_agents}
    filtered = [agent_map[name] for name, _ in sorted_agents if name in agent_map]

    tracing.append_to_span_output(
        {
            "filtered_agents": [a.name for a in filtered],
            "l2_scores": {name: round(float(dist), 4) for name, (dist, _) in agent_best_dist.items()},
            "best_tables": {name: table for name, (_, table) in agent_best_dist.items()},
        }
    )

    hooks.emit(
        HookEvent.EMBEDDING_PRE_FILTER_END,
        query=query,
        embedding_filtered_agents=filtered,
        total_agents=len(data_agents),
        l2_scores={name: round(float(dist), 4) for name, (dist, _) in agent_best_dist.items()},
    )

    return filtered


async def filter_embeddings(
    query: str,
    data_agents: list[DataAgent],
    registry_type: AgentRegistryType = AgentRegistryType.TECHNICAL,
) -> list[DataAgent]:
    return await hooks.call_override(
        HookEvent.EMBEDDING_PRE_FILTER_OVERRIDE,
        default=lambda: _filter_embeddings_internal(query, data_agents, registry_type),
        query=query,
        data_agents=data_agents,
    )

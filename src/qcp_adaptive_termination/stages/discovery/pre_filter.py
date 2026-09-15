"""Embedding-based pre-filter for data agent selection."""

import logging
import resource

import qcp_adaptive_termination.agent_registry as agent_registry
import qcp_adaptive_termination.embedding as embedding
import qcp_adaptive_termination.hooks as hooks
from qcp_adaptive_termination.embedding import VectorIndex
from qcp_adaptive_termination.hooks import HookEvent
from qcp_adaptive_termination.models import AgentRegistryType, DataAgent
from qcp_adaptive_termination.observability import tracing
from qcp_adaptive_termination.observability.context import get_output_dir
from qcp_adaptive_termination.stages.discovery.cluster_index import (
    filter_agents_by_cluster,
    get_or_build_cluster_index,
    render_query_visualization,
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


async def _filter_embeddings_internal(
    query: str,
    data_agents: list[DataAgent],
    registry_type: AgentRegistryType,
) -> list[DataAgent]:
    # --- Stage 1: cluster-based pre-filter ---
    # cluster_index = await get_or_build_cluster_index(data_agents, registry_type)
    [query_embedding] = await embedding.create([query])
    # agents_before = data_agents
    # data_agents, selected_clusters = filter_agents_by_cluster(query_embedding, cluster_index, data_agents, find_l2_clusters)
    # logging.info("Cluster pre-filter: %d / %d agents remaining", len(data_agents), len(agents_before))

    # render per-query visualization into the run output dir
    # output_dir = get_output_dir()
    # if output_dir is not None:
    #     query_slug = query[:60].replace(" ", "_").replace("/", "-")
    #     viz_path = output_dir / f"cluster_filter_{query_slug}.png"
    #     try:
    #         render_query_visualization(query, query_embedding, cluster_index, selected_clusters, viz_path)
    #     except Exception as exc:
    #         logging.warning("ClusterIndex: visualization failed: %s", exc)

    # --- Stage 2: per-column similarity filter with largest-gap cutoff ---
    agent_data = await agent_registry.load(data_agents)

    col_entries: list[tuple[str, str, str]] = []
    col_descriptions: list[str] = []
    for agent in data_agents:
        data = agent_data[agent.name]
        for table_name, table_desc in data.individual_descriptions.items():
            for col_name, col_desc in table_desc.columns.items():
                col_entries.append((agent.name, table_name, col_name))
                col_descriptions.append(f"{col_name}: {col_desc.natural_language_description}")

    if not col_entries:
        logging.warning("No column embeddings found, returning all agents unfiltered")
        return data_agents

    col_embeddings = await embedding.create(col_descriptions)

    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    logging.info("Peak RSS after column embeddings (%d columns): %.1f MB", len(col_descriptions), peak_mb)

    vector_index: VectorIndex[tuple[str, str, str]] = VectorIndex.build(
        list(zip(col_entries, col_embeddings))
    )

    agent_best_dist: dict[str, tuple[float, str, str]] = {}
    for (agent_name, table_name, col_name), dist in vector_index.search_all(query_embedding):
        if agent_name not in agent_best_dist or dist < agent_best_dist[agent_name][0]:
            agent_best_dist[agent_name] = (dist, table_name, col_name)

    sorted_agents = sorted(agent_best_dist.items(), key=lambda x: x[1][0])

    logging.info("Embedding similarity per agent (L2 distance, ascending):")
    for agent_name, (dist, table_name, col_name) in sorted_agents:
        logging.info("  %s: L2=%.4f (table: %s, col: %s)", agent_name, dist, table_name, col_name)

    l2_scores = [dist for _, (dist, _, _) in sorted_agents]
    cutoff_idx = find_l2_clusters(l2_scores, min_cluster_size=1)
    sorted_agents = sorted_agents[:cutoff_idx]

    logging.info("Embedding pre-filter: kept %d / %d agents", len(sorted_agents), len(data_agents))

    agent_map = {a.name: a for a in data_agents}
    filtered = [agent_map[name] for name, _ in sorted_agents if name in agent_map]

    tracing.append_to_span_output(
        {
            "filtered_agents": [a.name for a in filtered],
            "l2_scores": {name: round(float(dist), 4) for name, (dist, _, _) in agent_best_dist.items()},
            "best_columns": {name: f"{tbl}.{col}" for name, (_, tbl, col) in agent_best_dist.items()},
        }
    )

    hooks.emit(
        HookEvent.EMBEDDING_PRE_FILTER_END,
        query=query,
        embedding_filtered_agents=filtered,
        total_agents=len(data_agents),
        l2_scores={name: round(float(dist), 4) for name, (dist, _, _) in agent_best_dist.items()},
        best_columns={name: f"{tbl}.{col}" for name, (_, tbl, col) in agent_best_dist.items()},
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

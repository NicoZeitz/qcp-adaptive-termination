"""Embedding-based pre-filter for data agent selection."""

import asyncio
import logging
import resource

import numpy as np
from pydantic import BaseModel

import qcp_adaptive_termination.agent_registry as agent_registry
import qcp_adaptive_termination.embedding as embedding
import qcp_adaptive_termination.hooks as hooks
from qcp_adaptive_termination.embedding import VectorIndex
from qcp_adaptive_termination.hooks import HookEvent
from qcp_adaptive_termination.llm import runner
from qcp_adaptive_termination.models import AgentRegistryType, AgentRole, DataAgent
from qcp_adaptive_termination.models.agent import AgentRegistryData


class ColumnRelevanceResult(BaseModel):
    reasoning: str
    relevant: bool
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


def find_kneedle_cutoff(
    sorted_agents: list[tuple[str, tuple[float, str, str]]],
) -> list[tuple[str, tuple[float, str, str]]]:
    """Geometric elbow (kneedle) cutoff on sorted (ascending) L2-distance agents.

    Mirrors the algorithm used in analyze_best_cutoff.py:
      1. Collapse agents with identical L2 distances into groups.
      2. Find the index with maximum perpendicular distance from the diagonal
         (both axes normalised to [0, 1]).
      3. Expand the kept groups back to individual agent entries.
    """
    if len(sorted_agents) <= 1:
        return sorted_agents

    # 1. Collapse equal distances
    groups: list[list[tuple[str, tuple[float, str, str]]]] = []
    for entry in sorted_agents:
        dist = entry[1][0]
        if groups and groups[-1][0][1][0] == dist:
            groups[-1].append(entry)
        else:
            groups.append([entry])

    n = len(groups)
    if n <= 2:
        return sorted_agents

    scores = np.array([g[0][1][0] for g in groups], dtype=float)
    x = np.arange(n, dtype=float)

    x_range = x[-1] - x[0]
    s_range = float(scores[-1] - scores[0])
    if x_range == 0 or s_range == 0:
        return sorted_agents

    xn = (x - x[0]) / x_range
    yn = (scores - scores[0]) / s_range
    knee = int(np.argmax(np.abs(yn - xn)))
    cutoff = knee + 1  # exclusive

    logging.info(
        "Kneedle cutoff: keeping %d / %d score groups (%d / %d agents), knee at L2=%.4f",
        cutoff,
        n,
        sum(len(g) for g in groups[:cutoff]),
        len(sorted_agents),
        float(scores[knee]),
    )

    return [entry for g in groups[:cutoff] for entry in g]


async def _column_relevance_llm(
    query: str,
    agent_name: str,
    table_name: str,
    col_name: str,
    agent_data: dict[str, AgentRegistryData],
) -> bool:
    td = agent_data[agent_name].individual_descriptions[table_name]
    col_desc = td.columns[col_name].natural_language_description
    result = await runner.run_llm(
        role=AgentRole.EMBEDDING_COLUMN_RELEVANCE,
        agent_name=agent_name,
        prompt_parameters={
            "query": query,
            "database_description": agent_data[agent_name].natural_language_description,
            "schema": agent_data[agent_name].technical_description,
            "table_name": table_name,
            "table_description": td.natural_language_description,
            "column_name": col_name,
            "column_description": col_desc,
        },
        output_type=ColumnRelevanceResult,
    )
    logging.info(
        "Column-relevance LLM [%s] relevant=%s — %s",
        agent_name,
        result.relevant,
        result.reasoning,
    )
    return result.relevant


async def filter_by_column_relevance(
    query: str,
    sorted_agents: list[tuple[str, tuple[float, str, str]]],
    agent_data: dict[str, AgentRegistryData],
) -> list[tuple[str, tuple[float, str, str]]]:
    """Run one LLM call per agent to confirm relevance based on best-matching column."""
    semaphore = asyncio.Semaphore(25)

    async def _check(entry: tuple[str, tuple[float, str, str]]) -> tuple[str, tuple[float, str, str]] | None:
        name, (dist, table_name, col_name) = entry
        async with semaphore:
            relevant = await _column_relevance_llm(query, name, table_name, col_name, agent_data)
        return entry if relevant else None

    results = await asyncio.gather(*[_check(e) for e in sorted_agents])
    kept = [e for e in results if e is not None]
    logging.info("Column-relevance LLM filter: kept %d / %d agents", len(kept), len(sorted_agents))
    return kept


async def _filter_embeddings_internal(
    query: str,
    data_agents: list[DataAgent],
    registry_type: AgentRegistryType,
) -> list[DataAgent]:
    # --- Stage 1: cluster-based pre-filter ---
    # cluster_index = await get_or_build_cluster_index(data_agents, registry_type)
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

    # Fire query embedding and registry load concurrently — they are independent.
    (query_embedding,), agent_data = await asyncio.gather(
        embedding.create([query]),
        agent_registry.load(data_agents),
    )

    # --- Stage 2: per-column similarity filter with kneedle (geometric elbow) cutoff ---
    col_entries: list[tuple[str, str, str]] = []
    col_descriptions: list[str] = []
    for agent in data_agents:
        data = agent_data[agent.name]
        for table_name, table_desc in data.individual_descriptions.items():
            for col_name, col_desc in table_desc.columns.items():
                col_entries.append((agent.name, table_name, col_name))
                col_descriptions.append(f"{table_name}: {table_desc.natural_language_description}, {col_name}: {col_desc.natural_language_description}")

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

    sorted_agents_after_kneedle = find_kneedle_cutoff(sorted_agents)

    kneedle_names = {name for name, _ in sorted_agents_after_kneedle}
    logging.info(
        "After kneedle: %d / %d agents  rejected=%s",
        len(sorted_agents_after_kneedle),
        len(sorted_agents),
        sorted(name for name, _ in sorted_agents if name not in kneedle_names),
    )
    tracing.append_to_span_output(
        {
            "after_kneedle_agents": [name for name, _ in sorted_agents_after_kneedle],
            "after_kneedle_count": len(sorted_agents_after_kneedle),
        }
    )

    sorted_agents_after_llm = await filter_by_column_relevance(query, sorted_agents_after_kneedle, agent_data)

    llm_names = {name for name, _ in sorted_agents_after_llm}
    logging.info(
        "After column-relevance LLM: %d / %d agents  rejected=%s",
        len(sorted_agents_after_llm),
        len(sorted_agents_after_kneedle),
        sorted(name for name, _ in sorted_agents_after_kneedle if name not in llm_names),
    )
    tracing.append_to_span_output(
        {
            "after_llm_filter_agents": [name for name, _ in sorted_agents_after_llm],
            "after_llm_filter_count": len(sorted_agents_after_llm),
        }
    )

    logging.info(
        "Embedding pre-filter total: %d → kneedle → %d → llm → %d",
        len(data_agents),
        len(sorted_agents_after_kneedle),
        len(sorted_agents_after_llm),
    )

    agent_map = {a.name: a for a in data_agents}
    filtered = [agent_map[name] for name, _ in sorted_agents_after_llm if name in agent_map]

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
        after_kneedle_agents=[name for name, _ in sorted_agents_after_kneedle],
        after_llm_filter_agents=[name for name, _ in sorted_agents_after_llm],
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

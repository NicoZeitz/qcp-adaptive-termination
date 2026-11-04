"""Embedding-based pre-filter for data agent selection."""

import asyncio
import logging
from collections import defaultdict

import qcp_adaptive_termination.agent_registry as agent_registry
import qcp_adaptive_termination.embedding as embedding
from qcp_adaptive_termination.embedding import VectorIndex
from qcp_adaptive_termination.models import DataAgent
from qcp_adaptive_termination.observability import tracing

from ._llm_steps import (
    QueryCompletenessResult,
    agent_relevance,
    baseline_filter,
    query_completeness,
    select_candidates,
    try_decompose,
)


async def filter_by_llm_baseline(
    query: str,
    data_agents: list[DataAgent],
) -> list[DataAgent]:
    """Baseline: one parallel LLM call per agent schema, no embedding guidance."""
    semaphore = asyncio.Semaphore(5)

    async def _check_with_semaphore(agent: DataAgent) -> tuple[DataAgent, bool]:
        async with semaphore:
            return await baseline_filter(query, agent)

    results = await asyncio.gather(*[_check_with_semaphore(a) for a in data_agents])
    relevant = [agent for agent, is_relevant in results if is_relevant]
    logging.info("Baseline filter: kept %d / %d agents", len(relevant), len(data_agents))
    return relevant


async def iterative_embedding_filter(
    query: str,
    data_agents: list[DataAgent],
    max_rounds: int = 10,
    top_cols_per_agent: int = 3,
    boundary_window: int = 50,
    max_expansions: int = 3,
) -> list[DataAgent]:
    """Parallel queue-based filter with semantic boundary finding.

    Each round:
      A: embed all current queries
      B: select candidates per query (parallel) — NL descriptions + L2 distances
      C: group results by agent
      D: one agent-relevance LLM call per unique candidate agent (parallel)
      E: one query-completeness call → is_complete + new queries for next round
    """
    agent_data = await agent_registry.load(data_agents)

    col_entries: list[tuple[str, str, str]] = []
    col_descriptions: list[str] = []
    for agent in data_agents:
        data = agent_data[agent.name]
        for table_name, table_desc in data.individual_descriptions.items():
            for col_name, col_desc in table_desc.columns.items():
                col_entries.append((agent.name, table_name, col_name))
                col_descriptions.append(
                    f"{table_desc.natural_language_description} {col_desc.natural_language_description}"
                )

    if not col_entries:
        logging.warning("Iterative Embedding Filter: no column embeddings, returning all agents")
        return data_agents

    col_embeddings = await embedding.create(col_descriptions)
    vector_index: VectorIndex[tuple[str, str, str]] = VectorIndex.build(
        list(zip(col_entries, col_embeddings, strict=True))
    )

    agent_map = {a.name: a for a in data_agents}
    selected_order: list[str] = []
    selected_set: set[str] = set()
    agent_coverage: dict[str, tuple[str, str]] = {}
    agent_best: dict[str, tuple[float, str, str]] = {}
    semaphore = asyncio.Semaphore(5)

    current_queries: list[str] = [query]
    failed_queries: list[tuple[str, str]] = []
    all_tried_queries: set[str] = {query}

    for round_idx in range(max_rounds):
        round_label = f"round {round_idx}"
        logging.info(
            "Iterative Embedding Filter round %d: %d queries: %s",
            round_idx,
            len(current_queries),
            current_queries,
        )

        # Phase A: embed all queries
        query_embeddings = await embedding.create(current_queries)

        for q_emb in query_embeddings:
            for (agent_name, table_name, col_name), dist in vector_index.search_all(q_emb):
                if agent_name not in agent_best or dist < agent_best[agent_name][0]:
                    agent_best[agent_name] = (dist, table_name, col_name)

        # Phase B: select candidates per query (parallel)
        boundary_results: list[tuple[str, list[str]]] = await asyncio.gather(*[
            select_candidates(
                q, emb, vector_index, agent_data, selected_set,
                top_cols_per_agent, boundary_window, max_expansions,
            )
            for q, emb in zip(current_queries, query_embeddings, strict=True)
        ])

        # Phase C: group by agent and by query
        agent_to_queries: dict[str, list[str]] = defaultdict(list)
        query_to_agents: dict[str, list[str]] = defaultdict(list)
        for q, candidates in boundary_results:
            for name in candidates:
                if name not in selected_set and name in agent_data:
                    agent_to_queries[name].append(q)
                    query_to_agents[q].append(name)

        already_failed = {q for q, _ in failed_queries}
        for q, candidates in boundary_results:
            if q in already_failed:
                continue
            if not candidates:
                failed_queries.append((q, "no agents found"))
            elif all(name in selected_set for name in candidates):
                failed_queries.append((q, "only matched already-confirmed agents"))

        if not agent_to_queries:
            if set(current_queries) != {query}:
                logging.info(
                    "Iterative Embedding Filter round %d: no candidates found for sub-queries — stopping",
                    round_idx,
                )
                break
            logging.info(
                "Iterative Embedding Filter round %d: no candidates found — running decomposer",
                round_idx,
            )
            new_queries = await try_decompose(query, failed_queries, all_tried_queries, round_idx)
            if new_queries is None:
                break
            current_queries = new_queries
            continue

        # Phase D: agent-relevance check per candidate (parallel)
        agent_results = await asyncio.gather(*[
            agent_relevance(name, sqs, query, agent_data, semaphore, round_label)
            for name, sqs in agent_to_queries.items()
        ])

        newly_confirmed: set[str] = set()
        for name, is_relevant, covered, missing in agent_results:
            if is_relevant and name not in selected_set:
                selected_order.append(name)
                selected_set.add(name)
                agent_coverage[name] = (covered, missing)
                newly_confirmed.add(name)

        already_failed = {q for q, _ in failed_queries}
        for q, agents in query_to_agents.items():
            if q not in already_failed and not any(name in newly_confirmed for name in agents):
                failed_queries.append((q, "candidates found but all rejected as irrelevant"))

        logging.info(
            "Iterative Embedding Filter round %d: confirmed=%s",
            round_idx,
            [n for n, rel, _, _ in agent_results if rel],
        )

        if not newly_confirmed and not selected_order:
            if set(current_queries) != {query}:
                logging.info(
                    "Iterative Embedding Filter round %d: nothing confirmed for sub-queries — stopping",
                    round_idx,
                )
                break
            logging.info(
                "Iterative Embedding Filter round %d: nothing confirmed yet — running decomposer",
                round_idx,
            )
            new_queries = await try_decompose(query, failed_queries, all_tried_queries, round_idx)
            if new_queries is None:
                break
            current_queries = new_queries
            continue

        # Phase E: check if coverage is complete
        agg: QueryCompletenessResult = await query_completeness(
            query, selected_order, agent_coverage, failed_queries, round_label
        )
        logging.info(
            "Iterative Embedding Filter round %d: is_complete=%s data_not_available=%s queries=%s",
            round_idx,
            agg.is_complete,
            agg.data_not_available,
            agg.queries,
        )

        if agg.is_complete or agg.data_not_available:
            break

        current_queries = [q for q in agg.queries if q and q not in all_tried_queries] if agg.queries else []
        all_tried_queries.update(current_queries)
        if not current_queries:
            logging.info(
                "Iterative Embedding Filter round %d: no new queries from completeness check — running decomposer",
                round_idx,
            )
            new_queries = await try_decompose(query, failed_queries, all_tried_queries, round_idx)
            if new_queries is None:
                break
            current_queries = new_queries

    filtered = [agent_map[name] for name in selected_order if name in agent_map]
    l2_scores = {name: round(float(dist), 4) for name, (dist, _, _) in agent_best.items()}

    logging.info(
        "Iterative Embedding Filter total: kept %d / %d agents  rejected=%s",
        len(filtered),
        len(data_agents),
        sorted(a.name for a in data_agents if a.name not in selected_set),
    )
    tracing.append_to_span_output(
        {
            "after_llm_filter_agents": selected_order,
            "after_llm_filter_count": len(filtered),
            "filtered_agents": [a.name for a in filtered],
            "l2_scores": l2_scores,
            "best_columns": {name: f"{tbl}.{col}" for name, (_, tbl, col) in agent_best.items()},
        }
    )
    return filtered

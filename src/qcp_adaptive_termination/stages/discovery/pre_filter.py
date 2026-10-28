"""Embedding-based pre-filter for data agent selection."""

import asyncio
import logging
import resource
import time
from collections import defaultdict

import numpy as np
from openai import APITimeoutError

from pydantic import BaseModel

import qcp_adaptive_termination.agent_registry as agent_registry
import qcp_adaptive_termination.embedding as embedding
import qcp_adaptive_termination.hooks as hooks
from qcp_adaptive_termination.embedding import VectorIndex
from qcp_adaptive_termination.hooks import HookEvent
from qcp_adaptive_termination.llm import runner
from qcp_adaptive_termination.models import AgentRegistryType, AgentRole, DataAgent
from qcp_adaptive_termination.models.agent import AgentRegistryData
from qcp_adaptive_termination.observability import tracing
from qcp_adaptive_termination.observability import context as obs_context

class _SingleAgentRelevanceResult(BaseModel):
    is_relevant: bool

async def _check_single_agent(query: str, agent: DataAgent) -> tuple[DataAgent, bool]:
    schema = await agent_registry.build_registry_string(agent, AgentRegistryType.TECHNICAL)
    result = await runner.run_llm(
        role=AgentRole.BASELINE_FILTER,
        agent_name=agent.name,
        prompt_parameters={
            "schema": schema,
            "query": query,
        },
        output_type=_SingleAgentRelevanceResult,
    )
    logging.info(
        "Baseline filter [%s] is_relevant=%s — %s",
        agent.name,
        result.is_relevant,
    )
    tracing.append_to_span_output(
        {
            f"{agent.name}_relevant": result.is_relevant,
        }
    )
    return agent, result.is_relevant


async def filter_by_llm_baseline(
    query: str,
    data_agents: list[DataAgent],
) -> list[DataAgent]:
    """Baseline: one parallel LLM call per agent schema, no embedding guidance."""
    # Initialise the accumulator *before* spawning child tasks so all children
    # share a pointer to the same object (asyncio.gather copies the context snapshot).
    obs_context.get_usage_accumulator()

    semaphore = asyncio.Semaphore(5)

    async def _check_with_semaphore(agent: DataAgent) -> tuple[DataAgent, bool]:
        async with semaphore:
            while True:
                try:
                    return await _check_single_agent(query, agent)
                except APITimeoutError:
                    logging.warning("Baseline filter [%s] timed out — retrying", agent.name)
                    await asyncio.sleep(2)

    results = await asyncio.gather(*[_check_with_semaphore(a) for a in data_agents])
    relevant = [agent for agent, is_relevant in results if is_relevant]
    logging.info("Baseline filter: kept %d / %d agents", len(relevant), len(data_agents))
    return relevant

class _BoundaryResult(BaseModel):
    cutoff_rank: int | None
    excluded_ranks: list[int]


class _DecomposerResult(BaseModel):
    queries: list[str]


class _AgentRelevanceResult(BaseModel):
    is_relevant: bool
    covered_aspects: str
    missing_aspects: str


class _AggregateResult(BaseModel):
    is_complete: bool
    data_not_available: bool
    queries: list[str]


async def _boundary_finder(
    query: str,
    cq_embedding: list[float],
    vector_index: VectorIndex[tuple[str, str, str]],
    agent_data: dict[str, AgentRegistryData],
    selected_set: set[str],
    top_cols_per_agent: int = 1,
    boundary_window: int = 50,
) -> tuple[str, list[str]]:
    """Run boundary finder for one query. Returns (query, candidate_agent_names)."""
    t_rank = time.monotonic()
    cq_agent_cols: dict[str, list[tuple[float, str, str]]] = {}
    for (agent_name, table_name, col_name), dist in vector_index.search_all(cq_embedding):
        cols = cq_agent_cols.setdefault(agent_name, [])
        if len(cols) < top_cols_per_agent:
            cols.append((dist, table_name, col_name))
    obs_context.get_usage_accumulator().ranking_duration_ms += int((time.monotonic() - t_rank) * 1000)

    cq_agent_best = {name: cols[0] for name, cols in cq_agent_cols.items()}
    sorted_cq = sorted(cq_agent_best.items(), key=lambda x: x[1][0])
    windowed_cq = sorted_cq[:boundary_window]
    ranked_lines = []
    for rank, (name, (dist, best_tbl, _)) in enumerate(windowed_cq, start=1):
        td = agent_data[name].individual_descriptions.get(best_tbl)
        tbl_nl = td.natural_language_description if td else best_tbl
        col_parts = []
        for col_dist, col_tbl, col_col in cq_agent_cols[name]:
            col_td = agent_data[name].individual_descriptions.get(col_tbl)
            col_nl = col_td.columns[col_col].natural_language_description if col_td and col_col in col_td.columns else col_col
            col_parts.append(f"{col_nl} (L2={col_dist:.4f})")
        ranked_lines.append(f"[{rank}] {tbl_nl} {' '.join(col_parts)}")

    logging.info("Iterative Embedding Filter query %r — boundary ranked agents (window=%d/%d):", query, len(windowed_cq), len(sorted_cq))
    for line in ranked_lines:
        logging.info("  %s", line)

    while True:
        try:
            result = await runner.run_llm(
                role=AgentRole.EMBEDDING_BOUNDARY_FINDER,
                prompt_parameters={
                    "query": query,
                    "ranked_agents": "\n".join(ranked_lines),
                },
                output_type=_BoundaryResult,
            )
            break
        except APITimeoutError:
            logging.warning("Iterative Embedding Filter LLM timed out for %r — retrying", query)
            await asyncio.sleep(2)

    logging.info(
        "Iterative Embedding Filter query %r: cutoff_rank=%s excluded_ranks=%s — %s",
        query,
        result.cutoff_rank,
        result.excluded_ranks,
    )

    if result.cutoff_rank is None:
        return query, []

    if result.cutoff_rank >= boundary_window and len(sorted_cq) > boundary_window:
        logging.warning(
            "Iterative Embedding Filter query %r: cutoff_rank=%d hit the window ceiling (%d) — "
            "%d agents were not shown; consider increasing boundary_window",
            query, result.cutoff_rank, boundary_window, len(sorted_cq) - boundary_window,
        )

    excluded = set(result.excluded_ranks)
    candidates = [
        name for rank, (name, _) in enumerate(windowed_cq[:result.cutoff_rank], start=1)
        if name not in selected_set and rank not in excluded
    ]
    return query, candidates


async def _check_agent(
    agent_name: str,
    matched_queries: list[str],
    query: str,
    agent_data: dict[str, AgentRegistryData],
    semaphore: asyncio.Semaphore,
    round_label: str,
) -> tuple[str, bool, str, str]:
    schema = agent_data[agent_name].technical_description
    matched_str = "\n".join(f"- {q}" for q in matched_queries)
    async with semaphore:
        while True:
            try:
                result = await runner.run_llm(
                    role=AgentRole.ITERATIVE_EMBEDDING_FILTER_AGENT_RELEVANCE,
                    agent_name=f"{round_label} - {agent_name}",
                    prompt_parameters={
                        "query": query,
                        "matched_queries": matched_str,
                        "schema": schema,
                    },
                    output_type=_AgentRelevanceResult,
                )
                break
            except APITimeoutError:
                logging.warning("Iterative Embedding Filter agent LLM [%s] timed out — retrying", agent_name)
                await asyncio.sleep(2)
    logging.info(
        "Iterative Embedding Filter agent [%s] is_relevant=%s matched=%s covered=%r missing=%r — %s",
        agent_name,
        result.is_relevant,
        matched_queries,
        result.covered_aspects,
        result.missing_aspects,
    )
    return agent_name, result.is_relevant, result.covered_aspects, result.missing_aspects


async def _decompose(
    query: str,
    failed_queries: list[tuple[str, str]],
) -> list[str]:
    failed_str = "\n".join(f"- {q} ({reason})" for q, reason in failed_queries)
    while True:
        try:
            result = await runner.run_llm(
                role=AgentRole.ITERATIVE_EMBEDDING_FILTER_DECOMPOSER,
                prompt_parameters={
                    "query": query,
                    "failed_queries": failed_str,
                },
                output_type=_DecomposerResult,
            )
            break
        except APITimeoutError:
            logging.warning("Iterative Embedding Filter decomposer LLM timed out — retrying")
            await asyncio.sleep(2)
    logging.info(
        "Iterative Embedding Filter: queries=%s — %s",
        result.queries,
    )
    return result.queries


async def iterative_embedding_filter(
    query: str,
    data_agents: list[DataAgent],
    max_rounds: int = 10,
    top_cols_per_agent: int = 3,
    boundary_window: int = 50,
) -> list[DataAgent]:
    """Parallel queue-based filter with semantic boundary finding.

    Each round processes all current queries in parallel:
      Phase A: embed all queries in one batch
      Phase B: boundary finder per query (parallel), each sees NL descriptions + L2
      Phase C: group results by agent → each agent accumulates its matched queries
      Phase D: one agent-relevance LLM call per unique agent (parallel)
      Phase E: one aggregate call → is_complete + new queries for next round

    No rejected_set: agents can be re-discovered via different queries in later rounds.
    """
    obs_context.get_usage_accumulator()

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
        list(zip(col_entries, col_embeddings))
    )

    agent_map = {a.name: a for a in data_agents}
    selected_order: list[str] = []
    selected_set: set[str] = set()
    agent_coverage: dict[str, tuple[str, str]] = {}
    agent_best: dict[str, tuple[float, str, str]] = {}
    semaphore = asyncio.Semaphore(5)
    original_query_found_nothing = False

    current_queries: list[str] = [query]
    failed_queries: list[tuple[str, str]] = []  # (query, reason)
    all_tried_queries: set[str] = {query}

    for round_idx in range(max_rounds):
        logging.info(
            "Iterative Embedding Filter round %d: %d queries: %s",
            round_idx,
            len(current_queries),
            current_queries,
        )

        # --- Phase A + B: embed all queries and run boundary finder in parallel ---
        cq_embeddings = await embedding.create(current_queries)

        # Update global agent_best for tracing (side-effectful but read-only on vector_index)
        agent_all_dists: dict[str, list[float]] = defaultdict(list)
        for cq_emb in cq_embeddings:
            for (agent_name, table_name, col_name), dist in vector_index.search_all(cq_emb):
                if agent_name not in agent_best or dist < agent_best[agent_name][0]:
                    agent_best[agent_name] = (dist, table_name, col_name)
                agent_all_dists[agent_name].append(dist)

        boundary_tasks = [
            _boundary_finder(q, emb, vector_index, agent_data, selected_set, top_cols_per_agent, boundary_window)
            for q, emb in zip(current_queries, cq_embeddings)
        ]
        boundary_results: list[tuple[str, list[str]]] = await asyncio.gather(*boundary_tasks)

        # --- Phase C: group by agent and by query ---
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
            logging.info(
                "Iterative Embedding Filter round %d: all boundary finders returned no candidates — running decomposer",
                round_idx,
            )
            if round_idx == 0:
                original_query_found_nothing = True
            decomposed = await _decompose(query, failed_queries)
            new_queries = [q for q in decomposed if q not in all_tried_queries]
            if not new_queries:
                logging.info("Iterative Embedding Filter: decomposer returned no new queries, stopping")
                break
            all_tried_queries.update(new_queries)
            current_queries = new_queries
            continue

        # --- Phase D: one LLM call per unique agent (parallel) ---
        round_label = f"round {round_idx}"
        agent_tasks = [
            _check_agent(name, sqs, query, agent_data, semaphore, round_label)
            for name, sqs in agent_to_queries.items()
        ]
        agent_results = await asyncio.gather(*agent_tasks)

        any_confirmed = False
        newly_confirmed: set[str] = set()
        for name, is_relevant, covered, missing in agent_results:
            if is_relevant and name not in selected_set:
                selected_order.append(name)
                selected_set.add(name)
                agent_coverage[name] = (covered, missing)
                any_confirmed = True
                newly_confirmed.add(name)

        # queries where all candidates were checked and all rejected
        already_failed = {q for q, _ in failed_queries}
        for q, agents in query_to_agents.items():
            if q not in already_failed and not any(name in newly_confirmed for name in agents):
                failed_queries.append((q, "candidates found but all rejected as irrelevant"))

        logging.info(
            "Iterative Embedding Filter round %d: confirmed=%s",
            round_idx,
            [n for n, rel, _, _ in agent_results if rel],
        )

        # --- Phase E: aggregate ---
        coverage_lines = [
            f"Agent {name}: {agent_coverage[name][0]}"
            + (f" - Missing: {agent_coverage[name][1]}" if agent_coverage[name][1] else "")
            for name in selected_order
        ]
        agent_coverage_summary = "\n".join(coverage_lines)
        failed_str = "\n".join(f"- {q} ({reason})" for q, reason in failed_queries)

        while True:
            try:
                agg = await runner.run_llm(
                    role=AgentRole.ITERATIVE_EMBEDDING_FILTER_QUERY_COMPLETENESS,
                    agent_name=round_label,
                    prompt_parameters={
                        "query": query,
                        "agent_coverage_summary": agent_coverage_summary,
                        "failed_queries": failed_str,
                        "original_query_found_nothing": "true" if original_query_found_nothing else "",
                    },
                    output_type=_AggregateResult,
                )
                break
            except APITimeoutError:
                logging.warning("Iterative Embedding Filter query completeness LLM timed out — retrying")
                await asyncio.sleep(2)

        logging.info(
            "Iterative Embedding Filter round %d aggregate: is_complete=%s data_not_available=%s queries=%s — %s",
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
            break

    filtered = [agent_map[name] for name in selected_order if name in agent_map]
    l2_scores = {name: round(float(dist), 4) for name, (dist, _, _) in agent_best.items()}

    logging.info(
        "Iterative Embedding Filter total: kept %d / %d agents  rejected=%s",
        len(filtered),
        len(data_agents),
        sorted(a.name for a in data_agents if a.name not in selected_set),
    )

    hooks.emit(
        HookEvent.EMBEDDING_PRE_FILTER_END,
        query=query,
        embedding_filtered_agents=filtered,
        total_agents=len(data_agents),
        l2_scores=l2_scores,
        after_llm_filter_agents=selected_order,
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

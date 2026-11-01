"""LLM call wrappers for the iterative embedding filter."""

import asyncio
import logging
from collections.abc import Callable, Coroutine
from typing import Any

import numpy as np
from openai import APITimeoutError
from pydantic import BaseModel

import qcp_adaptive_termination.agent_registry as agent_registry
import qcp_adaptive_termination.embedding as embedding
from qcp_adaptive_termination.embedding import VectorIndex
from qcp_adaptive_termination.llm import runner
from qcp_adaptive_termination.models import AgentRegistryType, AgentRole, DataAgent
from qcp_adaptive_termination.models.agent import AgentRegistryData
from qcp_adaptive_termination.observability import tracing


async def _run_with_retry[T](
    make_coro: Callable[[], Coroutine[Any, Any, T]],
    label: str,
) -> T:
    while True:
        try:
            return await make_coro()
        except APITimeoutError:
            logging.warning("LLM timed out (%s) — retrying", label)
            await asyncio.sleep(2)


# --- Output models ---

class _SingleAgentRelevanceResult(BaseModel):
    is_relevant: bool


class _RankDecision(BaseModel):
    rank: int
    is_relevant: bool


class BoundaryResult(BaseModel):
    decisions: list[_RankDecision]
    expand_to_rank: int | None = None


class DecomposerResult(BaseModel):
    queries: list[str]


class AgentRelevanceResult(BaseModel):
    is_relevant: bool
    covered_aspects: str
    missing_aspects: str


class QueryCompletenessResult(BaseModel):
    is_complete: bool
    data_not_available: bool
    queries: list[str]


# --- LLM steps ---

async def baseline_filter(query: str, agent: DataAgent) -> tuple[DataAgent, bool]:
    schema = await agent_registry.build_registry_string(agent, AgentRegistryType.TECHNICAL)
    result = await _run_with_retry(
        lambda: runner.run_llm(
            role=AgentRole.BASELINE_FILTER,
            agent_name=agent.name,
            prompt_parameters={"schema": schema, "query": query},
            output_type=_SingleAgentRelevanceResult,
        ),
        label=f"baseline_filter/{agent.name}",
    )
    is_relevant = result.output.is_relevant
    logging.info("Baseline filter [%s] is_relevant=%s", agent.name, is_relevant)
    tracing.append_to_span_output({f"{agent.name}_relevant": is_relevant})
    return agent, is_relevant


async def select_candidates(
    query: str,
    query_embedding: np.ndarray,
    vector_index: VectorIndex[tuple[str, str, str]],
    agent_data: dict[str, AgentRegistryData],
    selected_set: set[str],
    top_cols_per_agent: int,
    boundary_window: int,
    max_expansions: int,
) -> tuple[str, list[str]]:
    """Evaluate a ranked list of agents for one query. Returns (query, candidate_agent_names)."""
    agent_col_hits: dict[str, list[tuple[float, str, str]]] = {}
    for (agent_name, table_name, col_name), dist in vector_index.search_all(query_embedding):
        cols = agent_col_hits.setdefault(agent_name, [])
        if len(cols) < top_cols_per_agent:
            cols.append((dist, table_name, col_name))

    agent_best_hit = {name: cols[0] for name, cols in agent_col_hits.items()}
    agents_by_distance = sorted(agent_best_hit.items(), key=lambda x: x[1][0])

    def _build_ranked_lines(window: list[tuple[str, tuple[float, str, str]]]) -> list[str]:
        lines = []
        for rank, (name, (_dist, best_tbl, _)) in enumerate(window, start=1):
            td = agent_data[name].individual_descriptions.get(best_tbl)
            tbl_nl = td.natural_language_description if td else best_tbl
            col_parts = []
            for col_dist, col_tbl, col_col in agent_col_hits[name]:
                col_td = agent_data[name].individual_descriptions.get(col_tbl)
                col_nl = (
                    col_td.columns[col_col].natural_language_description
                    if col_td and col_col in col_td.columns
                    else col_col
                )
                col_parts.append(f"{col_nl} (L2={col_dist:.4f})")
            lines.append(f"[{rank}] {tbl_nl} {' '.join(col_parts)}")
        return lines

    def _tail_line(shown_count: int) -> str | None:
        tail = agents_by_distance[shown_count:]
        if not tail:
            return None
        tail_dists = [d for _, (d, _, _) in tail]
        return f"[tail] {len(tail)} agents not shown, L2 range: {min(tail_dists):.4f} to {max(tail_dists):.4f}"

    all_ranked_lines = _build_ranked_lines(agents_by_distance)
    current_window_size = min(boundary_window, len(agents_by_distance))
    prompt_lines = all_ranked_lines[:current_window_size]
    tail = _tail_line(current_window_size)
    if tail:
        prompt_lines = prompt_lines + [tail]
    else:
        prompt_lines = prompt_lines + ["[all agents shown — no further agents exist]"]

    logging.info(
        "Iterative Embedding Filter query %r — boundary ranked agents (window=%d/%d):",
        query,
        current_window_size,
        len(agents_by_distance),
    )
    for line in all_ranked_lines[:current_window_size]:
        logging.info("  %s", line)

    all_decisions: dict[int, bool] = {}

    for expansion in range(max_expansions + 1):
        result = await _run_with_retry(
            lambda pl=prompt_lines: runner.run_llm(
                role=AgentRole.EMBEDDING_BOUNDARY_FINDER,
                prompt_parameters={
                    "query": query,
                    "ranked_agents": "\n".join(pl),
                },
                output_type=BoundaryResult,
            ),
            label=f"boundary_finder/{query!r}",
        )
        boundary = result.output

        for d in boundary.decisions:
            if d.rank not in all_decisions:
                all_decisions[d.rank] = d.is_relevant

        logging.info(
            "Iterative Embedding Filter query %r (expansion=%d): decisions=%s expand_to_rank=%s",
            query,
            expansion,
            {r: v for r, v in sorted(all_decisions.items())},
            boundary.expand_to_rank,
        )

        if boundary.expand_to_rank is None or boundary.expand_to_rank <= current_window_size:
            break
        if expansion == max_expansions:
            logging.warning(
                "Iterative Embedding Filter query %r: expand_to_rank=%d still requested after %d expansions — stopping",
                query,
                boundary.expand_to_rank,
                max_expansions,
            )
            break
        if current_window_size >= len(agents_by_distance):
            break

        new_window_size = min(boundary.expand_to_rank, len(agents_by_distance))
        new_tail_lines = all_ranked_lines[current_window_size:new_window_size]
        logging.info(
            "Iterative Embedding Filter query %r: expanding window %d → %d (%d new agents)",
            query,
            current_window_size,
            new_window_size,
            len(new_tail_lines),
        )
        current_window_size = new_window_size

        relevant_ranks = sorted(r for r, v in all_decisions.items() if v)
        curated_prior = [all_ranked_lines[r - 1] for r in relevant_ranks if r <= len(all_ranked_lines)]
        tail = _tail_line(current_window_size)
        tail_part = [tail] if tail else ["[all agents shown — no further agents exist]"]
        prompt_lines = curated_prior + new_tail_lines + tail_part

    candidates = [
        name
        for rank, (name, _) in enumerate(agents_by_distance[:current_window_size], start=1)
        if all_decisions.get(rank, False) and name not in selected_set
    ]
    return query, candidates


async def agent_relevance(
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
        result = await _run_with_retry(
            lambda: runner.run_llm(
                role=AgentRole.ITERATIVE_EMBEDDING_FILTER_AGENT_RELEVANCE,
                agent_name=f"{round_label} - {agent_name}",
                prompt_parameters={
                    "query": query,
                    "matched_queries": matched_str,
                    "schema": schema,
                },
                output_type=AgentRelevanceResult,
            ),
            label=f"agent_relevance/{agent_name}",
        )
    relevance = result.output
    logging.info(
        "Iterative Embedding Filter agent [%s] is_relevant=%s matched=%s covered=%r missing=%r",
        agent_name,
        relevance.is_relevant,
        matched_queries,
        relevance.covered_aspects,
        relevance.missing_aspects,
    )
    return agent_name, relevance.is_relevant, relevance.covered_aspects, relevance.missing_aspects


async def decompose(query: str, failed_queries: list[tuple[str, str]]) -> list[str]:
    failed_str = "\n".join(f"- {q} ({reason})" for q, reason in failed_queries)
    result = await _run_with_retry(
        lambda: runner.run_llm(
            role=AgentRole.ITERATIVE_EMBEDDING_FILTER_DECOMPOSER,
            prompt_parameters={"query": query, "failed_queries": failed_str},
            output_type=DecomposerResult,
        ),
        label="decomposer",
    )
    queries = result.output.queries
    logging.info("Iterative Embedding Filter: queries=%s", queries)
    return queries


async def query_completeness(
    query: str,
    selected_order: list[str],
    agent_coverage: dict[str, tuple[str, str]],
    failed_queries: list[tuple[str, str]],
    round_label: str,
) -> QueryCompletenessResult:
    coverage_lines = [
        f"Agent {name}: {agent_coverage[name][0]}"
        + (f" - Missing: {agent_coverage[name][1]}" if agent_coverage[name][1] else "")
        for name in selected_order
    ]
    result = await _run_with_retry(
        lambda: runner.run_llm(
            role=AgentRole.ITERATIVE_EMBEDDING_FILTER_QUERY_COMPLETENESS,
            agent_name=round_label,
            prompt_parameters={
                "query": query,
                "agent_coverage_summary": "\n".join(coverage_lines),
                "failed_queries": "\n".join(f"- {q} ({reason})" for q, reason in failed_queries),
                "original_query_found_nothing": "",
            },
            output_type=QueryCompletenessResult,
        ),
        label=f"aggregate/{round_label}",
    )
    return result.output


async def try_decompose(
    query: str,
    failed_queries: list[tuple[str, str]],
    all_tried_queries: set[str],
    round_idx: int,
) -> list[str] | None:
    """Call the decomposer and return new queries, or None if nothing new was produced."""
    new_queries = [q for q in await decompose(query, failed_queries) if q not in all_tried_queries]
    if not new_queries:
        logging.info(
            "Iterative Embedding Filter round %d: decomposer returned no new queries, stopping",
            round_idx,
        )
        return None
    all_tried_queries.update(new_queries)
    return new_queries

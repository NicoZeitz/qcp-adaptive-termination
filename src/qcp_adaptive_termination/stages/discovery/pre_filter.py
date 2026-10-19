"""Embedding-based pre-filter for data agent selection."""

import asyncio
import logging
import resource
import time

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


class ColumnRelevanceResult(BaseModel):
    reasoning: str
    relevant: bool


class IterativeRelevanceResult(BaseModel):
    reasoning: str
    is_relevant: bool
    is_complete: bool
    data_not_available: bool
    missing_sub_query: str


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
    semaphore = asyncio.Semaphore(5)

    async def _check(entry: tuple[str, tuple[float, str, str]]) -> tuple[str, tuple[float, str, str]] | None:
        name, (dist, table_name, col_name) = entry
        async with semaphore:
            while True:
                try:
                    relevant = await _column_relevance_llm(query, name, table_name, col_name, agent_data)
                    break
                except APITimeoutError:
                    logging.warning("Column-relevance LLM [%s] timed out — retrying", name)
                    await asyncio.sleep(2)
        return entry if relevant else None

    results = await asyncio.gather(*[_check(e) for e in sorted_agents])
    kept = [e for e in results if e is not None]
    logging.info("Column-relevance LLM filter: kept %d / %d agents", len(kept), len(sorted_agents))
    return kept


def _build_context_tables_str(
    relevant_agent_names: set[str],
    agent_data: dict[str, AgentRegistryData],
) -> str:
    if not relevant_agent_names:
        return ""
    return "\n\n".join(agent_data[name].technical_description for name in sorted(relevant_agent_names))


async def filter_by_iterative_relevance(
    query: str,
    vector_index: VectorIndex[tuple[str, str, str]],
    agent_data: dict[str, AgentRegistryData],
    agent_map: dict[str, DataAgent],
    max_iterations: int = 50,
) -> list[str]:
    """Iterative embedding-guided relevance filter.

    Starts with the original query, finds the closest agent, asks an LLM whether
    it is relevant and whether the query is fully answerable. If not complete, the
    LLM returns the single most important missing sub-query which is used to find
    the next candidate. Only already-confirmed-relevant agents are skipped; rejected
    agents can be re-evaluated with a new sub-query.
    """
    relevant_agent_names: list[str] = []
    relevant_agent_names_set: set[str] = set()
    current_query: str = query
    unresolved_sub_query_counts: dict[str, int] = {}  # sub-query → number of failed attempts
    evaluated_for_current_query: set[str] = set()

    for iteration in range(max_iterations):
        # Embed the current search query
        (current_query_embedding,) = await embedding.create([current_query])

        # Find the best not-yet-decided agent for the current query
        candidate_agent: str | None = None
        best_dist = float("inf")
        for (agent_name, _table, _col), dist in vector_index.search_all(current_query_embedding):
            if agent_name not in relevant_agent_names_set and agent_name not in evaluated_for_current_query and dist < best_dist:
                best_dist = dist
                candidate_agent = agent_name

        if candidate_agent is None:
            logging.info("Iterative filter: no more candidates after %d iterations", iteration)
            break

        context_str = _build_context_tables_str(relevant_agent_names_set, agent_data)
        schema = agent_data[candidate_agent].technical_description
        if unresolved_sub_query_counts:
            unresolved_str = "\n".join(
                f"- {q} (failed {c} time{'s' if c > 1 else ''})"
                for q, c in unresolved_sub_query_counts.items()
            )
        else:
            unresolved_str = ""

        logging.info(
            "Iterative filter iteration %d: evaluating %s (L2=%.4f, matched query: %r), context has %d relevant agents",
            iteration,
            candidate_agent,
            best_dist,
            current_query,
            len(relevant_agent_names_set),
        )

        while True:
            try:
                result = await runner.run_llm(
                    role=AgentRole.EMBEDDING_ITERATIVE_RELEVANCE,
                    agent_name=candidate_agent,
                    prompt_parameters={
                        "query": query,
                        "context_tables": context_str,
                        "unresolved_sub_queries": unresolved_str,
                        "matched_query": current_query if current_query != query else "",
                        "agent_name": candidate_agent,
                        "schema": schema,
                    },
                    output_type=IterativeRelevanceResult,
                )
                break
            except APITimeoutError:
                logging.warning("Iterative relevance LLM [%s] timed out — retrying", candidate_agent)
                await asyncio.sleep(2)

        logging.info(
            "Iterative filter [%s] is_relevant=%s is_complete=%s data_not_available=%s missing=%s — %s",
            candidate_agent,
            result.is_relevant,
            result.is_complete,
            result.data_not_available,
            result.missing_sub_query,
            result.reasoning,
        )

        evaluated_for_current_query.add(candidate_agent)

        if result.is_relevant:
            relevant_agent_names.append(candidate_agent)
            relevant_agent_names_set.add(candidate_agent)
            unresolved_sub_query_counts.pop(current_query, None)
        else:
            unresolved_sub_query_counts[current_query] = unresolved_sub_query_counts.get(current_query, 0) + 1

        if result.is_complete:
            logging.info("Iterative filter: query complete after %d iterations", iteration + 1)
            break

        if result.data_not_available:
            logging.info("Iterative filter: required data not available, stopping after %d iterations", iteration + 1)
            break

        new_query = result.missing_sub_query if result.missing_sub_query else query
        if new_query == current_query:
            logging.info("Iterative filter: sub-query unchanged, stopping to avoid loop")
            break
        current_query = new_query
        # reset so previously-rejected agents can be re-evaluated for the new sub-query
        evaluated_for_current_query = set(relevant_agent_names_set)

    logging.info(
        "Iterative filter total: %d relevant agents after %d max iterations",
        len(relevant_agent_names),
        max_iterations,
    )
    return relevant_agent_names


async def _filter_embeddings_internal(
    query: str,
    data_agents: list[DataAgent],
    registry_type: AgentRegistryType,
) -> list[DataAgent]:
    (query_embedding,), agent_data = await asyncio.gather(
        embedding.create([query]),
        agent_registry.load(data_agents),
    )

    col_entries: list[tuple[str, str, str]] = []
    col_descriptions: list[str] = []
    for agent in data_agents:
        data = agent_data[agent.name]
        for table_name, table_desc in data.individual_descriptions.items():
            for col_name, col_desc in table_desc.columns.items():
                col_entries.append((agent.name, table_name, col_name))
                col_descriptions.append(f"{table_desc.natural_language_description} {col_desc.natural_language_description}") #  E.g. Sovereign countries in world. Capital city of the country.

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
    t_rank = time.monotonic()
    for (agent_name, table_name, col_name), dist in vector_index.search_all(query_embedding):
        if agent_name not in agent_best_dist or dist < agent_best_dist[agent_name][0]:
            agent_best_dist[agent_name] = (dist, table_name, col_name)

    sorted_agents = sorted(agent_best_dist.items(), key=lambda x: x[1][0])
    obs_context.get_usage_accumulator().ranking_duration_ms += int((time.monotonic() - t_rank) * 1000)
    logging.info("Embedding similarity per agent (L2 distance, ascending):")
    for agent_name, (dist, table_name, col_name) in sorted_agents:
        logging.info("  %s: L2=%.4f (table: %s, col: %s)", agent_name, dist, table_name, col_name)

    relevant_names = await filter_by_iterative_relevance(
        query=query,
        vector_index=vector_index,
        agent_data=agent_data,
        agent_map={a.name: a for a in data_agents},
    )

    logging.info(
        "After iterative relevance: %d / %d agents  rejected=%s",
        len(relevant_names),
        len(data_agents),
        sorted(a.name for a in data_agents if a.name not in set(relevant_names)),
    )
    tracing.append_to_span_output(
        {
            "after_llm_filter_agents": relevant_names,
            "after_llm_filter_count": len(relevant_names),
        }
    )

    agent_map = {a.name: a for a in data_agents}
    filtered = [agent_map[name] for name in relevant_names if name in agent_map]

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
        after_llm_filter_agents=relevant_names,
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


class BoundaryResult(BaseModel):
    reasoning: str
    relevant_agent_names: list[str]


async def filter_embeddings_boundary(
    query: str,
    data_agents: list[DataAgent],
) -> list[DataAgent]:
    """Single-LLM-call boundary finder: ranks agents by embedding similarity, then asks
    the LLM to find the relevance boundary in the ranked list."""
    obs_context.get_usage_accumulator()

    (query_embedding,), agent_data = await asyncio.gather(
        embedding.create([query]),
        agent_registry.load(data_agents),
    )

    col_entries: list[tuple[str, str, str]] = []
    col_descriptions: list[str] = []
    col_desc_map: dict[tuple[str, str, str], str] = {}
    for agent in data_agents:
        data = agent_data[agent.name]
        for table_name, table_desc in data.individual_descriptions.items():
            for col_name, col_desc in table_desc.columns.items():
                entry = (agent.name, table_name, col_name)
                desc = f"{table_desc.natural_language_description} {col_desc.natural_language_description}"
                col_entries.append(entry)
                col_descriptions.append(desc)
                col_desc_map[entry] = desc

    if not col_entries:
        logging.warning("Boundary filter: no column embeddings found, returning all agents unfiltered")
        return data_agents

    col_embeddings = await embedding.create(col_descriptions)
    vector_index: VectorIndex[tuple[str, str, str]] = VectorIndex.build(
        list(zip(col_entries, col_embeddings))
    )

    agent_best: dict[str, tuple[float, str, str]] = {}
    t_rank = time.monotonic()
    for (agent_name, table_name, col_name), dist in vector_index.search_all(query_embedding):
        if agent_name not in agent_best or dist < agent_best[agent_name][0]:
            agent_best[agent_name] = (dist, table_name, col_name)

    sorted_agents = sorted(agent_best.items(), key=lambda x: x[1][0])
    obs_context.get_usage_accumulator().ranking_duration_ms += int((time.monotonic() - t_rank) * 1000)
    logging.info("Boundary filter — embedding similarity per agent (L2 distance, ascending):")
    for agent_name, (dist, table_name, col_name) in sorted_agents:
        logging.info("  %s: L2=%.4f (table: %s, col: %s)", agent_name, dist, table_name, col_name)

    ranked_lines = [
        f"[{rank}] {agent_name} | table: {table_name} · column: {col_name} | L2={dist:.4f}"
        for rank, (agent_name, (dist, table_name, col_name)) in enumerate(sorted_agents, start=1)
    ]

    while True:
        try:
            result = await runner.run_llm(
                role=AgentRole.EMBEDDING_BOUNDARY_FINDER,
                prompt_parameters={
                    "query": query,
                    "ranked_columns": "\n".join(ranked_lines),
                },
                output_type=BoundaryResult,
            )
            break
        except APITimeoutError:
            logging.warning("Boundary-finder LLM timed out — retrying")
            await asyncio.sleep(2)

    logging.info(
        "Boundary filter: relevant_agent_names=%s — %s",
        result.relevant_agent_names,
        result.reasoning,
    )

    relevant_set = set(result.relevant_agent_names)
    agent_map = {a.name: a for a in data_agents}
    filtered = [agent_map[name] for name in result.relevant_agent_names if name in agent_map]

    hooks.emit(
        HookEvent.EMBEDDING_PRE_FILTER_END,
        query=query,
        embedding_filtered_agents=filtered,
        total_agents=len(data_agents),
        l2_scores={name: round(float(dist), 4) for name, (dist, _, _) in agent_best.items()},
        after_llm_filter_agents=result.relevant_agent_names,
    )
    tracing.append_to_span_output(
        {
            "after_llm_filter_agents": result.relevant_agent_names,
            "after_llm_filter_count": len(filtered),
            "filtered_agents": [a.name for a in filtered],
            "l2_scores": {name: round(float(dist), 4) for name, (dist, _, _) in agent_best.items()},
            "best_columns": {name: f"{tbl}.{col}" for name, (_, tbl, col) in agent_best.items()},
        }
    )

    logging.info(
        "Boundary filter: kept %d / %d agents  rejected=%s",
        len(filtered),
        len(data_agents),
        sorted(a.name for a in data_agents if a.name not in relevant_set),
    )
    return filtered


async def filter_embeddings_boundary_schema(
    query: str,
    data_agents: list[DataAgent],
) -> list[DataAgent]:
    """Boundary finder using one embedding per agent (natural_language_description).

    Cheaper than the column-level variant: N vectors instead of N×cols vectors.
    The ranked list passed to the LLM shows each agent's NL database description.
    """
    obs_context.get_usage_accumulator()

    (query_embedding,), agent_data = await asyncio.gather(
        embedding.create([query]),
        agent_registry.load(data_agents),
    )

    agent_descs = [(a.name, agent_data[a.name].natural_language_description) for a in data_agents]
    desc_embeddings = await embedding.create([desc for _, desc in agent_descs])

    t_rank = time.monotonic()
    agent_dists: list[tuple[str, float]] = []
    for (name, _), vec in zip(agent_descs, desc_embeddings):
        dist = float(np.linalg.norm(np.array(query_embedding, dtype="float32") - np.array(vec, dtype="float32")))
        agent_dists.append((name, dist))

    sorted_agents = sorted(agent_dists, key=lambda x: x[1])
    obs_context.get_usage_accumulator().ranking_duration_ms += int((time.monotonic() - t_rank) * 1000)

    logging.info("Schema boundary filter — agents by L2 distance (ascending):")
    for name, dist in sorted_agents:
        logging.info("  %s: L2=%.4f  %s", name, dist, agent_data[name].natural_language_description)

    ranked_lines = [
        f"[{rank}] {name} | L2={dist:.4f} | {agent_data[name].natural_language_description}"
        for rank, (name, dist) in enumerate(sorted_agents, start=1)
    ]

    while True:
        try:
            result = await runner.run_llm(
                role=AgentRole.EMBEDDING_BOUNDARY_FINDER_SCHEMA,
                prompt_parameters={
                    "query": query,
                    "ranked_agents": "\n".join(ranked_lines),
                },
                output_type=BoundaryResult,
            )
            break
        except APITimeoutError:
            logging.warning("Schema boundary-finder LLM timed out — retrying")
            await asyncio.sleep(2)

    logging.info(
        "Schema boundary filter: relevant_agent_names=%s — %s",
        result.relevant_agent_names,
        result.reasoning,
    )

    relevant_set = set(result.relevant_agent_names)
    agent_map = {a.name: a for a in data_agents}
    filtered = [agent_map[name] for name in result.relevant_agent_names if name in agent_map]
    agent_l2 = dict(sorted_agents)

    hooks.emit(
        HookEvent.EMBEDDING_PRE_FILTER_END,
        query=query,
        embedding_filtered_agents=filtered,
        total_agents=len(data_agents),
        l2_scores={name: round(dist, 4) for name, dist in agent_l2.items()},
        after_llm_filter_agents=result.relevant_agent_names,
    )
    tracing.append_to_span_output(
        {
            "after_llm_filter_agents": result.relevant_agent_names,
            "after_llm_filter_count": len(filtered),
            "filtered_agents": [a.name for a in filtered],
            "l2_scores": {name: round(dist, 4) for name, dist in agent_l2.items()},
        }
    )

    logging.info(
        "Schema boundary filter: kept %d / %d agents  rejected=%s",
        len(filtered),
        len(data_agents),
        sorted(a.name for a in data_agents if a.name not in relevant_set),
    )
    return filtered


class RecursiveRelevanceResult(BaseModel):
    reasoning: str
    confirmed_relevant: list[str]
    is_complete: bool
    data_not_available: bool
    sub_queries: list[str]


class _AgentRelevanceResult(BaseModel):
    is_relevant: bool
    reasoning: str


class _AggregateResult(BaseModel):
    is_complete: bool
    data_not_available: bool
    sub_queries: list[str]
    reasoning: str


class _AgentRelevanceResultV3(BaseModel):
    is_relevant: bool
    covered_aspects: str
    missing_aspects: str
    reasoning: str


class _AggregateResultV3(BaseModel):
    is_complete: bool
    data_not_available: bool
    sub_queries: list[str]
    reasoning: str


async def filter_embeddings_boundary_recursive(
    query: str,
    data_agents: list[DataAgent],
    max_rounds: int = 10,
) -> list[DataAgent]:
    """Two-phase filter: boundary pass to get candidates, then iterative relevance
    with one LLM call per round seeing all remaining candidates at once.

    Phase 1: column-embedding → ranked list → boundary LLM → candidate set
    Phase 2: per round, embed current queries in batch → ranked sections per query →
             one LLM call marks multiple agents relevant + returns sub_queries for next round
    """
    obs_context.get_usage_accumulator()

    (query_embedding,), agent_data = await asyncio.gather(
        embedding.create([query]),
        agent_registry.load(data_agents),
    )

    # --- Build column vector index (shared across all rounds) ---
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
        logging.warning("Boundary-recursive filter: no column embeddings, returning all agents")
        return data_agents

    col_embeddings = await embedding.create(col_descriptions)
    vector_index: VectorIndex[tuple[str, str, str]] = VectorIndex.build(
        list(zip(col_entries, col_embeddings))
    )

    # --- Phase 1: Boundary pass ---
    t_rank = time.monotonic()
    agent_best: dict[str, tuple[float, str, str]] = {}
    for (agent_name, table_name, col_name), dist in vector_index.search_all(query_embedding):
        if agent_name not in agent_best or dist < agent_best[agent_name][0]:
            agent_best[agent_name] = (dist, table_name, col_name)
    sorted_p1 = sorted(agent_best.items(), key=lambda x: x[1][0])
    obs_context.get_usage_accumulator().ranking_duration_ms += int((time.monotonic() - t_rank) * 1000)

    ranked_lines_p1 = [
        f"[{rank}] {name} | table: {tbl} · column: {col} | L2={dist:.4f}"
        for rank, (name, (dist, tbl, col)) in enumerate(sorted_p1, start=1)
    ]
    logging.info("Boundary-recursive Phase 1 — ranked agents:")
    for line in ranked_lines_p1:
        logging.info("  %s", line)

    while True:
        try:
            p1_result = await runner.run_llm(
                role=AgentRole.EMBEDDING_BOUNDARY_FINDER,
                prompt_parameters={
                    "query": query,
                    "ranked_columns": "\n".join(ranked_lines_p1),
                },
                output_type=BoundaryResult,
            )
            break
        except APITimeoutError:
            logging.warning("Boundary-recursive Phase 1 LLM timed out — retrying")
            await asyncio.sleep(2)

    boundary_candidate_names: list[str] = [
        n for n in p1_result.relevant_agent_names if n in agent_data
    ]
    logging.info(
        "Boundary-recursive Phase 1: %d boundary candidates: %s",
        len(boundary_candidate_names),
        boundary_candidate_names,
    )

    if not boundary_candidate_names:
        logging.info("Boundary-recursive: no candidates after Phase 1, returning empty")
        return []

    # --- Phase 2: Iterative relevance on boundary candidates ---
    agent_map = {a.name: a for a in data_agents}
    selected_order: list[str] = []
    selected_set: set[str] = set()
    current_queries: list[str] = [query]
    failed_sub_queries: dict[str, int] = {}  # sub_query → number of rounds it found nothing new

    for round_idx in range(max_rounds):
        remaining_set = set(boundary_candidate_names) - selected_set
        if not remaining_set:
            logging.info("Boundary-recursive: all candidates evaluated after round %d", round_idx)
            break

        # Embed all current queries in one batch call
        query_embeddings = await embedding.create(current_queries)

        # Build ranked sections: one section per current_query
        sections: list[str] = []
        t_rank = time.monotonic()
        for cq, cq_emb in zip(current_queries, query_embeddings):
            sub_best: dict[str, tuple[float, str, str]] = {}
            for (agent_name, table, col), dist in vector_index.search_all(cq_emb):
                if agent_name in remaining_set:
                    if agent_name not in sub_best or dist < sub_best[agent_name][0]:
                        sub_best[agent_name] = (dist, table, col)
            sorted_sub = sorted(sub_best.items(), key=lambda x: x[1][0])
            lines = [
                f"[{r}] {n} | table: {t} · col: {c} | L2={d:.4f}\n"
                f"{agent_data[n].technical_description}"
                for r, (n, (d, t, c)) in enumerate(sorted_sub, start=1)
            ]
            sections.append(f'Query: "{cq}"\n' + "\n\n".join(lines))
        obs_context.get_usage_accumulator().ranking_duration_ms += int((time.monotonic() - t_rank) * 1000)

        ranked_sections = "\n---\n".join(sections)
        context_tables = "\n\n".join(
            agent_data[name].technical_description for name in selected_order
        )
        failed_sub_queries_str = "\n".join(
            f"- \"{q}\" (failed {c} time{'s' if c > 1 else ''})"
            for q, c in failed_sub_queries.items()
        )

        logging.info(
            "Boundary-recursive round %d: %d remaining candidates, %d current queries, %d failed sub-queries",
            round_idx,
            len(remaining_set),
            len(current_queries),
            len(failed_sub_queries),
        )

        while True:
            try:
                result = await runner.run_llm(
                    role=AgentRole.EMBEDDING_BOUNDARY_RECURSIVE,
                    prompt_parameters={
                        "query": query,
                        "context_tables": context_tables,
                        "failed_sub_queries": failed_sub_queries_str,
                        "ranked_sections": ranked_sections,
                    },
                    output_type=RecursiveRelevanceResult,
                )
                break
            except APITimeoutError:
                logging.warning("Boundary-recursive round %d LLM timed out — retrying", round_idx)
                await asyncio.sleep(2)

        logging.info(
            "Boundary-recursive round %d: confirmed=%s is_complete=%s data_not_available=%s sub_queries=%s — %s",
            round_idx,
            result.confirmed_relevant,
            result.is_complete,
            result.data_not_available,
            result.sub_queries,
            result.reasoning,
        )

        confirmed_before = len(selected_set)
        for name in result.confirmed_relevant:
            if name in agent_map and name not in selected_set:
                selected_order.append(name)
                selected_set.add(name)

        if result.is_complete or result.data_not_available:
            break

        if not result.sub_queries:
            logging.info("Boundary-recursive: no sub_queries returned, stopping")
            break

        # Track sub-queries that didn't yield new agents
        newly_confirmed = len(selected_set) - confirmed_before
        for sq in current_queries:
            if sq != query and newly_confirmed == 0:
                failed_sub_queries[sq] = failed_sub_queries.get(sq, 0) + 1

        current_queries = result.sub_queries

    filtered = [agent_map[name] for name in selected_order if name in agent_map]
    l2_scores = {name: round(float(dist), 4) for name, (dist, _, _) in agent_best.items()}

    logging.info(
        "Boundary-recursive total: kept %d / %d agents  rejected=%s",
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


async def filter_embeddings_boundary_recursive_v2(
    query: str,
    data_agents: list[DataAgent],
    max_queries: int = 20,
) -> list[DataAgent]:
    """Queue-based two-phase filter.

    A queue of queries to process starts with [query]. Each query goes through:
      1. Boundary Finder: embedding search → ranked list → LLM picks candidates
      2. Parallel per-agent relevance calls for new candidates
      3. Aggregate call: is_complete + sub_queries → enqueue new sub_queries

    Agents already seen (selected or rejected) are not re-evaluated for the same query.
    Rejected agents get a fresh evaluation when a new sub_query triggers a new boundary pass.
    """
    obs_context.get_usage_accumulator()

    agent_data = await agent_registry.load(data_agents)

    # --- Build column vector index (shared across all queries) ---
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
        logging.warning("Boundary-recursive-v2 filter: no column embeddings, returning all agents")
        return data_agents

    col_embeddings = await embedding.create(col_descriptions)
    vector_index: VectorIndex[tuple[str, str, str]] = VectorIndex.build(
        list(zip(col_entries, col_embeddings))
    )

    agent_map = {a.name: a for a in data_agents}
    selected_order: list[str] = []
    selected_set: set[str] = set()
    processed_queries: set[str] = set()
    failed_queries: set[str] = set()  # queries where all candidates were rejected
    query_queue: list[str] = [query]
    agent_best: dict[str, tuple[float, str, str]] = {}  # for tracing l2 scores
    semaphore = asyncio.Semaphore(5)

    async def _check_agent(agent_name: str, matched_sub_query: str) -> tuple[str, bool]:
        schema = agent_data[agent_name].technical_description
        async with semaphore:
            while True:
                try:
                    result = await runner.run_llm(
                        role=AgentRole.EMBEDDING_BOUNDARY_RECURSIVE_AGENT,
                        agent_name=agent_name,
                        prompt_parameters={
                            "query": query,
                            "matched_sub_query": matched_sub_query if matched_sub_query != query else "",
                            "agent_name": agent_name,
                            "schema": schema,
                        },
                        output_type=_AgentRelevanceResult,
                    )
                    break
                except APITimeoutError:
                    logging.warning("Boundary-recursive-v2 agent LLM [%s] timed out — retrying", agent_name)
                    await asyncio.sleep(2)
        logging.info(
            "Boundary-recursive-v2 agent [%s] is_relevant=%s matched_sub_query=%r — %s",
            agent_name,
            result.is_relevant,
            matched_sub_query,
            result.reasoning,
        )
        return agent_name, result.is_relevant

    query_idx = 0
    while query_idx < len(query_queue) and query_idx < max_queries:
        current_query = query_queue[query_idx]

        if current_query in processed_queries:
            logging.info("Boundary-recursive-v2: skipping already-processed query %r", current_query)
            continue
        processed_queries.add(current_query)

        query_idx += 1

        # --- Boundary Finder for current_query ---
        (cq_embedding,) = await embedding.create([current_query])

        t_rank = time.monotonic()
        cq_agent_best: dict[str, tuple[float, str, str]] = {}
        for (agent_name, table_name, col_name), dist in vector_index.search_all(cq_embedding):
            if agent_name not in cq_agent_best or dist < cq_agent_best[agent_name][0]:
                cq_agent_best[agent_name] = (dist, table_name, col_name)
            # track global best for tracing
            if agent_name not in agent_best or dist < agent_best[agent_name][0]:
                agent_best[agent_name] = (dist, table_name, col_name)
        obs_context.get_usage_accumulator().ranking_duration_ms += int((time.monotonic() - t_rank) * 1000)

        sorted_cq = sorted(cq_agent_best.items(), key=lambda x: x[1][0])
        ranked_lines = [
            f"[{rank}] {name} | table: {tbl} · column: {col} | L2={dist:.4f}"
            for rank, (name, (dist, tbl, col)) in enumerate(sorted_cq, start=1)
        ]
        logging.info("Boundary-recursive-v2 query %r — boundary ranked agents:", current_query)
        for line in ranked_lines:
            logging.info("  %s", line)

        while True:
            try:
                boundary_result = await runner.run_llm(
                    role=AgentRole.EMBEDDING_BOUNDARY_FINDER,
                    prompt_parameters={
                        "query": current_query,
                        "ranked_columns": "\n".join(ranked_lines),
                    },
                    output_type=BoundaryResult,
                )
                break
            except APITimeoutError:
                logging.warning("Boundary-recursive-v2 boundary LLM timed out for %r — retrying", current_query)
                await asyncio.sleep(2)

        new_candidates = [
            n for n in boundary_result.relevant_agent_names
            if n in agent_data and n not in selected_set
        ]
        logging.info(
            "Boundary-recursive-v2 query %r: %d new candidates: %s",
            current_query,
            len(new_candidates),
            new_candidates,
        )

        if not new_candidates:
            logging.info("Boundary-recursive-v2: no new candidates for query %r, skipping aggregate", current_query)
            failed_queries.add(current_query)
            continue

        # --- Parallel per-agent relevance calls for new candidates ---
        agent_results = await asyncio.gather(*[
            _check_agent(n, current_query)
            for n in new_candidates
        ])

        newly_confirmed = []
        any_confirmed = False
        for name, is_relevant in agent_results:
            if is_relevant and name not in selected_set:
                selected_order.append(name)
                selected_set.add(name)
                newly_confirmed.append(name)
                any_confirmed = True

        if not any_confirmed:
            failed_queries.add(current_query)

        logging.info(
            "Boundary-recursive-v2 query %r: newly confirmed=%s",
            current_query,
            newly_confirmed,
        )

        # --- Aggregate call: is the query complete, what's missing? ---
        context_tables = "\n\n".join(
            agent_data[name].technical_description for name in selected_order
        )
        failed_queries_str = "\n".join(
            f"- {q}" for q in failed_queries if q != query
        )

        while True:
            try:
                agg_result = await runner.run_llm(
                    role=AgentRole.EMBEDDING_BOUNDARY_RECURSIVE_AGGREGATE,
                    prompt_parameters={
                        "query": query,
                        "context_tables": context_tables,
                        "failed_sub_queries": failed_queries_str,
                    },
                    output_type=_AggregateResult,
                )
                break
            except APITimeoutError:
                logging.warning("Boundary-recursive-v2 aggregate LLM timed out — retrying")
                await asyncio.sleep(2)

        logging.info(
            "Boundary-recursive-v2 query %r aggregate: is_complete=%s data_not_available=%s sub_queries=%s — %s",
            current_query,
            agg_result.is_complete,
            agg_result.data_not_available,
            agg_result.sub_queries,
            agg_result.reasoning,
        )

        if agg_result.is_complete or agg_result.data_not_available:
            logging.info(
                "Boundary-recursive-v2: stopping — is_complete=%s data_not_available=%s after %d queries processed",
                agg_result.is_complete,
                agg_result.data_not_available,
                len(processed_queries),
            )
            break

        for sq in agg_result.sub_queries:
            if sq not in processed_queries and sq not in query_queue:
                query_queue.append(sq)
                logging.info("Boundary-recursive-v2: enqueued sub_query %r", sq)

    if query_idx >= max_queries:
        logging.warning("Boundary-recursive-v2: max_queries=%d reached, stopping", max_queries)

    filtered = [agent_map[name] for name in selected_order if name in agent_map]
    l2_scores = {name: round(float(dist), 4) for name, (dist, _, _) in agent_best.items()}

    logging.info(
        "Boundary-recursive-v2 total: kept %d / %d agents after %d queries  queue=%s  rejected=%s",
        len(filtered),
        len(data_agents),
        len(processed_queries),
        query_queue,
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


async def filter_embeddings_boundary_recursive_v3(
    query: str,
    data_agents: list[DataAgent],
    max_queries: int = 20,
) -> list[DataAgent]:
    """Queue-based two-phase filter with per-agent coverage summaries.

    Like v2, but each per-agent call also returns covered_aspects and missing_aspects.
    The aggregate call receives a compact coverage summary instead of full schemas,
    drastically reducing context size.
    """
    obs_context.get_usage_accumulator()

    agent_data = await agent_registry.load(data_agents)

    # --- Build column vector index (shared across all queries) ---
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
        logging.warning("Boundary-recursive-v3 filter: no column embeddings, returning all agents")
        return data_agents

    col_embeddings = await embedding.create(col_descriptions)
    vector_index: VectorIndex[tuple[str, str, str]] = VectorIndex.build(
        list(zip(col_entries, col_embeddings))
    )

    agent_map = {a.name: a for a in data_agents}
    selected_order: list[str] = []
    selected_set: set[str] = set()
    rejected_set: set[str] = set()
    agent_coverage: dict[str, tuple[str, str]] = {}  # name -> (covered_aspects, missing_aspects)
    processed_queries: set[str] = set()
    failed_queries: set[str] = set()
    original_query_found_nothing: bool = False
    query_queue: list[str] = [query]
    agent_best: dict[str, tuple[float, str, str]] = {}
    semaphore = asyncio.Semaphore(5)

    async def _check_agent(agent_name: str, matched_sub_query: str, q_label: str) -> tuple[str, bool, str, str]:
        schema = agent_data[agent_name].technical_description
        async with semaphore:
            while True:
                try:
                    result = await runner.run_llm(
                        role=AgentRole.EMBEDDING_BOUNDARY_RECURSIVE_V3_AGENT,
                        agent_name=f"{q_label} - {agent_name}",
                        prompt_parameters={
                            "query": query,
                            "matched_sub_query": matched_sub_query if matched_sub_query != query else "",
                            "agent_name": agent_name,
                            "schema": schema,
                        },
                        output_type=_AgentRelevanceResultV3,
                    )
                    break
                except APITimeoutError:
                    logging.warning("Boundary-recursive-v3 agent LLM [%s] timed out — retrying", agent_name)
                    await asyncio.sleep(2)
        logging.info(
            "Boundary-recursive-v3 agent [%s] is_relevant=%s matched_sub_query=%r covered=%r missing=%r — %s",
            agent_name,
            result.is_relevant,
            matched_sub_query,
            result.covered_aspects,
            result.missing_aspects,
            result.reasoning,
        )
        return agent_name, result.is_relevant, result.covered_aspects, result.missing_aspects

    query_idx = 0
    query_num = 0
    while query_idx < len(query_queue) and len(processed_queries) < max_queries:
        current_query = query_queue[query_idx]
        query_idx += 1

        if current_query in processed_queries:
            logging.info("Boundary-recursive-v3: skipping already-processed query %r", current_query)
            continue
        processed_queries.add(current_query)
        query_num += 1
        query_label = f"query {query_num}"

        # --- Boundary Finder for current_query ---
        (cq_embedding,) = await embedding.create([current_query])

        t_rank = time.monotonic()
        cq_agent_best: dict[str, tuple[float, str, str]] = {}
        for (agent_name, table_name, col_name), dist in vector_index.search_all(cq_embedding):
            if agent_name not in cq_agent_best or dist < cq_agent_best[agent_name][0]:
                cq_agent_best[agent_name] = (dist, table_name, col_name)
            if agent_name not in agent_best or dist < agent_best[agent_name][0]:
                agent_best[agent_name] = (dist, table_name, col_name)
        obs_context.get_usage_accumulator().ranking_duration_ms += int((time.monotonic() - t_rank) * 1000)

        sorted_cq = sorted(cq_agent_best.items(), key=lambda x: x[1][0])
        ranked_lines = [
            f"[{rank}] {name} | table: {tbl} · column: {col} | L2={dist:.4f}"
            for rank, (name, (dist, tbl, col)) in enumerate(sorted_cq, start=1)
        ]
        logging.info("Boundary-recursive-v3 query %r — boundary ranked agents:", current_query)
        for line in ranked_lines:
            logging.info("  %s", line)

        while True:
            try:
                boundary_result = await runner.run_llm(
                    role=AgentRole.EMBEDDING_BOUNDARY_FINDER,
                    agent_name=query_label,
                    prompt_parameters={
                        "query": current_query,
                        "ranked_columns": "\n".join(ranked_lines),
                    },
                    output_type=BoundaryResult,
                )
                break
            except APITimeoutError:
                logging.warning("Boundary-recursive-v3 boundary LLM timed out for %r — retrying", current_query)
                await asyncio.sleep(2)

        new_candidates = [
            n for n in boundary_result.relevant_agent_names
            if n in agent_data and n not in selected_set and n not in rejected_set
        ]
        logging.info(
            "Boundary-recursive-v3 query %r: %d new candidates: %s",
            current_query,
            len(new_candidates),
            new_candidates,
        )

        if not new_candidates:
            logging.info("Boundary-recursive-v3: no new candidates for query %r, skipping aggregate", current_query)
            failed_queries.add(current_query)
            continue

        # --- Parallel per-agent relevance calls for new candidates ---
        agent_results = await asyncio.gather(*[
            _check_agent(n, current_query, query_label)
            for n in new_candidates
        ])

        newly_confirmed = []
        any_confirmed = False
        for name, is_relevant, covered, missing in agent_results:
            if is_relevant and name not in selected_set:
                selected_order.append(name)
                selected_set.add(name)
                agent_coverage[name] = (covered, missing)
                newly_confirmed.append(name)
                any_confirmed = True
            elif not is_relevant:
                rejected_set.add(name)

        if not any_confirmed:
            failed_queries.add(current_query)
            if current_query == query:
                original_query_found_nothing = True

        logging.info(
            "Boundary-recursive-v3 query %r: newly confirmed=%s",
            current_query,
            newly_confirmed,
        )

        # --- Aggregate call with compact coverage summary ---
        coverage_lines = [
            f"Agent {name}: covers {agent_coverage[name][0]}"
            + (f" — missing: {agent_coverage[name][1]}" if agent_coverage[name][1] else "")
            for name in selected_order
        ]
        agent_coverage_summary = "\n".join(coverage_lines)
        failed_queries_str = "\n".join(
            f"- {q}" for q in failed_queries
        )
        pending_sub_queries_str = "\n".join(
            f"- {q}" for q in query_queue[query_idx:] if q not in processed_queries
        )

        while True:
            try:
                agg_result = await runner.run_llm(
                    role=AgentRole.EMBEDDING_BOUNDARY_RECURSIVE_V3_AGGREGATE,
                    agent_name=query_label,
                    prompt_parameters={
                        "query": query,
                        "agent_coverage_summary": agent_coverage_summary,
                        "pending_sub_queries": pending_sub_queries_str,
                        "failed_sub_queries": failed_queries_str,
                        "original_query_found_nothing": "true" if original_query_found_nothing else "",
                    },
                    output_type=_AggregateResultV3,
                )
                break
            except APITimeoutError:
                logging.warning("Boundary-recursive-v3 aggregate LLM timed out — retrying")
                await asyncio.sleep(2)

        logging.info(
            "Boundary-recursive-v3 query %r aggregate: is_complete=%s data_not_available=%s sub_queries=%s — %s",
            current_query,
            agg_result.is_complete,
            agg_result.data_not_available,
            agg_result.sub_queries,
            agg_result.reasoning,
        )

        if agg_result.is_complete or agg_result.data_not_available:
            logging.info(
                "Boundary-recursive-v3: stopping — is_complete=%s data_not_available=%s after %d queries processed",
                agg_result.is_complete,
                agg_result.data_not_available,
                len(processed_queries),
            )
            break

        for sq in agg_result.sub_queries:
            if sq not in processed_queries and sq not in query_queue:
                query_queue.append(sq)
                logging.info("Boundary-recursive-v3: enqueued sub_query %r", sq)

    if len(processed_queries) >= max_queries:
        logging.warning("Boundary-recursive-v3: max_queries=%d reached, stopping", max_queries)

    filtered = [agent_map[name] for name in selected_order if name in agent_map]
    l2_scores = {name: round(float(dist), 4) for name, (dist, _, _) in agent_best.items()}

    logging.info(
        "Boundary-recursive-v3 total: kept %d / %d agents after %d queries  queue=%s  rejected=%s",
        len(filtered),
        len(data_agents),
        len(processed_queries),
        query_queue,
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


class _SingleAgentRelevanceResult(BaseModel):
    is_relevant: bool
    reasoning: str


async def _check_single_agent(query: str, agent: DataAgent) -> tuple[DataAgent, bool, str]:
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
        result.reasoning,
    )
    tracing.append_to_span_output(
        {
            f"{agent.name}_relevant": result.is_relevant,
            f"{agent.name}_reasoning": result.reasoning,
        }
    )
    return agent, result.is_relevant, result.reasoning


async def filter_by_llm_baseline(
    query: str,
    data_agents: list[DataAgent],
) -> list[DataAgent]:
    """Baseline: one parallel LLM call per agent schema, no embedding guidance."""
    # Initialise the accumulator *before* spawning child tasks so all children
    # share a pointer to the same object (asyncio.gather copies the context snapshot).
    obs_context.get_usage_accumulator()

    semaphore = asyncio.Semaphore(5)

    async def _check_with_semaphore(agent: DataAgent) -> tuple[DataAgent, bool, str]:
        async with semaphore:
            while True:
                try:
                    return await _check_single_agent(query, agent)
                except APITimeoutError:
                    logging.warning("Baseline filter [%s] timed out — retrying", agent.name)
                    await asyncio.sleep(2)

    results = await asyncio.gather(*[_check_with_semaphore(a) for a in data_agents])
    relevant = [agent for agent, is_relevant, _ in results if is_relevant]
    logging.info("Baseline filter: kept %d / %d agents", len(relevant), len(data_agents))
    return relevant

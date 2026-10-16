"""Embedding-based pre-filter for data agent selection."""

import asyncio
import logging
import resource
import time
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
                f"- \"{q}\" (failed {c} time{'s' if c > 1 else ''})"
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

"""Discovery stage: select and pre-filter the data agents relevant to a query."""

import asyncio
import logging

import qcp_adaptive_termination.agent_registry as agent_registry
import qcp_adaptive_termination.hooks as hooks
from qcp_adaptive_termination.hooks import HookEvent
from qcp_adaptive_termination.models import AgentRegistryType, CacheMode, DataAgent
from qcp_adaptive_termination.models.stages import FilteredRelevantAgents, Stage
from qcp_adaptive_termination.observability import cache, tracing
from qcp_adaptive_termination.stages.discovery.coordinator_filter import filter_coordinator
from qcp_adaptive_termination.stages.discovery.data_agent_filter import filter_data_agent
from qcp_adaptive_termination.stages.discovery.embedding_pre_filter import filter_embedding_pre_filter


@hooks.hook(HookEvent.DISCOVERY_STAGE, stage_context=True)
@cache.cached(mode=CacheMode.WRITE_ONLY)
async def discovery_stage(query: str, data_agents: list[DataAgent]) -> FilteredRelevantAgents:
    """Return the data agents relevant to a query.

    Args:
        query: the user query
        data_agents: all known data agents

    Returns:
        (FilteredRelevantAgents): the queried agents and the relevant subset
    """
    with tracing.own_span(
        Stage.DISCOVERY.label,
        input={
            "query": query,
            "data_agents": data_agents,
        },
    ):
        result = await filter_relevant_agents(
            query=query,
            data_agents=data_agents,
        )

        tracing.append_to_span_output(
            {
                "relevant_data_agents": result.relevant_data_agents,
            }
        )
        return result


@hooks.hook(HookEvent.DISCOVERY_FILTER_RELEVANT_AGENTS)
async def filter_relevant_agents(
    query: str,
    data_agents: list[DataAgent],
    registry_type: AgentRegistryType = AgentRegistryType.NATURAL_LANGUAGE,
    additional_context: str | None = None,
    pre_filter: bool = True,
) -> FilteredRelevantAgents:
    """Select relevant data agents via embedding pre-filter, coordinator, and per-agent checks.

    Args:
        query: the user query
        data_agents: all known data agents
        registry_type: how data agents are described to the LLM
        additional_context: extra context passed to the filters
        pre_filter: whether to apply the embedding pre-filter

    Returns:
        (FilteredRelevantAgents): the relevant data agents
    """
    if pre_filter:
        with tracing.own_span(
            "Embedding Pre-Filter",
            input={
                "query": query,
                "total_agents": len(data_agents),
            },
        ):
            candidate_data_agents = await filter_embedding_pre_filter(query, data_agents, registry_type)
            logging.info("Embedding filter: %d → %d agents", len(data_agents), len(candidate_data_agents))
            tracing.append_to_span_output(
                {
                    "candidates_after_embedding_filter": [a.name for a in candidate_data_agents],
                    "filtered_out_count": len(data_agents) - len(candidate_data_agents),
                }
            )
    else:
        candidate_data_agents = data_agents

    with tracing.own_span(
        "Building Agent Registry",
        input={
            "registry_type": registry_type.value,
            "candidate_data_agents": candidate_data_agents,
        },
    ):
        registry_string = await agent_registry.build_registry_string(candidate_data_agents, registry_type)
        tracing.append_to_span_input({"agent_registry": registry_string})

    with tracing.own_span(
        "Filtering with Coordinator",
        input={
            "query": query,
            "candidate_data_agents": candidate_data_agents,
        },
    ):
        result = await filter_coordinator(query, candidate_data_agents, registry_string, additional_context)

    with tracing.own_span(
        "Filtering with Agents",
        input={"query": query},
    ):
        agents = await asyncio.gather(*[filter_data_agent(query, item, additional_context) for item in result])
        relevant_data_agents = [agent for agent in agents if agent is not None]

    return FilteredRelevantAgents(data_agents=data_agents, relevant_data_agents=relevant_data_agents, query=query)

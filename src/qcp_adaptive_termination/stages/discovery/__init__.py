import asyncio

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
    if pre_filter:
        with tracing.own_span(
            "Embedding Pre-Filter",
            input={
                "query": query,
                "total_agents": len(data_agents),
            },
        ):
            candidate_data_agents = await filter_embedding_pre_filter(query, data_agents, registry_type)
            tracing.indented_logging(f"Embedding filter: {len(data_agents)} → {len(candidate_data_agents)} agents")
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

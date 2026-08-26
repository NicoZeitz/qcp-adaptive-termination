import asyncio

import qcp_adaptive_termination.agent_registry as agent_registry
import qcp_adaptive_termination.hooks as hooks
from qcp_adaptive_termination.hooks import HookEvent
from qcp_adaptive_termination.models import AgentRegistryType, CacheMode, DataAgent
from qcp_adaptive_termination.models.stages import FilteredRelevantAgents
from qcp_adaptive_termination.observability import cache, tracing
from qcp_adaptive_termination.stages.discovery.coordinator_filter import filter_coordinator_with_reasoning
from qcp_adaptive_termination.stages.discovery.data_agent_filter import filter_data_agent_with_reasoning
from qcp_adaptive_termination.stages.discovery.pre_filter import filter_embeddings

DISCOVERY_STAGE_NAME = "Stage 1: Discovery"


@cache.cached(name_prefix=DISCOVERY_STAGE_NAME, mode=CacheMode.WRITE_ONLY)
async def filter_relevant_agents(query: str, data_agents: list[DataAgent]) -> FilteredRelevantAgents:
    with (
        hooks.stage_context(),
        tracing.own_span(
            DISCOVERY_STAGE_NAME,
            input={
                "query": query,
                "data_agents": data_agents,
            },
        ),
    ):
        hooks.emit(HookEvent.FILTER_RELEVANT_AGENTS_START, query=query, data_agents=data_agents)

        result = await hooks.call_override(
            HookEvent.FILTER_RELEVANT_AGENTS_OVERRIDE,
            default=lambda: filter_relevant_agents_internal(
                query=query,
                data_agents=data_agents,
                registry_type=AgentRegistryType.TECHNICAL,
            ),
            query=query,
            data_agents=data_agents,
        )

        tracing.append_to_span_output(
            {
                "relevant_data_agents": result.relevant_data_agents,
            }
        )
        hooks.emit(HookEvent.FILTER_RELEVANT_AGENTS_END, filtered_relevant_agents=result)
        return result


async def filter_relevant_agents_internal(
    query: str,
    data_agents: list[DataAgent],
    registry_type: AgentRegistryType,
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
            candidate_data_agents = await filter_embeddings(query, data_agents, registry_type)
            tracing.indented_logging(f"Embedding filter: {len(data_agents)} → {len(candidate_data_agents)} agents")
            tracing.append_to_span_output(
                {
                    "candidates_after_embedding_filter": [a.name for a in candidate_data_agents],
                    "filtered_out_count": len(data_agents) - len(candidate_data_agents),
                }
            )
    else:
        candidate_data_agents = data_agents

    CHUNK_SIZE = 10

    with tracing.own_span(
        "Filtering with Coordinator",
        input={
            "query": query,
            "candidate_data_agents": candidate_data_agents,
            "chunk_size": CHUNK_SIZE,
        },
    ):
        chunks = [candidate_data_agents[i:i + CHUNK_SIZE] for i in range(0, len(candidate_data_agents), CHUNK_SIZE)]
        result: list[tuple[DataAgent, str]] = []
        for chunk in chunks:
            chunk_registry = await agent_registry.build_registry_string(chunk, registry_type)
            chunk_result = await filter_coordinator_with_reasoning(
                query, chunk, chunk_registry, additional_context
            )
            result.extend(chunk_result)

    with tracing.own_span(
        "Filtering with Agents",
        input={"query": query},
    ):
        semaphore = asyncio.Semaphore(5)

        async def _filtered(item):
            async with semaphore:
                return await filter_data_agent_with_reasoning(query, item, additional_context)

        agents = await asyncio.gather(*[_filtered(item) for item in result])
        relevant_data_agents = [agent for agent in agents if agent is not None]

    return FilteredRelevantAgents(data_agents=data_agents, relevant_data_agents=relevant_data_agents, query=query)

import qcp_adaptive_termination.hooks as hooks
from qcp_adaptive_termination.hooks import HookEvent
from qcp_adaptive_termination.models import AgentRegistryType, CacheMode, DataAgent
from qcp_adaptive_termination.models.stages import FilteredRelevantAgents
from qcp_adaptive_termination.observability import cache, context, tracing
from qcp_adaptive_termination.stages.discovery.pre_filter import filter_by_llm_baseline, iterative_embedding_filter

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
                registry_type=AgentRegistryType.GENERATED_NL,
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
    filter_mode = context.get_config().discovery_filter_mode

    if filter_mode == "iterative_embedding_filter" and pre_filter:
        with tracing.own_span(
            "Iterative Embedding Filter",
            input={
                "query": query,
                "total_agents": len(data_agents),
            },
        ):
            candidate_data_agents = await iterative_embedding_filter(query, data_agents)
            tracing.indented_logging(f"Iterative Embedding Filter: {len(data_agents)} → {len(candidate_data_agents)} agents")
            tracing.append_to_span_output(
                {
                    "candidates_after_iterative_embedding_filter": [a.name for a in candidate_data_agents],
                    "filtered_out_count": len(data_agents) - len(candidate_data_agents),
                }
            )
    elif filter_mode == "llm_baseline":
        with tracing.own_span(
            "Baseline LLM Filter",
            input={
                "query": query,
                "total_agents": len(data_agents),
            },
        ):
            candidate_data_agents = await filter_by_llm_baseline(query, data_agents)
            tracing.indented_logging(f"Baseline LLM filter: {len(data_agents)} → {len(candidate_data_agents)} agents")
            tracing.append_to_span_output(
                {
                    "candidates_after_baseline_filter": [a.name for a in candidate_data_agents],
                    "filtered_out_count": len(data_agents) - len(candidate_data_agents),
                }
            )
    else:
        candidate_data_agents = data_agents

    return FilteredRelevantAgents(data_agents=data_agents, relevant_data_agents=candidate_data_agents, query=query)

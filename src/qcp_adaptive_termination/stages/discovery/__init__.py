import qcp_adaptive_termination.hooks as hooks
from qcp_adaptive_termination.hooks import HookEvent
from qcp_adaptive_termination.models import AgentRegistryType, CacheMode, DataAgent
from qcp_adaptive_termination.models.stages import FilteredRelevantAgents
from qcp_adaptive_termination.observability import cache, context, tracing
from qcp_adaptive_termination.stages.discovery.pre_filter import filter_by_llm_baseline, filter_embeddings, filter_embeddings_boundary, filter_embeddings_boundary_recursive, filter_embeddings_boundary_recursive_v2, filter_embeddings_boundary_recursive_v3, filter_embeddings_boundary_schema

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

    if filter_mode == "embedding_iterative" and pre_filter:
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
    elif filter_mode == "embedding_boundary" and pre_filter:
        with tracing.own_span(
            "Embedding Boundary Filter",
            input={
                "query": query,
                "total_agents": len(data_agents),
            },
        ):
            candidate_data_agents = await filter_embeddings_boundary(query, data_agents)
            tracing.indented_logging(f"Embedding boundary filter: {len(data_agents)} → {len(candidate_data_agents)} agents")
            tracing.append_to_span_output(
                {
                    "candidates_after_boundary_filter": [a.name for a in candidate_data_agents],
                    "filtered_out_count": len(data_agents) - len(candidate_data_agents),
                }
            )
    elif filter_mode == "embedding_boundary_schema" and pre_filter:
        with tracing.own_span(
            "Embedding Boundary Schema Filter",
            input={
                "query": query,
                "total_agents": len(data_agents),
            },
        ):
            candidate_data_agents = await filter_embeddings_boundary_schema(query, data_agents)
            tracing.indented_logging(f"Embedding boundary schema filter: {len(data_agents)} → {len(candidate_data_agents)} agents")
            tracing.append_to_span_output(
                {
                    "candidates_after_boundary_schema_filter": [a.name for a in candidate_data_agents],
                    "filtered_out_count": len(data_agents) - len(candidate_data_agents),
                }
            )
    elif filter_mode == "embedding_boundary_recursive" and pre_filter:
        with tracing.own_span(
            "Embedding Boundary Recursive Filter",
            input={
                "query": query,
                "total_agents": len(data_agents),
            },
        ):
            max_rounds = context.get_config().embedding_boundary_recursive_max_rounds
            candidate_data_agents = await filter_embeddings_boundary_recursive(query, data_agents, max_rounds=max_rounds)
            tracing.indented_logging(f"Embedding boundary recursive filter: {len(data_agents)} → {len(candidate_data_agents)} agents")
            tracing.append_to_span_output(
                {
                    "candidates_after_boundary_recursive_filter": [a.name for a in candidate_data_agents],
                    "filtered_out_count": len(data_agents) - len(candidate_data_agents),
                }
            )
    elif filter_mode == "embedding_boundary_recursive_v2" and pre_filter:
        with tracing.own_span(
            "Embedding Boundary Recursive V2 Filter",
            input={
                "query": query,
                "total_agents": len(data_agents),
            },
        ):
            max_queries = context.get_config().embedding_boundary_recursive_v2_max_queries
            candidate_data_agents = await filter_embeddings_boundary_recursive_v2(query, data_agents, max_queries=max_queries)
            tracing.indented_logging(f"Embedding boundary recursive v2 filter: {len(data_agents)} → {len(candidate_data_agents)} agents")
            tracing.append_to_span_output(
                {
                    "candidates_after_boundary_recursive_v2_filter": [a.name for a in candidate_data_agents],
                    "filtered_out_count": len(data_agents) - len(candidate_data_agents),
                }
            )
    elif filter_mode == "embedding_boundary_recursive_v3" and pre_filter:
        with tracing.own_span(
            "Embedding Boundary Recursive V3 Filter",
            input={
                "query": query,
                "total_agents": len(data_agents),
            },
        ):
            max_queries = context.get_config().embedding_boundary_recursive_v3_max_queries
            candidate_data_agents = await filter_embeddings_boundary_recursive_v3(query, data_agents, max_queries=max_queries)
            tracing.indented_logging(f"Embedding boundary recursive v3 filter: {len(data_agents)} → {len(candidate_data_agents)} agents")
            tracing.append_to_span_output(
                {
                    "candidates_after_boundary_recursive_v3_filter": [a.name for a in candidate_data_agents],
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

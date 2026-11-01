"""Discovery stage: select the data agents relevant to a query via embedding pre-filter."""

import logging

import qcp_adaptive_termination.hooks as hooks
from qcp_adaptive_termination.hooks import HookEvent
from qcp_adaptive_termination.models import CacheMode, DataAgent
from qcp_adaptive_termination.models.stages import FilteredRelevantAgents, Stage
from qcp_adaptive_termination.observability import cache, context, tracing
from qcp_adaptive_termination.stages.discovery.discovery_filter import filter_by_llm_baseline, iterative_embedding_filter


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
        result = await filter_relevant_agents(query=query, data_agents=data_agents)

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
    pre_filter: bool = True,
) -> FilteredRelevantAgents:
    """Select relevant data agents via the configured embedding/LLM pre-filter.

    Args:
        query: the user query
        data_agents: all known data agents
        pre_filter: whether to apply the pre-filter at all

    Returns:
        (FilteredRelevantAgents): the relevant data agents
    """
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
            logging.info(
                "Iterative Embedding Filter: %d → %d agents", len(data_agents), len(candidate_data_agents)
            )
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
            logging.info(
                "Baseline LLM filter: %d → %d agents", len(data_agents), len(candidate_data_agents)
            )
            tracing.append_to_span_output(
                {
                    "candidates_after_baseline_filter": [a.name for a in candidate_data_agents],
                    "filtered_out_count": len(data_agents) - len(candidate_data_agents),
                }
            )
    else:
        candidate_data_agents = data_agents

    return FilteredRelevantAgents(
        data_agents=data_agents, relevant_data_agents=candidate_data_agents, query=query
    )

from enum import Enum
from typing import Any

from common.agent_registry.agent_registry import AgentRegistry, AgentRegistryType
from stages.discovery.filter_with_reasoning import (
    filter_coordinator_with_reasoning,
    filter_data_agent_with_reasoning,
)
from common.span import (
    append_to_span_input,
    append_to_span_output,
    own_custom_span,
)
from common.model.stages.filtered_relevant_agents import FilteredRelevantAgents
from common.model.agent.data_agent import DataAgent
from common.cache import cached

STAGE_NAME = "Stage 1: Discovery"


class FilterType(Enum):
    REASONING = "reasoning"


@cached(name_prefix=STAGE_NAME, mode="WRITE_ONLY")
async def filter_relevant_agents(
    query: str, data_agents: list[DataAgent]
) -> FilteredRelevantAgents:
    with own_custom_span(
        STAGE_NAME,
        input={
            "query": query,
            "data_agents": data_agents,
        },
    ):
        result = await filter_relevant_agents_internal(
            query=query,
            data_agents=data_agents,
            registry_type=AgentRegistryType.NATURAL_LANGUAGE,
            filter_type=FilterType.REASONING,
        )

        append_to_span_output(
            {
                "relevant_data_agents": result.relevant_data_agents,
            }
        )
        return result


async def filter_relevant_agents_internal(
    query: str,
    data_agents: list[DataAgent],
    registry_type: AgentRegistryType,
    filter_type: FilterType,
    additional_context: str | None = None,
) -> FilteredRelevantAgents:
    result: Any

    with own_custom_span(
        "Building Agent Registry",
        input={
            "registry_type": registry_type.value,
            "data_agents": data_agents,
        },
    ):
        agent_registry = await AgentRegistry.build_registry_string(
            data_agents, registry_type
        )
        append_to_span_input({"agent_registry": agent_registry})

    with own_custom_span(
        "Filtering with Coordinator",
        input={
            "query": query,
            "filter_type": filter_type.value,
            "data_agents": data_agents,
        },
    ):
        if filter_type == FilterType.REASONING:
            result = await filter_coordinator_with_reasoning(
                query, data_agents, agent_registry, additional_context
            )
        else:
            raise ValueError(f"Unsupported filter type: {filter_type}")

    with own_custom_span(
        "Filtering with Agents",
        input={"query": query, "filter_type": filter_type.value},
    ):
        relevant_data_agents = []
        for item in result:
            if filter_type == FilterType.REASONING:
                agent = await filter_data_agent_with_reasoning(
                    query, item, additional_context
                )

            if agent is not None:
                relevant_data_agents.append(agent)

    return FilteredRelevantAgents(
        data_agents=data_agents, relevant_data_agents=relevant_data_agents, query=query
    )

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, create_model

import qcp_adaptive_termination.agents.registry as agent_registry
from qcp_adaptive_termination.models.agent import AgentRegistryType
from qcp_adaptive_termination.models.agent import DataAgent
from qcp_adaptive_termination.models.stages.filtered_relevant_agents import (
    FilteredRelevantAgents,
)
from qcp_adaptive_termination.agents.configuration import (
    AgentRole,
    get_agent_configuration,
)
from qcp_adaptive_termination.cache import CacheMode, cached
from qcp_adaptive_termination.llm.runner import run_llm
from qcp_adaptive_termination.tracing.span import (
    append_to_span_input,
    append_to_span_output,
    own_custom_span,
)

STAGE_NAME = "Stage 1: Discovery"


class FilterType(Enum):
    REASONING = "reasoning"


def build_relevance_model(data_agents: list[DataAgent]) -> type[BaseModel]:
    agent_fields: dict[str, Any] = {}
    for agent in data_agents:
        AgentRelevanceModel = create_model(
            f"{agent.name}Relevance",
            is_relevant=(
                bool,
                Field(..., description="Whether this agent is relevant to the query"),
            ),
            reasoning=(
                str,
                Field(
                    ...,
                    description="Explanation of why this agent is or isn't relevant",
                ),
            ),
        )
        agent_fields[agent.name] = (AgentRelevanceModel, Field(...))

    return create_model("RelevanceModel", **agent_fields)


async def filter_coordinator_with_reasoning(
    query: str,
    data_agents: list[DataAgent],
    registry_string: str,
    additional_context: str | None = None,
) -> list[tuple[DataAgent, str]]:
    RelevanceModel = build_relevance_model(data_agents)

    relevances = await run_llm(
        agent_configuration=get_agent_configuration(AgentRole.COORDINATOR_RELEVANCE),
        prompt_name="discovery-coordinator-relevance",
        prompt_parameters={
            "agent_registry": registry_string,
            "query": query,
            "additional_context": additional_context,
        },
        output_type=RelevanceModel,
    )

    relevant_agents_with_reasoning: list[tuple[DataAgent, str]] = []
    for agent in data_agents:
        relevant = getattr(getattr(relevances, agent.name), "is_relevant")
        reasoning = getattr(getattr(relevances, agent.name), "reasoning")

        append_to_span_output(
            {
                f"{agent.name}_relevant": relevant,
                f"{agent.name}_reasoning": reasoning,
            }
        )

        if relevant:
            relevant_agents_with_reasoning.append((agent, reasoning))

    return relevant_agents_with_reasoning


async def filter_data_agent_with_reasoning(
    query: str,
    item: tuple[DataAgent, str],
    additional_context: str | None = None,
) -> DataAgent | None:
    data_agent, reasoning = item
    prompt_parameters: dict[str, str | None] = {
        "query": query,
        "reasoning": reasoning,
        "additional_context": additional_context,
        "m_schema": await agent_registry.build_registry_string(
            data_agent, AgentRegistryType.TECHNICAL, include_capabilities=False
        ),
    }

    is_relevant = await run_llm(
        agent_configuration=get_agent_configuration(
            AgentRole.DATA_AGENT_RELEVANCE, data_agent.name
        ),
        prompt_name="discovery-data-agent-sql-relevance",
        prompt_parameters=prompt_parameters,
        output_type=bool,
    )

    append_to_span_output({f"{data_agent.name}_relevant": is_relevant})
    return data_agent if is_relevant else None


@cached(name_prefix=STAGE_NAME, mode=CacheMode.WRITE_ONLY)
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
        registry_string = await agent_registry.build_registry_string(
            data_agents, registry_type
        )
        append_to_span_input({"agent_registry": registry_string})

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
                query, data_agents, registry_string, additional_context
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

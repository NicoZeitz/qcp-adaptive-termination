import asyncio
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, create_model

import qcp_adaptive_termination.agent_registry as agent_registry
import qcp_adaptive_termination.hooks as hooks
from qcp_adaptive_termination.hooks import HookEvent
from qcp_adaptive_termination.llm import runner
from qcp_adaptive_termination.models import AgentRegistryType, AgentRole, CacheMode, DataAgent
from qcp_adaptive_termination.models.stages import FilteredRelevantAgents
from qcp_adaptive_termination.observability import cache, tracing

DISCOVERY_STAGE_NAME = "Stage 1: Discovery"


class FilterType(StrEnum):
    REASONING = "reasoning"


def _build_relevance_model(data_agents: list[DataAgent]) -> type[BaseModel]:
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


async def _filter_coordinator_with_reasoning(
    query: str,
    data_agents: list[DataAgent],
    registry_string: str,
    additional_context: str | None = None,
) -> list[tuple[DataAgent, str]]:
    RelevanceModel = _build_relevance_model(data_agents)

    relevances = await runner.run_llm(
        role=AgentRole.COORDINATOR_RELEVANCE,
        prompt_parameters={
            "agent_registry": registry_string,
            "query": query,
            "additional_context": additional_context,
        },
        output_type=RelevanceModel,
    )

    relevant_agents_with_reasoning: list[tuple[DataAgent, str]] = []
    for agent in data_agents:
        relevant = getattr(relevances, agent.name).is_relevant
        reasoning = getattr(relevances, agent.name).reasoning

        tracing.append_to_span_output(
            {
                f"{agent.name}_relevant": relevant,
                f"{agent.name}_reasoning": reasoning,
            }
        )

        if relevant:
            relevant_agents_with_reasoning.append((agent, reasoning))

    return relevant_agents_with_reasoning


async def _filter_data_agent_with_reasoning(
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

    is_relevant = await runner.run_llm(
        role=AgentRole.DATA_AGENT_RELEVANCE,
        agent_name=data_agent.name,
        prompt_parameters=prompt_parameters,
        output_type=bool,
    )

    tracing.append_to_span_output({f"{data_agent.name}_relevant": is_relevant})
    return data_agent if is_relevant else None


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

        result = await filter_relevant_agents_internal(
            query=query,
            data_agents=data_agents,
            registry_type=AgentRegistryType.NATURAL_LANGUAGE,
            filter_type=FilterType.REASONING,
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
    filter_type: FilterType,
    additional_context: str | None = None,
) -> FilteredRelevantAgents:
    result: Any

    with tracing.own_span(
        "Building Agent Registry",
        input={
            "registry_type": registry_type.value,
            "data_agents": data_agents,
        },
    ):
        registry_string = await agent_registry.build_registry_string(data_agents, registry_type)
        tracing.append_to_span_input({"agent_registry": registry_string})

    with tracing.own_span(
        "Filtering with Coordinator",
        input={
            "query": query,
            "filter_type": filter_type.value,
            "data_agents": data_agents,
        },
    ):
        if filter_type == FilterType.REASONING:
            result = await _filter_coordinator_with_reasoning(query, data_agents, registry_string, additional_context)
        else:
            raise ValueError(f"Unsupported filter type: {filter_type}")

    with tracing.own_span(
        "Filtering with Agents",
        input={"query": query, "filter_type": filter_type.value},
    ):
        if filter_type == FilterType.REASONING:
            agents = await asyncio.gather(
                *[_filter_data_agent_with_reasoning(query, item, additional_context) for item in result]
            )
        else:
            raise ValueError(f"Unsupported filter type: {filter_type}")
        relevant_data_agents = [agent for agent in agents if agent is not None]

    return FilteredRelevantAgents(data_agents=data_agents, relevant_data_agents=relevant_data_agents, query=query)

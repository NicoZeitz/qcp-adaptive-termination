from typing import Any

from pydantic import BaseModel, Field, create_model

from query_coordinator.build_agent_registry import (
    build_nl_agent_registry,
)
from query_coordinator.query_coordinator import QueryCoordinator
from utils.span import own_custom_span
from data_agent.data_agent import DataAgent
from utils.run_agent import run_agent
from utils.span import append_to_span_input, append_to_span_output


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


async def filter_relevant_agents(
    coordinator: QueryCoordinator, data_agents: list[DataAgent], query: str
) -> list[tuple[DataAgent, str]]:
    with own_custom_span("Coordinator - Filtering Relevant Agents") as span:
        # agent_registry = await build_agent_registry(data_agents, span)
        agent_registry = await build_nl_agent_registry(data_agents, span)
        RelevanceModel = build_relevance_model(data_agents)

        append_to_span_input(span, {"query": query})

        relevances, _ = await run_agent(
            name_postfix="Relevance",
            agent_configuration=coordinator.agent_configuration,
            prompt_name="Query Coordinator/Relevance",
            prompt_parameters={"agent_registry": agent_registry, "query": query},
            output_type=RelevanceModel,
        )

        relevant_agents_with_reasoning: list[tuple[DataAgent, str]] = []
        for agent in data_agents:
            relevant = getattr(getattr(relevances, agent.name), "is_relevant")
            reasoning = getattr(getattr(relevances, agent.name), "reasoning")

            append_to_span_output(
                span,
                {
                    f"{agent.name}_relevant": relevant,
                    f"{agent.name}_reasoning": reasoning,
                },
            )
            if relevant:
                relevant_agents_with_reasoning.append((agent, reasoning))

        return relevant_agents_with_reasoning

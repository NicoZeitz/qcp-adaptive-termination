from typing import Any

from pydantic import BaseModel, Field, create_model

from qtypes.data_agent import DataAgent
from stages.common.agent_configuration import (
    get_coordinator_agent_configuration,
    get_data_agent_agent_configuration,
)
from stages.common.data_agent.data_description import data_agent_data_description
from utils.run_agent import run_llm
from utils.span import append_to_span_output


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
    agent_registry: str,
    additional_context: str | None = None,
) -> list[tuple[DataAgent, str]]:
    RelevanceModel = build_relevance_model(data_agents)

    relevances = await run_llm(
        agent_configuration=get_coordinator_agent_configuration("Relevance"),
        prompt_name="discovery-coordinator-relevance",
        prompt_parameters={
            "agent_registry": agent_registry,
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
    }

    if data_agent.type == "sql":
        prompt_name = "discovery-data-agent-sql-relevance"
        prompt_parameters["m_schema"] = await data_agent_data_description(data_agent)
    else:
        raise ValueError(f"Unsupported agent type: {data_agent.type}")

    is_relevant = await run_llm(
        agent_configuration=get_data_agent_agent_configuration(data_agent, "Relevance"),
        prompt_name=prompt_name,
        prompt_parameters=prompt_parameters,
        output_type=bool,
    )

    append_to_span_output({f"{data_agent.name}_relevant": is_relevant})
    return data_agent if is_relevant else None

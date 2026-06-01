from typing import Any

from pydantic import BaseModel, Field, create_model

from qtypes.data_agent import DataAgent
from stages.common.agent_configuration import (
    get_coordinator_agent_configuration,
    get_data_agent_agent_configuration,
)
from stages.common.data_agent.data_description import data_agent_data_description
from stages.common.data_agent.text_data_agent.get_text import get_text
from utils.preview import CHAR_LIMIT_LONG_LLM, preview_text
from utils.run_agent import run_agent
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
    query: str, data_agents: list[DataAgent], agent_registry: str
) -> list[tuple[DataAgent, str]]:
    RelevanceModel = build_relevance_model(data_agents)

    relevances = await run_agent(
        name_postfix="Relevance",
        agent_configuration=get_coordinator_agent_configuration(),
        prompt_name="Query Coordinator/Relevance",
        prompt_parameters={"agent_registry": agent_registry, "query": query},
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
    query: str, item: tuple[DataAgent, str]
) -> DataAgent | None:
    data_agent, reasoning = item
    prompt_parameters = {
        "query": query,
        "reasoning": reasoning,
    }

    if data_agent.type == "text":
        prompt_name = "Text Data Agent/Relevance"
        prompt_parameters["summary"] = await data_agent_data_description(data_agent)
        prompt_parameters["preview"] = preview_text(
            get_text(data_agent), CHAR_LIMIT_LONG_LLM
        )
    elif data_agent.type == "sql":
        prompt_name = "SQL Data Agent/Relevance"
        prompt_parameters["m_schema"] = await data_agent_data_description(data_agent)
    else:
        raise ValueError(f"Unsupported agent type: {data_agent.type}")

    is_relevant = await run_agent(
        name_postfix="Relevance",
        agent_configuration=get_data_agent_agent_configuration(data_agent),
        prompt_name=prompt_name,
        prompt_parameters=prompt_parameters,
        output_type=bool,
    )

    append_to_span_output({ f"{data_agent.name}_relevant": is_relevant })
    return data_agent if is_relevant else None

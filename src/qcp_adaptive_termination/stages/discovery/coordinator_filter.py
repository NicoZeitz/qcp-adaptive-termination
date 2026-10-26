from typing import Any

from pydantic import BaseModel, Field, create_model

import qcp_adaptive_termination.hooks as hooks
from qcp_adaptive_termination.hooks import HookEvent
from qcp_adaptive_termination.llm import runner
from qcp_adaptive_termination.models import AgentRole, DataAgent
from qcp_adaptive_termination.observability import tracing


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


@hooks.hook(HookEvent.DISCOVERY_COORDINATOR_RELEVANCE)
async def filter_coordinator(
    query: str,
    data_agents: list[DataAgent],
    registry_string: str,
    additional_context: str | None = None,
) -> list[tuple[DataAgent, str]]:
    RelevanceModel = _build_relevance_model(data_agents)

    relevances = await runner.run_llm(
        role=AgentRole.DISCOVERY_COORDINATOR_RELEVANCE,
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

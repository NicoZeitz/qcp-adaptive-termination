from typing import Any

from pydantic import BaseModel, Field, create_model

from query_coordinator.build_agent_registry import build_agent_registry
from query_coordinator.query_coordinator import QueryCoordinator
from utils.span import own_custom_span
from utils.prompt import load_prompt
from data_agent.data_agent import DataAgent
from utils.run_agent import run_agent
from utils.span import append_to_span_input, append_to_span_output


async def filter_relevant_agents(
    coordinator: QueryCoordinator, data_agents: list[DataAgent], query: str
) -> list[DataAgent]:
    with own_custom_span("Coordinator - Filtering Relevant Agents") as span:

        def build_output_model(data_agents: list[DataAgent]) -> type[BaseModel]:
            agent_fields: dict[str, Any] = {
                agent.name: (bool, Field(...)) for agent in data_agents
            }
            return create_model("RelevanceModel", **agent_fields)

        agent_registry = await build_agent_registry(data_agents, span)
        RelevanceModel = build_output_model(data_agents)

        append_to_span_input(span, {"query": query})

        relevances, _ = await run_agent(
            name_postfix="Relevance",
            agent_configuration=coordinator.agent_configuration,
            instructions=load_prompt(
                "Query Coordinator/Relevance/Instructions",
                {"agent_registry": agent_registry},
            ),
            prompt=load_prompt("Query Coordinator/Relevance/Prompt", {"query": query}),
            output_type=RelevanceModel,
        )

        relevant_agents = []
        for agent in data_agents:
            relevant = getattr(relevances, agent.name, False)

            append_to_span_output(span, {f"{agent.name}_relevant": relevant})
            if relevant:
                relevant_agents.append(agent)

        return relevant_agents

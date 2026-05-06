import asyncio

from data_agent.data_agent import DataAgent
from query_coordinator.query_coordinator import QueryCoordinator
from utils.span import append_to_span_input, own_custom_span


async def ask_relevant_agents(
    data_agents: list[DataAgent], query: str
) -> list[DataAgent]:
    with own_custom_span("Coordinator - Agent Relevance Filtering") as span:
        append_to_span_input(span, {"query": query, "data_agents": [agent.name for agent in data_agents]})

        relevant_data_agents = []
        for agent, is_relevant in zip(
            data_agents,
            await asyncio.gather(
                *(agent.is_relevant(query) for agent in data_agents)
            ),
        ):
            append_to_span_input(span, {agent.name: is_relevant})
            if is_relevant:
                relevant_data_agents.append(agent)

        return relevant_data_agents
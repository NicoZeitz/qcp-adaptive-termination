import asyncio

from data_agent.data_agent import DataAgent
from utils.span import append_to_span_input, own_custom_span


async def ask_relevant_agents(
    data_agents_with_reasoning: list[tuple[DataAgent, str]], query: str
) -> list[DataAgent]:
    with own_custom_span("Coordinator - Agent Relevance Filtering") as span:
        append_to_span_input(
            span,
            {
                "query": query,
                "data_agents": [agent.name for agent, _ in data_agents_with_reasoning],
            },
        )

        relevant_data_agents = []
        for agent, is_relevant in zip(
            [agent for agent, _ in data_agents_with_reasoning],
            await asyncio.gather(
                *(
                    agent.is_relevant(query, reasoning)
                    for agent, reasoning in data_agents_with_reasoning
                )
            ),
        ):
            append_to_span_input(span, {agent.name: is_relevant})
            if is_relevant:
                relevant_data_agents.append(agent)

        return relevant_data_agents

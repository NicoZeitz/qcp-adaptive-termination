from qtypes.data_agent import DataAgent
from qtypes.need import NeedType, serialize_need
from stages.common.data_agent.agent_registry import AgentRegistryType
from stages.filter_relevant_agents import FilterType, filter_relevant_agents_internal
from stages.utils.span import (
    append_to_span_input,
    append_to_span_output,
    own_custom_span,
)


async def add_agents_for_needs(
    data_agents: list[DataAgent],
    current_data_agents: list[DataAgent],
    needs: list[NeedType],
    query: str,
):
    current_agent_names = {agent.name for agent in current_data_agents}
    remaining_agents = [
        agent for agent in data_agents if agent.name not in current_agent_names
    ]

    with own_custom_span(
        "Add Agents for Raised Needs",
        input={
            "all_agents_count": len(data_agents),
            "current_agents_count": len(current_data_agents),
            "data_agents": data_agents,
            "current_agents": current_data_agents,
            "remaining_agents": remaining_agents,
            "needs": needs,
            "query": query,
        },
    ):
        if not remaining_agents or not needs:
            append_to_span_output({"newly_added_agents": []})
            return current_data_agents

        needs_context = "\n".join(f"- {serialize_need(need)}" for need in needs)
        extended_query = f"{query}\n\nAdditional requirements:\n{needs_context}"

        append_to_span_input({"extended_query": extended_query})

        result = await filter_relevant_agents_internal(
            query=extended_query,
            data_agents=remaining_agents,
            agent_registry_type=AgentRegistryType.NATURAL_LANGUAGE,
            filter_type=FilterType.REASONING,
        )
        newly_relevant_agents = result.relevant_data_agents

        append_to_span_output({"newly_added_agents": newly_relevant_agents})

        return current_data_agents + newly_relevant_agents

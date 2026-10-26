"""Per-agent relevance filtering of data agents."""

import qcp_adaptive_termination.agent_registry as agent_registry
import qcp_adaptive_termination.hooks as hooks
from qcp_adaptive_termination.hooks import HookEvent
from qcp_adaptive_termination.llm import runner
from qcp_adaptive_termination.models import AgentRegistryType, AgentRole, DataAgent
from qcp_adaptive_termination.observability import tracing


@hooks.hook(HookEvent.DISCOVERY_DATA_AGENT_RELEVANCE)
async def filter_data_agent(
    query: str,
    item: tuple[DataAgent, str],
    additional_context: str | None = None,
) -> DataAgent | None:
    """Ask one data agent whether it is relevant to the query.

    Args:
        query: the user query
        item: the agent and its coordinator reasoning
        additional_context: extra context

    Returns:
        (DataAgent | None): the agent if relevant, else None
    """
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
        role=AgentRole.DISCOVERY_DATA_AGENT_RELEVANCE,
        agent_name=data_agent.name,
        prompt_parameters=prompt_parameters,
        output_type=bool,
    )

    tracing.append_to_span_output({f"{data_agent.name}_relevant": is_relevant})
    return data_agent if is_relevant else None

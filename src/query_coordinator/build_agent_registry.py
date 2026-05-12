import asyncio

from agents import CustomSpanData, Span

from data_agent.data_agent import DataAgent
from utils.preview import CHAR_LIMIT_SPAN, preview_text
from utils.span import append_to_span_input


async def return_agent_info(
    data_agent: DataAgent,
) -> tuple[str, str, list[str]]:
    return data_agent.name, *await asyncio.gather(
        data_agent.data_description(),
        data_agent.query_capabilities(),
    )


async def return_nl_agent_info(
    data_agent: DataAgent,
) -> tuple[str, str, list[str]]:
    return data_agent.name, *await asyncio.gather(
        data_agent.natural_language_description(),
        data_agent.query_capabilities(),
    )


async def build_agent_registry(
    data_agents: list[DataAgent], span: Span[CustomSpanData] | None = None
) -> str:
    agent_infos = await asyncio.gather(
        *(return_agent_info(agent) for agent in data_agents)
    )

    if span is not None:
        append_to_span_input(
            span,
            {
                "data_agents": [
                    {
                        "name": name,
                        "description": preview_text(description, CHAR_LIMIT_SPAN),
                        "capabilities": [
                            preview_text(capability, CHAR_LIMIT_SPAN)
                            for capability in capabilities
                        ],
                    }
                    for name, description, capabilities in agent_infos
                ]
            },
        )

    return "\n\n".join(
        f"【Agent Name】 {name}\n{description}\n【Capabilities】 {', '.join(capabilities)}"
        for name, description, capabilities in agent_infos
    )


async def build_nl_agent_registry(
    data_agents: list[DataAgent], span: Span[CustomSpanData] | None = None
) -> str:
    agent_infos = await asyncio.gather(
        *(return_nl_agent_info(agent) for agent in data_agents)
    )

    if span is not None:
        append_to_span_input(
            span,
            {
                "data_agents": [
                    {
                        "name": name,
                        "description": preview_text(description, CHAR_LIMIT_SPAN),
                        "capabilities": [
                            preview_text(capability, CHAR_LIMIT_SPAN)
                            for capability in capabilities
                        ],
                    }
                    for name, description, capabilities in agent_infos
                ]
            },
        )

    return "\n\n".join(
        f"【Agent Name】 {name}\n{description}\n【Capabilities】 {', '.join(capabilities)}"
        for name, description, capabilities in agent_infos
    )

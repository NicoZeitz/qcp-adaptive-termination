from openai.types.shared import Reasoning

from qtypes.agent_configuration import AgentConfiguration
from qtypes.data_agent import DataAgent

coordinator_reasoning_mapping: dict[str, Reasoning] = {
    "Relevance": Reasoning(effort="low", summary="concise"),
    "Draft Query Plan": Reasoning(effort="high", summary="concise"),
    "Code": Reasoning(effort="medium", summary="concise"),
    "Replan Query Plan": Reasoning(effort="high", summary="concise"),
}

data_agent_reasoning_mapping: dict[str, Reasoning] = {
    "Schema Description": Reasoning(effort="minimal", summary="concise"),
    "Relevance": Reasoning(effort="low", summary="concise"),
    "Summary": Reasoning(effort="low", summary="concise"),
    "Code": Reasoning(effort="medium", summary="concise"),
}

coordinator_agent_model = "gpt-5"
data_agent_model = "gpt-5"


def set_agent_models(coordinator_model: str | None, data_model: str | None):
    global coordinator_agent_model, data_agent_model
    if coordinator_model is not None:
        coordinator_agent_model = coordinator_model
    if data_model is not None:
        data_agent_model = data_model


def get_coordinator_agent_model() -> str:
    return coordinator_agent_model


def get_data_agent_model() -> str:
    return data_agent_model


def get_coordinator_agent_configuration(name_postfix: str) -> AgentConfiguration:
    reasoning = coordinator_reasoning_mapping.get(
        name_postfix, Reasoning(effort="high", summary="concise")
    )

    return AgentConfiguration(
        name=f"Query Coordinator - {name_postfix}",
        model=get_coordinator_agent_model(),
        reasoning=reasoning,
    )


def get_data_agent_agent_configuration(
    data_agent: DataAgent, name_postfix: str
) -> AgentConfiguration:
    reasoning = data_agent_reasoning_mapping.get(
        name_postfix, Reasoning(effort="medium", summary="concise")
    )

    return AgentConfiguration(
        name=f"{data_agent.name} - {name_postfix}",
        model=get_data_agent_model(),
        reasoning=reasoning,
    )

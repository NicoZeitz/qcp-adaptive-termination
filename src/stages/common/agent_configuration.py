from openai.types.shared import Reasoning, ReasoningEffort

from qtypes.agent_configuration import AgentConfiguration
from qtypes.data_agent import DataAgent

coordinator_reasoning_mapping: dict[str, ReasoningEffort] = {
    "Relevance": "low",
    "Draft Query Plan": "high",
    "Code": "medium",
    "Replan Query Plan": "high",
}

data_agent_reasoning_mapping: dict[str, ReasoningEffort] = {
    "Schema Description": "minimal",
    "Relevance": "low",
    "Summary": "low",
    "Code": "medium",
}


def get_coordinator_agent_configuration(name_postfix: str) -> AgentConfiguration:
    reasoning_effort = coordinator_reasoning_mapping.get(name_postfix, "high")

    return AgentConfiguration(
        name=f"Query Coordinator - {name_postfix}",
        model="gpt-5",
        reasoning=Reasoning(effort=reasoning_effort),
    )


def get_data_agent_agent_configuration(
    data_agent: DataAgent, name_postfix: str
) -> AgentConfiguration:
    reasoning_effort = data_agent_reasoning_mapping.get(name_postfix, "medium")

    return AgentConfiguration(
        name=data_agent.name,
        model="gpt-5",
        reasoning=Reasoning(effort=reasoning_effort or "medium"),
    )

from openai.types.shared import Reasoning, ReasoningEffort

from qtypes.agent_configuration import AgentConfiguration
from qtypes.data_agent import DataAgent

coordinator_reasoning_mapping: dict[str, Reasoning] = {
    "Relevance": Reasoning(effort="low"),
    "Draft Query Plan": Reasoning(effort="high"),
    "Code": Reasoning(effort="medium"),
    "Replan Query Plan": Reasoning(effort="high"),
}

data_agent_reasoning_mapping: dict[str, Reasoning] = {
    "Schema Description": Reasoning(effort="minimal"),
    "Relevance": Reasoning(effort="low", generate_summary="concise"),
    "Summary": Reasoning(effort="low"),
    "Code": Reasoning(effort="medium"),
}


def get_coordinator_agent_configuration(name_postfix: str) -> AgentConfiguration:
    reasoning = coordinator_reasoning_mapping.get(name_postfix, Reasoning(effort="high"))

    return AgentConfiguration(
        name=f"Query Coordinator - {name_postfix}",
        model="gpt-5",
        reasoning=reasoning,
    )


def get_data_agent_agent_configuration(
    data_agent: DataAgent, name_postfix: str
) -> AgentConfiguration:
    reasoning = data_agent_reasoning_mapping.get(name_postfix, Reasoning(effort="medium"))

    return AgentConfiguration(
        name=data_agent.name,
        model="gpt-5",
        reasoning=reasoning,
    )

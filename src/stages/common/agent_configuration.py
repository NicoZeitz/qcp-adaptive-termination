from typing import Optional

from openai.types.shared import Reasoning, ReasoningEffort

from qtypes.agent_configuration import AgentConfiguration
from qtypes.data_agent import DataAgent


def get_coordinator_agent_configuration(reasoning_effort: Optional[ReasoningEffort] = None) -> AgentConfiguration:
    return AgentConfiguration(
        name="Query Coordinator",
        model="gpt-5",
        reasoning=Reasoning(effort=reasoning_effort or "high"),
    )


def get_data_agent_agent_configuration(data_agent: DataAgent, reasoning_effort: Optional[ReasoningEffort] = None) -> AgentConfiguration:
    return AgentConfiguration(
        name=data_agent.name,
        model="gpt-5",
        reasoning=Reasoning(effort=reasoning_effort or "medium"),
    )

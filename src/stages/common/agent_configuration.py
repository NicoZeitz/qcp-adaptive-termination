from openai.types.shared import Reasoning

from qtypes.agent_configuration import AgentConfiguration
from qtypes.data_agent import DataAgent


def get_coordinator_agent_configuration() -> AgentConfiguration:
    return AgentConfiguration(
        name="Query Coordinator",
        model="gpt-5",
        reasoning=Reasoning(effort="high"),
    )


def get_data_agent_agent_configuration(data_agent: DataAgent) -> AgentConfiguration:
    return AgentConfiguration(
        name=data_agent.name,
        model="gpt-5",
        reasoning=Reasoning(effort="medium"),
    )

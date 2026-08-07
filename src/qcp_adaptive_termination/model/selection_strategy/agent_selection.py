from qcp_adaptive_termination.model.agent.data_agent import DataAgent


from dataclasses import dataclass


@dataclass
class AgentSelection:
    agent: DataAgent
    score: float | None = None

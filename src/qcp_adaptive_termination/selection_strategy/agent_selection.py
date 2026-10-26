"""Model for a selected agent and its score."""

from dataclasses import dataclass

from qcp_adaptive_termination.models import DataAgent


@dataclass(slots=True)
class AgentSelection:
    """A selected agent and its score."""

    agent: DataAgent
    """The selected data agent"""
    score: float | None = None
    """Score assigned to the agent, if any"""

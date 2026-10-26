from dataclasses import dataclass

from qcp_adaptive_termination.models import DataAgent


@dataclass(slots=True)
class AgentSelection:
    """A selected agent and its score."""

    agent: DataAgent
    score: float | None = None

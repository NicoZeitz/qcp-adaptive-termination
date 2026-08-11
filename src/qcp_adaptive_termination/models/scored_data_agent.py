from dataclasses import dataclass

from qcp_adaptive_termination.models.agent import DataAgent


@dataclass(slots=True)
class ScoredDataAgent:
    agent: DataAgent
    score: float | None = None

"""Model pairing a data agent with its relevance score."""

from dataclasses import dataclass

from qcp_adaptive_termination.models.agent import DataAgent


@dataclass(slots=True)
class ScoredDataAgent:
    """A data agent paired with its relevance score."""

    agent: DataAgent
    """The data agent being scored"""
    score: float | None = None
    """Relevance score, if computed"""

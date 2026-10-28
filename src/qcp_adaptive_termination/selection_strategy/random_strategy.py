"""Random static selection strategy."""

import random
from dataclasses import dataclass

from qcp_adaptive_termination.models import DataAgent, ScoredDataAgent
from qcp_adaptive_termination.selection_strategy.strategy import (
    StaticSelectionStrategy,
)


@dataclass(slots=True)
class RandomStrategy(StaticSelectionStrategy[float]):
    """Ranks agents in a seeded random order (independent of the query)."""

    seed: int = 42
    """Seed for the random ordering"""

    async def precompute(
        self,
        agents: list[DataAgent],
        table_per_agent: dict[str, str] | None,
        column_per_agent: dict[str, str] | None,
        query: str = "",
    ) -> dict[str, float]:
        """Precompute a per-agent seeded random ranking score."""
        names = [a.name for a in agents]
        random.Random(self.seed).shuffle(names)
        return {name: float(i) for i, name in enumerate(names)}

    def rank(
        self,
        remaining: list[ScoredDataAgent],
        precomputed: dict[str, float],
        state: None,
    ) -> list[ScoredDataAgent]:
        """Return agents in precomputed shuffle order without sorting."""
        by_name = {selection.agent.name: selection for selection in remaining}
        ranked = []
        for name in precomputed:
            if selection := by_name.get(name):
                selection.score = precomputed[name]
                ranked.append(selection)
        return ranked

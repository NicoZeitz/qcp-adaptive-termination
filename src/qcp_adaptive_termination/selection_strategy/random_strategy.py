"""Random static selection strategy."""

import random
from dataclasses import dataclass

from qcp_adaptive_termination.models import DataAgent
from qcp_adaptive_termination.selection_strategy.strategy import (
    StaticSelectionStrategy,
)


@dataclass(slots=True)
class RandomStrategy(StaticSelectionStrategy[float]):
    """Ranks agents in a seeded random order (independent of the query)."""

    ascending: bool = True
    """Whether to rank ascending (lowest score first)"""
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

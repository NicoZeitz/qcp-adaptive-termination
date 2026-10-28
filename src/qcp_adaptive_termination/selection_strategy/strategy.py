"""Abstract base classes for agent selection strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import cast

from qcp_adaptive_termination.models import DataAgent, ScoredDataAgent


@dataclass(slots=True)
class SelectionStrategy[PrecomputedState, RunningState](ABC):
    """Stateful ranking strategy for dynamic agent selection."""

    def needs_table(self) -> bool:
        """Return True if this strategy requires a table name to be resolved per agent."""
        return self.needs_column()

    def needs_column(self) -> bool:
        """Return True if this strategy requires a column name to be resolved per agent."""
        return False

    @abstractmethod
    async def precompute(
        self,
        agents: list[DataAgent],
        table_per_agent: dict[str, str] | None,
        column_per_agent: dict[str, str] | None,
        query: str = "",
    ) -> PrecomputedState:
        """Return all data precomputed for this strategy run."""
        raise NotImplementedError

    @abstractmethod
    def init_state(self) -> RunningState:
        """Return the initial running state (empty sketch, empty set, …)."""
        raise NotImplementedError

    @abstractmethod
    def rank(
        self,
        remaining: list[ScoredDataAgent],
        precomputed: PrecomputedState,
        state: RunningState,
    ) -> list[ScoredDataAgent]:
        """Return remaining agents sorted best-first given the current running state."""
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        state: RunningState,
        values: list[str],
        agent_names: list[str],
        precomputed: PrecomputedState,
    ) -> RunningState:
        """Fold the column values seen so far into the running state and return it."""
        raise NotImplementedError


@dataclass(slots=True)
class StaticSelectionStrategy[PrecomputedScoredState: float](
    SelectionStrategy[dict[str, PrecomputedScoredState], None]
):
    """Base for strategies whose ranking does not change between iterations."""

    # Whether to rank ascending (lowest score first).
    ascending: bool = False

    @abstractmethod
    async def precompute(
        self,
        agents: list[DataAgent],
        table_per_agent: dict[str, str] | None,
        column_per_agent: dict[str, str] | None,
        query: str = "",
    ) -> dict[str, PrecomputedScoredState]:
        """Return a per-agent mapping of static precomputed scores."""
        raise NotImplementedError

    def init_state(self):
        """Return None, since static strategies keep no running state."""
        return None

    def rank(
        self,
        remaining: list[ScoredDataAgent],
        precomputed: dict[str, PrecomputedScoredState],
        state: None,
    ) -> list[ScoredDataAgent]:
        """Score and sort remaining agents by their precomputed score."""
        for s in remaining:
            s.score = precomputed.get(s.agent.name, 0.0)

        return sorted(
            remaining,
            key=lambda s: cast(float, s.score),
            reverse=not self.ascending,
        )

    def update(
        self,
        state: None,
        values: list[str],
        agent_names: list[str] | None = None,
        precomputed: dict[str, PrecomputedScoredState] | None = None,
    ):
        """Return None, since static strategies fold no state."""
        return None

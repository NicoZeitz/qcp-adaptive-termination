from abc import ABC, abstractmethod
from typing import cast

from qcp_adaptive_termination.models import DataAgent, ScoredDataAgent


class SelectionStrategy[PrecomputedState, RunningState](ABC):
    """
    Stateful ranking strategy for dynamic agent selection.

    Lifecycle per multi-agent step:
      1. precompute(agents)  — build per-agent sketches/scores once
      2. rank(remaining, state)  — score remaining agents given current state
      3. update(state, values)  — fold newly-seen column values into state
         (repeat 2-3 until done)
    """

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
    ) -> dict[str, PrecomputedState]:
        """Return a per-agent-name mapping of precomputed data (sketches, scores, …)."""
        raise NotImplementedError

    @abstractmethod
    def init_state(self) -> RunningState:
        """Return the initial running state (empty sketch, empty set, …)."""
        raise NotImplementedError

    @abstractmethod
    def rank(
        self,
        remaining: list[ScoredDataAgent],
        precomputed: dict[str, PrecomputedState],
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
        precomputed: dict[str, PrecomputedState],
    ) -> RunningState:
        """Fold the column values seen so far into the running state and return it."""
        raise NotImplementedError


class StaticSelectionStrategy[PrecomputedScoredState: float](SelectionStrategy[PrecomputedScoredState, None]):
    """
    Base for strategies whose ranking does not change between iterations
    (COUNT, DISTINCT_COUNT, statistics-based, embedding).

    Subclasses only need to implement precompute(). Set ascending=True when
    lower scores are better (e.g. L2 distance). rank() is a no-op for update().
    """

    ascending: bool = False

    @abstractmethod
    async def precompute(
        self,
        agents: list[DataAgent],
        table_per_agent: dict[str, str] | None,
        column_per_agent: dict[str, str] | None,
        query: str = "",
    ) -> dict[str, PrecomputedScoredState]:
        raise NotImplementedError

    def init_state(self) -> None:
        return None

    def rank(
        self,
        remaining: list[ScoredDataAgent],
        precomputed: dict[str, PrecomputedScoredState],
        state: None,
    ) -> list[ScoredDataAgent]:
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
    ) -> None:
        return None

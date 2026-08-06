from abc import ABC, abstractmethod
from typing import Generic, Protocol, TypeVar, cast

from model.agent.data_agent import DataAgent
from model.selection_strategy.agent_selection import AgentSelection

P = TypeVar("P")  # precomputed value type (per-agent mapping value)
S = TypeVar("S")  # running state type
PS = TypeVar("PS", bound=float)  # numeric score type for static strategies


class SelectionStrategy(Protocol[P, S]):
    """
    Stateful ranking strategy for dynamic agent selection.

    Lifecycle per multi-agent step:
      1. precompute(agents)  — build per-agent sketches/scores once
      2. rank(remaining, state)  — score remaining agents given current state
      3. update(state, values)  — fold newly-seen column values into state
         (repeat 2-3 until done)
    """

    async def precompute(
        self,
        agents: list[DataAgent],
        table_per_agent: dict[str, str] | None,
        column_per_agent: dict[str, str] | None,
        query: str = "",
    ) -> dict[str, P]:
        """Return a per-agent-name mapping of precomputed data (sketches, scores, …)."""
        ...

    def init_state(self) -> S:
        """Return the initial running state (empty sketch, empty set, …)."""
        ...

    def rank(
        self,
        remaining: list[AgentSelection],
        precomputed: dict[str, P],
        state: S,
    ) -> list[AgentSelection]:
        """Return remaining agents sorted best-first given the current running state."""
        ...

    def update(
        self,
        state: S,
        values: list[str],
        agent_names: list[str],
        precomputed: dict[str, P],
    ) -> S:
        """Fold the column values seen so far into the running state and return it."""
        ...


class StaticSelectionStrategy(ABC, Generic[PS]):
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
    ) -> dict[str, PS]:
        raise NotImplementedError

    def init_state(self) -> None:
        return None

    def rank(
        self,
        remaining: list[AgentSelection],
        precomputed: dict[str, PS],
        state: None,
    ) -> list[AgentSelection]:
        for s in remaining:
            s.score = precomputed.get(s.agent.name, 0.0)

        return sorted(
            remaining,
            key=lambda s: cast(float, s.score),
            reverse=not self.ascending,
        )

    def update(self, state: None, values: list[str]) -> None:
        return None

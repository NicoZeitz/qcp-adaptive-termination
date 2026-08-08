"""
Agent selection strategies and `AgentSelector`.

Implements multiple ranking strategies (sequential, embedding-based,
count/distinct-count, sketch-based, posterior, null-coverage, value-range)
and `AgentSelector`, which precomputes strategy data and ranks agents at
each iteration of the multi-agent loop.
"""

from typing import Any

from qcp_adaptive_termination.models import (
    AgentSelection,
    AlignedResultList,
    Config,
    DataAgent,
    SelectionType,
)
from qcp_adaptive_termination.selection_strategy.counts import (
    CountStrategy,
    DistinctCountStrategy,
)
from qcp_adaptive_termination.selection_strategy.embedding_strategy import (
    EmbeddingStrategy,
)
from qcp_adaptive_termination.selection_strategy.posterior import PosteriorStrategy
from qcp_adaptive_termination.selection_strategy.resolve import (
    resolve_column,
    resolve_table,
)
from qcp_adaptive_termination.selection_strategy.sequential import SequentialStrategy
from qcp_adaptive_termination.selection_strategy.sketches import (
    HllSketchStrategy,
    MinHashStrategy,
    ThetaSketchStrategy,
)
from qcp_adaptive_termination.selection_strategy.strategy import SelectionStrategy


def needs_table(selection_type: SelectionType) -> bool:
    return selection_type not in (SelectionType.SEQUENTIAL, SelectionType.EMBEDDING)


def needs_column(selection_type: SelectionType) -> bool:
    return selection_type in (
        SelectionType.DISTINCT_COUNT,
        SelectionType.HLL_SKETCH,
        SelectionType.HLL_SKETCH_UCB,
        SelectionType.THETA_SKETCH,
        SelectionType.THETA_SKETCH_UCB,
        SelectionType.MIN_HASH,
        SelectionType.POSTERIOR,
    )


def build_strategy(selection_type: SelectionType) -> SelectionStrategy:
    if selection_type == SelectionType.SEQUENTIAL:
        return SequentialStrategy()
    elif selection_type == SelectionType.EMBEDDING:
        return EmbeddingStrategy()
    elif selection_type == SelectionType.COUNT:
        return CountStrategy()
    elif selection_type == SelectionType.DISTINCT_COUNT:
        return DistinctCountStrategy()
    elif selection_type == SelectionType.HLL_SKETCH:
        return HllSketchStrategy()
    elif selection_type == SelectionType.HLL_SKETCH_UCB:
        return HllSketchStrategy(use_ucb=True)
    elif selection_type == SelectionType.THETA_SKETCH:
        return ThetaSketchStrategy()
    elif selection_type == SelectionType.THETA_SKETCH_UCB:
        return ThetaSketchStrategy(use_ucb=True)
    elif selection_type == SelectionType.MIN_HASH:
        return MinHashStrategy()
    elif selection_type == SelectionType.POSTERIOR:
        return PosteriorStrategy()
    else:
        raise ValueError(f"Unsupported SelectionType: {selection_type}")


class AgentSelector:
    """Owns the precomputed data and running state for one multi-agent step.

    Usage:
        selector = await AgentSelector.create(agents, selection_type, query, ...)
        ranked = selector.rank(remaining)
        selector.update(queried)
    """

    def __init__(
        self,
        strategy: SelectionStrategy,
        precomputed: dict[str, Any],
        state: Any,
        selections: list[AgentSelection],
    ) -> None:
        self._strategy = strategy
        self._precomputed = precomputed
        self._state = state
        self.selections = selections

    @classmethod
    async def create(
        cls,
        agents: list[DataAgent],
        query: str,
    ) -> "AgentSelector":
        selection_type = Config.get().selection_type
        strategy = build_strategy(selection_type)

        table_per_agent: dict[str, str] | None = None
        column_per_agent: dict[str, str] | None = None

        if needs_table(selection_type):
            table_per_agent = await resolve_table(agents, query)
        if needs_column(selection_type):
            assert table_per_agent is not None
            column_per_agent = await resolve_column(agents, table_per_agent, query)

        precomputed = await strategy.precompute(
            agents, table_per_agent, column_per_agent, query
        )
        state = strategy.init_state()
        selections = [AgentSelection(agent=a, score=None) for a in agents]
        return cls(strategy, precomputed, state, selections)

    def rank(self, remaining: list[AgentSelection]) -> list[AgentSelection]:
        return self._strategy.rank(remaining, self._precomputed, self._state)

    def update(
        self, result_list: AlignedResultList, agent_names: list[str] | None = None
    ) -> None:
        col = (
            result_list.schema_metadata.value_column
            or result_list.schema_metadata.keys[0]
        )
        df = result_list.data
        values = df[col].dropna().astype(str).tolist()
        self._state = self._strategy.update(
            self._state,
            values,
            agent_names=agent_names or [],
            precomputed=self._precomputed,
        )

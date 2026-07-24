from typing import Any

from qtypes.aligned_result_list import AlignedResultList
from qtypes.data_agent import DataAgent
from qtypes.execution_config import ExecutionConfig
from qtypes.selection_type import SelectionType
from stages.common.data_agent.agent_registry import AgentRegistryType
from stages.execution.selection.agent_selection import AgentSelection
from stages.execution.selection.counts import CountStrategy, DistinctCountStrategy
from stages.execution.selection.embeddings import EmbeddingStrategy
from stages.execution.selection.sequential import SequentialStrategy
from stages.execution.selection.sketches import (
    HllSketchStrategy,
    MinHashStrategy,
    ThetaSketchStrategy,
)
from stages.execution.selection.statistics import (
    NullCoverageStrategy,
    ValueRangeStrategy,
)
from stages.execution.selection.strategy import SelectionStrategy


def _requires_table(selection_type: SelectionType, table_name: str | None) -> None:
    if table_name is None:
        raise ValueError(f"SelectionType.{selection_type.value} requires table_name")


def _requires_column(
    selection_type: SelectionType, table_name: str | None, column_name: str | None
) -> None:
    if table_name is None or column_name is None:
        raise ValueError(
            f"SelectionType.{selection_type.value} requires table_name and column_name"
        )


def _build_strategy(
    selection_type: SelectionType,
    table_name: str | None,
    column_name: str | None,
) -> SelectionStrategy:
    if selection_type == SelectionType.SEQUENTIAL:
        return SequentialStrategy()
    elif selection_type == SelectionType.EMBEDDING:
        return EmbeddingStrategy(registry_type=AgentRegistryType.NATURAL_LANGUAGE)
    elif selection_type == SelectionType.EMBEDDING_MSCHEMA:
        return EmbeddingStrategy(registry_type=AgentRegistryType.TECHNICAL)
    elif selection_type == SelectionType.COUNT:
        _requires_table(selection_type, table_name)
        return CountStrategy()
    elif selection_type == SelectionType.DISTINCT_COUNT:
        _requires_column(selection_type, table_name, column_name)
        return DistinctCountStrategy()
    elif selection_type == SelectionType.HLL_SKETCH:
        _requires_column(selection_type, table_name, column_name)
        return HllSketchStrategy()
    elif selection_type == SelectionType.HLL_SKETCH_UCB:
        _requires_column(selection_type, table_name, column_name)
        return HllSketchStrategy(use_ucb=True)
    elif selection_type == SelectionType.THETA_SKETCH:
        _requires_column(selection_type, table_name, column_name)
        return ThetaSketchStrategy()
    elif selection_type == SelectionType.THETA_SKETCH_UCB:
        _requires_column(selection_type, table_name, column_name)
        return ThetaSketchStrategy(use_ucb=True)
    elif selection_type == SelectionType.MIN_HASH:
        _requires_column(selection_type, table_name, column_name)
        return MinHashStrategy()
    elif selection_type == SelectionType.NULL_COVERAGE:
        _requires_column(selection_type, table_name, column_name)
        return NullCoverageStrategy()
    elif selection_type == SelectionType.VALUE_RANGE:
        _requires_column(selection_type, table_name, column_name)
        return ValueRangeStrategy()
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
        table_name: str | None = None,
        column_name: str | None = None,
    ) -> "AgentSelector":
        selection_type = ExecutionConfig.get().selection_type
        strategy = _build_strategy(selection_type, table_name, column_name)
        precomputed = await strategy.precompute(agents, table_name, column_name, query)
        state = strategy.init_state()
        selections = [
            AgentSelection(agent=a, table_name=table_name, column_name=column_name)
            for a in agents
        ]
        return cls(strategy, precomputed, state, selections)

    def rank(self, remaining: list[AgentSelection]) -> list[AgentSelection]:
        return self._strategy.rank(remaining, self._precomputed, self._state)

    def update(self, result_list: AlignedResultList) -> None:
        col = (
            result_list.schema_metadata.value_column
            or result_list.schema_metadata.keys[0]
        )
        df = result_list.data
        values = df[col].dropna().astype(str).tolist()
        self._state = self._strategy.update(self._state, values)

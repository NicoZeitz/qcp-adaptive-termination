from typing import Any

import qcp_adaptive_termination.agent_registry as agent_registry
import qcp_adaptive_termination.embedding as embedding
from qcp_adaptive_termination.embedding import VectorIndex
from qcp_adaptive_termination.models import (
    AlignedResultList,
    Config,
    DataAgent,
    SelectionStrategyType,
)
from qcp_adaptive_termination.selection_strategy import SequentialStrategy
from qcp_adaptive_termination.selection_strategy.agent_selection import AgentSelection
from qcp_adaptive_termination.selection_strategy.count_strategy import (
    CountStrategy,
    DistinctCountStrategy,
)
from qcp_adaptive_termination.selection_strategy.embedding_strategy import (
    EmbeddingStrategy,
)
from qcp_adaptive_termination.selection_strategy.posterior_strategy import (
    PosteriorStrategy,
)
from qcp_adaptive_termination.selection_strategy.sketch_strategy import (
    HllSketchStrategy,
    MinHashStrategy,
    ThetaSketchStrategy,
)
from qcp_adaptive_termination.selection_strategy.strategy import SelectionStrategy


def _build_strategy(selection_type: SelectionStrategyType) -> SelectionStrategy:
    if selection_type == SelectionStrategyType.SEQUENTIAL:
        return SequentialStrategy()
    elif selection_type == SelectionStrategyType.EMBEDDING:
        return EmbeddingStrategy()
    elif selection_type == SelectionStrategyType.COUNT:
        return CountStrategy()
    elif selection_type == SelectionStrategyType.DISTINCT_COUNT:
        return DistinctCountStrategy()
    elif selection_type == SelectionStrategyType.HLL_SKETCH:
        return HllSketchStrategy()
    elif selection_type == SelectionStrategyType.HLL_SKETCH_UCB:
        return HllSketchStrategy(use_ucb=True)
    elif selection_type == SelectionStrategyType.THETA_SKETCH:
        return ThetaSketchStrategy()
    elif selection_type == SelectionStrategyType.THETA_SKETCH_UCB:
        return ThetaSketchStrategy(use_ucb=True)
    elif selection_type == SelectionStrategyType.MIN_HASH:
        return MinHashStrategy()
    elif selection_type == SelectionStrategyType.POSTERIOR:
        return PosteriorStrategy()
    else:
        raise ValueError(f"Unsupported SelectionType: {selection_type}")


async def _resolve_table(agents: list[DataAgent], query: str) -> dict[str, str]:
    agents_vectors = await agent_registry.embedding_vectors_for_all_tables(agents)
    query_vector = await embedding.create(query)

    # for each agent find the table with the closest embedding to the query
    result: dict[str, str] = {}
    for agent, table_vectors in agents_vectors.items():
        vector_index = VectorIndex.build(list(table_vectors.items()))
        ranked = vector_index.search_top_k(query_vector, k=1)
        result[agent] = ranked[0][0]

    return result


async def _resolve_column(
    agents: list[DataAgent],
    table_per_agent: dict[str, str],
    query: str,
) -> dict[str, str]:
    agents_vectors = await agent_registry.embedding_vectors_for_all_columns(agents, table_per_agent)
    query_vector = await embedding.create(query)

    result: dict[str, str] = {}
    for agent, column_vectors in agents_vectors.items():
        vector_index = VectorIndex.build(list(column_vectors.items()))
        ranked = vector_index.search_top_k(query_vector, k=1)
        result[agent] = ranked[0][0]

    return result


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
        strategy = _build_strategy(selection_type)

        table_per_agent: dict[str, str] | None = None
        column_per_agent: dict[str, str] | None = None

        if strategy.needs_table():
            table_per_agent = await _resolve_table(agents, query)
        if strategy.needs_column():
            assert table_per_agent is not None
            column_per_agent = await _resolve_column(agents, table_per_agent, query)

        precomputed = await strategy.precompute(agents, table_per_agent, column_per_agent, query)
        state = strategy.init_state()
        selections = [AgentSelection(agent=a, score=None) for a in agents]
        return cls(strategy, precomputed, state, selections)

    def rank(self, remaining: list[AgentSelection]) -> list[AgentSelection]:
        return self._strategy.rank(remaining, self._precomputed, self._state)

    def update(self, result_list: AlignedResultList, agent_names: list[str] | None = None) -> None:
        col = result_list.schema_metadata.value_column or result_list.schema_metadata.keys[0]
        df = result_list.data
        values = df[col].dropna().astype(str).tolist()
        self._state = self._strategy.update(
            self._state,
            values,
            agent_names=agent_names or [],
            precomputed=self._precomputed,
        )

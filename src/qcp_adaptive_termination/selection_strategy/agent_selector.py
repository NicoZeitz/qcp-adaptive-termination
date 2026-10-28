"""Build selection strategies and run one multi-agent selection step."""

import dataclasses
from typing import Any

import pandas as pd

import qcp_adaptive_termination.agent_registry as agent_registry
import qcp_adaptive_termination.embedding as embedding
from qcp_adaptive_termination.embedding import VectorIndex
from qcp_adaptive_termination.models import (
    AlignedSchemaMetadata,
    DataAgent,
    ScoredDataAgent,
    SelectionStrategyType,
)
from qcp_adaptive_termination.observability import context
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
from qcp_adaptive_termination.selection_strategy.random_strategy import RandomStrategy
from qcp_adaptive_termination.selection_strategy.sketch_strategy import (
    HllSketchStrategy,
    MinHashStrategy,
    ThetaSketchStrategy,
)
from qcp_adaptive_termination.selection_strategy.strategy import (
    SelectionStrategy,
    StaticSelectionStrategy,
)

_STRATEGY_CLASSES: dict[SelectionStrategyType, type[SelectionStrategy]] = {
    SelectionStrategyType.RANDOM: RandomStrategy,
    SelectionStrategyType.EMBEDDING: EmbeddingStrategy,
    SelectionStrategyType.COUNT: CountStrategy,
    SelectionStrategyType.DISTINCT_COUNT: DistinctCountStrategy,
    SelectionStrategyType.HLL_SKETCH: HllSketchStrategy,
    SelectionStrategyType.THETA_SKETCH: ThetaSketchStrategy,
    SelectionStrategyType.MIN_HASH: MinHashStrategy,
    SelectionStrategyType.POSTERIOR: PosteriorStrategy,
}


def _build_strategy(selection_type: SelectionStrategyType, kwargs: dict[str, Any]) -> SelectionStrategy:
    """Instantiate the strategy class for a selection type with fitting kwargs."""
    strategy_cls = _STRATEGY_CLASSES.get(selection_type)
    if strategy_cls is None:
        raise ValueError(f"Unsupported SelectionType: {selection_type}")

    valid = {field.name for field in dataclasses.fields(strategy_cls)}
    kwargs = {k: v for k, v in kwargs.items() if k in valid}
    return strategy_cls(**kwargs)


async def _resolve_table(agents: list[DataAgent], query: str) -> dict[str, str]:
    """Resolve the closest-matching table for each agent by embedding similarity."""
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
    """Resolve the closest-matching column for each agent by embedding similarity."""
    agents_vectors = await agent_registry.embedding_vectors_for_all_columns(agents, table_per_agent)
    query_vector = await embedding.create(query)

    result: dict[str, str] = {}
    for agent, column_vectors in agents_vectors.items():
        vector_index = VectorIndex.build(list(column_vectors.items()))
        ranked = vector_index.search_top_k(query_vector, k=1)
        result[agent] = ranked[0][0]

    return result


class AgentSelector:
    """
    Owns the precomputed data and running state for one multi-agent step.

    Usage:
        selector = await AgentSelector.create(agents, selection_type, query, ...)
        ranked = selector.rank(remaining)
        selector.update(schema_metadata, new_results=queried, agent_names=[agent_name])
    """

    def __init__(
        self,
        strategy: SelectionStrategy,
        precomputed: Any,
        state: Any,
        selections: list[ScoredDataAgent],
    ):
        """Store the strategy, precomputed data, and running selection state."""
        self._strategy = strategy
        self._precomputed = precomputed
        self._state = state
        self.selections = selections
        self._has_ranked = False

    @classmethod
    async def create(
        cls,
        agents: list[DataAgent],
        query: str,
    ) -> AgentSelector:
        """Build a selector with a config-resolved strategy and its precomputed data."""
        selection_type = context.get_config().selection.strategy
        strategy = _build_strategy(selection_type, dict(context.get_config().selection.params))

        table_per_agent: dict[str, str] | None = None
        column_per_agent: dict[str, str] | None = None

        if strategy.needs_table():
            table_per_agent = await _resolve_table(agents, query)
        if strategy.needs_column():
            assert table_per_agent is not None
            column_per_agent = await _resolve_column(agents, table_per_agent, query)

        precomputed = await strategy.precompute(agents, table_per_agent, column_per_agent, query)
        state = strategy.init_state()
        selections = [ScoredDataAgent(agent=a, score=None) for a in agents]
        return cls(strategy, precomputed, state, selections)

    def rank(self, remaining: list[ScoredDataAgent]) -> list[ScoredDataAgent]:
        """Rank the remaining agents best-first using the strategy."""
        if isinstance(self._strategy, StaticSelectionStrategy) and self._has_ranked:
            return remaining

        ranked = self._strategy.rank(remaining, self._precomputed, self._state)
        self._has_ranked = True
        return ranked

    def update(
        self,
        schema_metadata: AlignedSchemaMetadata,
        new_results: pd.DataFrame,
        agent_names: list[str],
    ):
        """Fold newly returned column values into the strategy state."""
        if isinstance(self._strategy, StaticSelectionStrategy):
            return

        col = schema_metadata.value_column or schema_metadata.keys[0]
        values = new_results[col].dropna().astype(str).tolist() if col in new_results.columns else []
        self._state = self._strategy.update(
            self._state,
            values,
            agent_names=agent_names,
            precomputed=self._precomputed,
        )

from collections import defaultdict
from dataclasses import field
from typing import Protocol

import qcp_adaptive_termination.agent_registry as agent_registry
import qcp_adaptive_termination.embedding as embedding
from qcp_adaptive_termination.embedding import VectorIndex
from qcp_adaptive_termination.models import DataAgent
from qcp_adaptive_termination.selection_strategy.strategy import StaticSelectionStrategy


class EmbeddingScorer(Protocol):
    def __call__(self, distances: list[float]) -> float: ...


def min_distance_scorer(distances: list[float]) -> float:
    return min(distances)


class EmbeddingStrategy(StaticSelectionStrategy[float]):
    """
    Ranks agents by embedding distance to the query.

    Each agent's tables are embedded separately; scores are collapsed
    per-agent using `scorer` (default: min distance across tables).
    """

    ascending: bool = True
    scorer: EmbeddingScorer = field(default=min_distance_scorer)

    async def precompute(
        self,
        agents: list[DataAgent],
        table_per_agent: dict[str, str] | None,
        column_per_agent: dict[str, str] | None,
        query: str = "",
    ) -> dict[str, float]:
        agent_vectors = await agent_registry.embedding_vectors_for_all_tables(agents)
        query_embedding = await embedding.create(query)

        flat_data: list[tuple[str, list[float]]] = []
        for agent_name, table_vectors in agent_vectors.items():
            for vec in table_vectors.values():
                flat_data.append((agent_name, vec))

        vector_index = VectorIndex.build(flat_data)
        ranked = vector_index.search_top_k(query_embedding, k=len(flat_data))

        agent_distances: defaultdict[str, list[float]] = defaultdict(list)
        for agent_name, dist in ranked:
            agent_distances[agent_name].append(dist)

        return {
            name: self.scorer(distances) for name, distances in agent_distances.items()
        }

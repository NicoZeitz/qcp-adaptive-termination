from collections import defaultdict
from dataclasses import dataclass, field
from typing import Protocol

from model.agent.data_agent import DataAgent
from common.agent_registry.agent_registry import AgentRegistry
from common.selection_strategy.strategy import StaticSelectionStrategy
from common.embedding.embeddings import create_embedding
from common.embedding.vector_search import build_index, search_index_top_k


class EmbeddingScorer(Protocol):
    def __call__(self, distances: list[float]) -> float: ...


def min_distance_scorer(distances: list[float]) -> float:
    return min(distances)


@dataclass
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
        agent_vectors = await AgentRegistry.embedding_vectors_for_all_tables(agents)
        query_embedding = await create_embedding(query)

        flat_data: list[tuple[str, list[float]]] = []
        for agent_name, table_vectors in agent_vectors.items():
            for vec in table_vectors.values():
                flat_data.append((agent_name, vec))

        index = build_index(flat_data)
        ranked = search_index_top_k(index, query_embedding, k=len(flat_data))

        agent_distances: defaultdict[str, list[float]] = defaultdict(list)
        for agent_name, dist in ranked:
            agent_distances[agent_name].append(dist)

        return {
            name: self.scorer(distances) for name, distances in agent_distances.items()
        }

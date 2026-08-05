import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Protocol

from model.agent.data_agent import DataAgent
from common.agent_registry.agent_registry import AgentRegistry
from common.selection_strategy.strategy import StaticSelectionStrategy
from common.embeddings import create_embedding
from common.vector_search import build_index, search_index_top_k


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
        per_agent_texts: list[list[str]] = await AgentRegistry.embedding_descriptions(
            agents
        )
        per_agent_vectors: list[list[list[float]]] = list(
            await asyncio.gather(
                *[create_embedding(texts) for texts in per_agent_texts]
            )
        )

        query_embedding = await create_embedding(query)

        flat_data: list[tuple[DataAgent, list[float]]] = []
        for agent, vectors in zip(agents, per_agent_vectors):
            for vec in vectors:
                flat_data.append((agent, vec))

        index = build_index(flat_data)
        ranked = search_index_top_k(index, query_embedding, k=len(flat_data))

        agent_distances: defaultdict[str, list[float]] = defaultdict(list)
        for agent, dist in ranked:
            agent_distances[agent.name].append(dist)

        return {
            name: self.scorer(distances) for name, distances in agent_distances.items()
        }

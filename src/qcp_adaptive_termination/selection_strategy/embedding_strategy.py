"""Embedding-based static selection strategy."""

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

import qcp_adaptive_termination.agent_registry as agent_registry
import qcp_adaptive_termination.embedding as embedding
from qcp_adaptive_termination.embedding import VectorIndex
from qcp_adaptive_termination.models import AgentRegistryType, DataAgent
from qcp_adaptive_termination.selection_strategy.strategy import StaticSelectionStrategy


@dataclass(slots=True)
class EmbeddingStrategy(StaticSelectionStrategy[float]):
    """Ranks agents by embedding distance to the query."""

    ascending: ClassVar[bool] = True
    """Whether the closest query distance ranks first"""
    scorer: Callable[[list[float]], float] = field(default=min)
    """Aggregates per-agent distances into a single score"""
    registry_type: AgentRegistryType = field(default=AgentRegistryType.TECHNICAL)
    """Which registry type to embed tables from"""

    async def precompute(
        self,
        agents: list[DataAgent],
        table_per_agent: dict[str, str] | None,
        column_per_agent: dict[str, str] | None,
        query: str = "",
    ) -> dict[str, float]:
        """Precompute a per-agent distance score against the query."""
        agent_vectors = await agent_registry.embedding_vectors_for_all_tables(agents, registry_type=self.registry_type)
        query_embedding = await embedding.create(query)

        flat_data: list[tuple[str, np.ndarray]] = []
        for agent_name, table_vectors in agent_vectors.items():
            for vec in table_vectors.values():
                flat_data.append((agent_name, vec))

        vector_index = VectorIndex.build(flat_data)
        ranked = vector_index.search_top_k(query_embedding, k=len(flat_data))

        agent_distances: defaultdict[str, list[float]] = defaultdict(list)
        for agent_name, dist in ranked:
            agent_distances[agent_name].append(dist)

        return {name: self.scorer(distances) for name, distances in agent_distances.items()}

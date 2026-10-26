"""In-memory faiss vector index for nearest-neighbor search."""

from dataclasses import dataclass

import numpy as np
from faiss import Index, IndexFlatL2


@dataclass(slots=True)
class VectorIndex[T]:
    """In-memory vector index for top-k nearest-neighbor lookups."""

    index: Index
    """FAISS index holding the indexed vectors"""
    mapping: dict[int, T]
    """Maps FAISS vector ids to the indexed items"""

    @classmethod
    def build(cls, data: list[tuple[T, np.ndarray]]) -> VectorIndex[T]:
        """Build an index over the given (item, vector) pairs."""
        if not data:
            raise ValueError("Data cannot be empty")

        dimension = len(data[0][1])
        index = IndexFlatL2(dimension)
        mapping: dict[int, T] = {}

        for i, (item, vector) in enumerate(data):
            index.add(np.array([vector], dtype="float32"))
            mapping[i] = item

        return cls(index=index, mapping=mapping)

    def search_top_k(self, query: np.ndarray, k: int) -> list[tuple[T, float]]:
        """Return the k nearest items to the query with their distances."""
        query_vector = np.array([query], dtype="float32")
        distances, indices = self.index.search(query_vector, k)

        results = []
        for dist, idx in zip(distances[0], indices[0], strict=True):
            if idx != -1:
                results.append((self.mapping[idx], dist))

        return results

    def search_all(self, query: np.ndarray) -> list[tuple[T, float]]:
        """Return all indexed items sorted by distance to the query."""
        return self.search_top_k(query, k=len(self.mapping))

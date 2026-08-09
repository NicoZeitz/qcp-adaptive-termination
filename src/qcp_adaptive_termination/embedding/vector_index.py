from dataclasses import dataclass
from typing import TypeVar

import numpy as np
from faiss import Index, IndexFlatL2

T = TypeVar("T")


@dataclass(slots=True)
class VectorIndex[T]:
    index: Index
    mapping: dict[int, T]

    @classmethod
    def build(cls, data: list[tuple[T, list[float]]]) -> "VectorIndex[T]":
        if not data:
            raise ValueError("Data cannot be empty")

        dimension = len(data[0][1])
        index = IndexFlatL2(dimension)
        mapping: dict[int, T] = {}

        for i, (item, vector) in enumerate(data):
            index.add(np.array([vector], dtype="float32"))
            mapping[i] = item

        return cls(index=index, mapping=mapping)

    def search_top_k(self, query: list[float], k: int) -> list[tuple[T, float]]:
        query_vector = np.array([query], dtype="float32")
        distances, indices = self.index.search(query_vector, k)

        results = []
        for dist, idx in zip(distances[0], indices[0], strict=True):
            if idx != -1:
                results.append((self.mapping[idx], dist))

        return results

    def search_all(self, query: list[float]) -> list[tuple[T, float]]:
        return self.search_top_k(query, k=len(self.mapping))

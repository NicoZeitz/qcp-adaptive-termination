from dataclasses import dataclass
from typing import Any

from faiss import Index, IndexFlatL2
import numpy as np


@dataclass
class VectorIndex:
    index: Index
    mapping: dict[int, Any]


def build_index(data: list[tuple[Any, list[float]]]) -> VectorIndex:
    if not data:
        raise ValueError("Data cannot be empty")

    dimension = len(data[0][1])
    index = IndexFlatL2(dimension)
    mapping: dict[int, Any] = {}

    for i, (item, vector) in enumerate(data):
        index.add(np.array([vector], dtype="float32"))
        mapping[i] = item

    return VectorIndex(index=index, mapping=mapping)


def search_index_top_k(
    index: VectorIndex, query: list[float], k: int
) -> list[tuple[Any, float]]:
    query_vector = np.array([query], dtype="float32")
    distances, indices = index.index.search(
        query_vector, k
    )  # Searching for k nearest neighbors

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx != -1:  # Valid index
            results.append((index.mapping[idx], dist))

    return results

from dataclasses import dataclass

import numpy as np
from faiss import Index, IndexFlatL2


def _maybe_to_gpu(index: Index) -> Index:
    # Upgrade to GPU index when faiss-gpu is installed and a CUDA device is available.
    # Falls back silently to the CPU index otherwise.
    # NOTE: if the index is ever serialized, call faiss.index_gpu_to_cpu first.
    try:
        import faiss as _faiss  # noqa: PLC0415

        res = _faiss.StandardGpuResources()
        return _faiss.index_cpu_to_gpu(res, 0, index)
    except Exception:
        return index


@dataclass(slots=True)
class VectorIndex[T]:
    index: Index
    mapping: dict[int, T]

    @classmethod
    def build(cls, data: list[tuple[T, list[float]]]) -> VectorIndex[T]:
        if not data:
            raise ValueError("Data cannot be empty")

        dimension = len(data[0][1])
        index = IndexFlatL2(dimension)

        items, vectors = zip(*data, strict=True)
        index.add(np.array(vectors, dtype="float32"))
        mapping: dict[int, T] = dict(enumerate(items))

        index = _maybe_to_gpu(index)
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

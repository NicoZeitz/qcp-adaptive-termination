from dataclasses import dataclass

import numpy as np
from faiss import Index, IndexFlatL2

# Metal GPU backend enforces a maximum k in flat search; stay below it. TODO:FW remove, was only needed for local run
_METAL_MAX_K = 1024


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
    cpu_index: Index  # kept for searches that exceed the GPU k limit
    mapping: dict[int, T]

    @classmethod
    def build(cls, data: list[tuple[T, list[float]]]) -> VectorIndex[T]:
        if not data:
            raise ValueError("Data cannot be empty")

        dimension = len(data[0][1])
        cpu_index = IndexFlatL2(dimension)

        items, vectors = zip(*data, strict=True)
        cpu_index.add(np.array(vectors, dtype="float32"))
        mapping: dict[int, T] = dict(enumerate(items))

        gpu_index = _maybe_to_gpu(cpu_index)
        return cls(index=gpu_index, cpu_index=cpu_index, mapping=mapping)

    def search_top_k(self, query: list[float], k: int) -> list[tuple[T, float]]:
        query_vector = np.array([query], dtype="float32")
        index = self.cpu_index if k > _METAL_MAX_K else self.index
        distances, indices = index.search(query_vector, k)

        results = []
        for dist, idx in zip(distances[0], indices[0], strict=True):
            if idx != -1:
                results.append((self.mapping[idx], dist))

        return results

    def search_all(self, query: list[float]) -> list[tuple[T, float]]:
        return self.search_top_k(query, k=len(self.mapping))

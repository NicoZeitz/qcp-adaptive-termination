"""
Text embedding creation and vector search.

Wraps an embedding model to produce dense vectors and provides an in-memory
`VectorIndex` for top-k nearest-neighbour lookups.
"""

from qcp_adaptive_termination.embedding.embeddings import create_embedding
from qcp_adaptive_termination.embedding.vector_search import (
    VectorIndex,
    build_index,
    search_index_top_k,
)

__all__ = ["create_embedding", "VectorIndex", "build_index", "search_index_top_k"]

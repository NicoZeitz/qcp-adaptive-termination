"""
Text embedding creation and vector search.

Wraps an embedding model to produce dense vectors and provides an in-memory
`VectorIndex` for top-k nearest-neighbor lookups.
"""

from typing import cast, overload

from agents import OpenAIProvider
from openai import AsyncOpenAI

from qcp_adaptive_termination.cache import cached
from qcp_adaptive_termination.embedding.vector_index import VectorIndex
from qcp_adaptive_termination.llm import model_provider
from qcp_adaptive_termination.models import Config
from qcp_adaptive_termination.observability import tracing


def _get_embedding_client_and_model() -> tuple[AsyncOpenAI, str]:
    embedding_model = Config.get().embedding_model
    multi_provider = model_provider.get_multi_provider()
    prefix, stripped = cast(
        tuple[str, str], multi_provider._get_prefix_and_model_name(embedding_model)
    )
    provider, model_name = cast(
        tuple[OpenAIProvider, str],
        multi_provider._resolve_prefixed_model(
            original_model_name=embedding_model,
            prefix=prefix,
            stripped_model_name=stripped,
        ),
    )
    return provider._get_client(), model_name


@overload
async def create(text: str) -> list[float]: ...
@overload
async def create(text: list[str]) -> list[list[float]]: ...
@cached(name_prefix="Embedding")
async def create(text):
    with tracing.own_span("Embedding", input={"text": text}):
        client, model = _get_embedding_client_and_model()
        result = await client.embeddings.create(model=model, input=text)

        if isinstance(text, list):
            return [
                item.embedding for item in sorted(result.data, key=lambda x: x.index)
            ]
        else:
            return result.data[0].embedding


__all__ = [
    "VectorIndex",
    "create",
]

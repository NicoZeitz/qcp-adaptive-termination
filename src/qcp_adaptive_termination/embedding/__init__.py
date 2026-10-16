"""
Text embedding creation and vector search.

Wraps an embedding model to produce dense vectors and provides an in-memory
`VectorIndex` for top-k nearest-neighbor lookups.
"""

import asyncio
import time
from typing import cast, overload

import tiktoken
from agents import OpenAIProvider
from openai import AsyncOpenAI

from qcp_adaptive_termination.embedding.vector_index import VectorIndex
from qcp_adaptive_termination.llm import model_provider
from qcp_adaptive_termination.observability import context, tracing
from qcp_adaptive_termination.observability.cache import get_or_compute_batch

_MAX_TOKENS_PER_REQUEST = 300000
_MAX_CHUNKS = 2048
_BATCH_TOKEN_LIMIT = 200000


def _get_embedding_client_and_model() -> tuple[AsyncOpenAI, str]:
    embedding_model = context.get_config().embedding_model
    multi_provider = model_provider.get_multi_provider()
    prefix, stripped = cast(tuple[str, str], multi_provider._get_prefix_and_model_name(embedding_model))  # type: ignore[reportPrivateUsage]
    provider, model_name = cast(
        tuple[OpenAIProvider, str],
        multi_provider._resolve_prefixed_model(  # type: ignore[reportPrivateUsage]
            original_model_name=embedding_model,
            prefix=prefix,
            stripped_model_name=stripped,
        ),
    )
    return provider._get_client(), model_name  # type: ignore[reportPrivateUsage]


def _num_tokens_from_string(string: list[str], encoding_name: str = "cl100k_base") -> list[int]:
    encoding = tiktoken.get_encoding(encoding_name)
    return list(map(lambda x: len(x), encoding.encode_batch(string)))


@overload
async def _create_batched(texts: str) -> list[float]: ...
@overload
async def _create_batched(texts: list[str]) -> list[list[float]]: ...
async def _create_batched(texts):
    client, model = _get_embedding_client_and_model()

    if not isinstance(texts, list):
        result = await client.embeddings.create(model=model, input=texts)
        return result.data[0].embedding

    # https://developers.openai.com/api/reference/resources/embeddings/methods/create#embeddings-create-input
    # https://community.openai.com/t/max-total-embeddings-tokens-per-request/1254699/6
    # However the limit of those 300k tokens is not calculated with the actual tiktoken tokenizer but instead is an
    # estimate of 0.25 tokens per utf-8 byte. So you can actually embed 1 million tokens in one request as long as they
    # use few enough bytes.
    tokens_per_text = _num_tokens_from_string(texts)
    token_estimate_per_text = [len(text.encode("utf-8")) // 4 for text in texts]

    if (
        sum(tokens_per_text) <= _BATCH_TOKEN_LIMIT
        and sum(token_estimate_per_text) <= _BATCH_TOKEN_LIMIT
        and len(texts) <= _MAX_CHUNKS
    ):
        result = await client.embeddings.create(model=model, input=texts)
        return [item.embedding for item in sorted(result.data, key=lambda x: x.index)]

    batches: list[list[str]] = []
    current_batch: list[str] = []
    current_batch_tokens = 0

    for text, token_count, token_estimate in zip(texts, tokens_per_text, token_estimate_per_text, strict=True):
        if (
            current_batch_tokens + token_count > _BATCH_TOKEN_LIMIT
            or current_batch_tokens + token_estimate > _BATCH_TOKEN_LIMIT
            or len(current_batch) >= _MAX_CHUNKS
        ) and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_batch_tokens = 0

        current_batch.append(text)
        current_batch_tokens += token_count

    if current_batch:
        batches.append(current_batch)

    # send requests in parallel
    results = await asyncio.gather(*[client.embeddings.create(model=model, input=batch) for batch in batches])

    embeddings: list[list[float]] = []
    for result in results:
        embeddings.extend([item.embedding for item in sorted(result.data, key=lambda x: x.index)])
    return embeddings


@overload
async def create(text: str) -> list[float]: ...
@overload
async def create(text: list[str]) -> list[list[float]]: ...
async def create(text):
    with tracing.own_span("Embedding", input={"text": text}):
        is_single = not isinstance(text, list)
        texts = [text] if is_single else text
        t0 = time.monotonic()
        results = await get_or_compute_batch(
            texts,
            _create_batched,
            argument_name="text",
            name_prefix="Embedding",
        )
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        acc = context.get_usage_accumulator()
        acc.embedding_calls += 1
        acc.embedding_duration_ms += elapsed_ms
        return results[0] if is_single else results


__all__ = [
    "VectorIndex",
    "create",
]

"""Text embedding creation and vector search."""

import asyncio
from typing import cast, overload

import numpy as np
import tiktoken
from agents import OpenAIProvider
from openai import AsyncOpenAI

import qcp_adaptive_termination.observability.cache as cache
from qcp_adaptive_termination import hooks
from qcp_adaptive_termination.embedding.vector_index import VectorIndex
from qcp_adaptive_termination.hooks import HookEvent
from qcp_adaptive_termination.llm import model_provider
from qcp_adaptive_termination.observability import context, tracing

_MAX_TOKENS_PER_REQUEST = 200000
_MAX_CHUNKS = 2048
_EMBEDDING_SEMAPHORE = asyncio.Semaphore(5)


def _get_embedding_client_and_model() -> tuple[AsyncOpenAI, str]:
    """Resolve and return the embedding client and model name."""
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
    """Return the token count of each input string."""
    encoding = tiktoken.get_encoding(encoding_name)
    return list(map(lambda x: len(x), encoding.encode_batch(string)))


async def _create_batched(texts: list[str]) -> list[np.ndarray]:
    """Create embeddings for texts, batching when over the per-request token limit."""
    client, model = _get_embedding_client_and_model()

    if len(texts) == 1:
        async with _EMBEDDING_SEMAPHORE:
            result = await client.embeddings.create(model=model, input=texts)
        return [np.asarray(result.data[0].embedding, dtype=np.float32)]

    # https://developers.openai.com/api/reference/resources/embeddings/methods/create#embeddings-create-input
    # https://community.openai.com/t/max-total-embeddings-tokens-per-request/1254699/6
    # However the limit of those 300k tokens is not calculated with the actual tiktoken tokenizer but instead is an
    # estimate of 0.25 tokens per utf-8 byte. So you can actually embed 1 million tokens in one request as long as they
    # use few enough bytes.
    tokens_per_text = _num_tokens_from_string(texts)
    token_estimate_per_text = [len(text.encode("utf-8")) // 4 for text in texts]

    if (
        sum(tokens_per_text) <= _MAX_TOKENS_PER_REQUEST
        and sum(token_estimate_per_text) <= _MAX_TOKENS_PER_REQUEST
        and len(texts) <= _MAX_CHUNKS
    ):
        async with _EMBEDDING_SEMAPHORE:
            result = await client.embeddings.create(model=model, input=texts)
        return [np.asarray(item.embedding, dtype=np.float32) for item in sorted(result.data, key=lambda x: x.index)]

    batches: list[list[str]] = []
    current_batch: list[str] = []
    current_batch_tokens = 0

    for text, token_count, token_estimate in zip(texts, tokens_per_text, token_estimate_per_text, strict=True):
        if (
            current_batch_tokens + token_count > _MAX_TOKENS_PER_REQUEST
            or current_batch_tokens + token_estimate > _MAX_TOKENS_PER_REQUEST
            or len(current_batch) >= _MAX_CHUNKS
        ) and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_batch_tokens = 0

        current_batch.append(text)
        current_batch_tokens += token_count

    if current_batch:
        batches.append(current_batch)

    async def _fetch(batch: list[str]):
        async with _EMBEDDING_SEMAPHORE:
            return await client.embeddings.create(model=model, input=batch)

    # send requests in parallel
    results = await asyncio.gather(*[_fetch(batch) for batch in batches])

    embeddings: list[np.ndarray] = []
    for result in results:
        embeddings.extend(
            np.asarray(item.embedding, dtype=np.float32) for item in sorted(result.data, key=lambda x: x.index)
        )
    return embeddings


_inflight: dict[str, asyncio.Future[np.ndarray]] = {}


async def _create_with_inflight(texts: list[str]) -> list[np.ndarray]:
    """Create embeddings, sharing in-flight requests for identical texts."""
    loop = asyncio.get_running_loop()

    to_request: list[str] = []
    for text in texts:
        if text not in _inflight:
            fut: asyncio.Future[np.ndarray] = loop.create_future()
            _inflight[text] = fut
            to_request.append(text)

    futures = [_inflight[text] for text in texts]

    if to_request:
        try:
            new_embeddings = await _create_batched(to_request)
            for text, embedding in zip(to_request, new_embeddings, strict=True):
                _inflight[text].set_result(embedding)
        except Exception as exc:
            for text in to_request:
                if text in _inflight and not _inflight[text].done():
                    _inflight[text].set_exception(exc)
        finally:
            for text in to_request:
                _inflight.pop(text, None)

    return list(await asyncio.gather(*futures))


@overload
async def create(text: str) -> np.ndarray: ...
@overload
async def create(text: list[str]) -> list[np.ndarray]: ...
async def create(text):
    """Create dense embeddings for a single text or a list of texts."""
    with tracing.own_span("Embedding", input={"text": text}):
        is_single = not isinstance(text, list)
        texts = [text] if is_single else text
        results = await cache.get_or_compute_batch(
            texts,
            _create_with_inflight,
        )

        for t, tokens in zip(texts, _num_tokens_from_string(texts), strict=True):
            await hooks.fire_post(
                HookEvent.EMBEDDING_COMPLETED,
                text=t,
                model=context.get_config().embedding_model,
                prompt_tokens=tokens,
            )

        return results[0] if is_single else results


__all__ = [
    "VectorIndex",
    "create",
]

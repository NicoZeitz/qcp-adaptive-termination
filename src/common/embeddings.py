from typing import cast, overload

from agents import OpenAIProvider
from openai import AsyncOpenAI

from common.agent_configuration import EMBEDDING_MODEL, get_multi_provider
from common.cache import cached
from common.span import own_custom_span


def get_client_and_model() -> tuple[AsyncOpenAI, str]:
    multi_provider = get_multi_provider()
    prefix, stripped = cast(
        tuple[str, str], multi_provider._get_prefix_and_model_name(EMBEDDING_MODEL)
    )
    provider, model_name = cast(
        tuple[OpenAIProvider, str],
        multi_provider._resolve_prefixed_model(
            original_model_name=EMBEDDING_MODEL,
            prefix=prefix,
            stripped_model_name=stripped,
        ),
    )
    return provider._get_client(), model_name


@overload
async def create_embedding(text: str) -> list[float]: ...
@overload
async def create_embedding(text: list[str]) -> list[list[float]]: ...
@cached(name_prefix="Embedding")
async def create_embedding(text):
    with own_custom_span("Embedding", input={"text": text}):
        client, model = get_client_and_model()
        result = await client.embeddings.create(model=model, input=text)

        if isinstance(text, list):
            return [
                item.embedding for item in sorted(result.data, key=lambda x: x.index)
            ]
        else:
            return result.data[0].embedding

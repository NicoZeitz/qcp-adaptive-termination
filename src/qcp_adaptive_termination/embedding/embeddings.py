from typing import cast, overload

from agents import OpenAIProvider
from openai import AsyncOpenAI

from qcp_adaptive_termination.llm.providers import get_multi_provider
from qcp_adaptive_termination.cache import cached
from qcp_adaptive_termination.tracing.span import own_custom_span
from qcp_adaptive_termination.models.config import config_var


def get_client_and_model() -> tuple[AsyncOpenAI, str]:
    embedding_model = config_var.get().embedding_model
    multi_provider = get_multi_provider()
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

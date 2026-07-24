import asyncio
from dataclasses import dataclass
from typing import cast

from agents import OpenAIProvider
from openai import AsyncOpenAI

from qtypes.data_agent import DataAgent
from stages.common.agent_configuration import EMBEDDING_MODEL, get_multi_provider
from stages.common.data_agent.agent_registry import (
    AgentRegistryType,
    build_agent_registry,
)
from stages.execution.selection.agent_selection import AgentSelection
from stages.execution.selection.strategy import StaticSelectionStrategy
from stages.execution.selection.vector_search import build_index, search_index_top_k
from utils.cache import cached
from utils.span import own_custom_span


def _get_client_and_model() -> tuple[AsyncOpenAI, str]:
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


async def _create_embedding(text: str) -> list[float]:
    client, model = _get_client_and_model()
    result = await client.embeddings.create(model=model, input=text)
    return result.data[0].embedding


@cached(name_prefix="Text Embedding")
async def _create_text_embedding(text: str) -> list[float]:
    with own_custom_span("Text Embedding", input={"text": text}):
        return await _create_embedding(text)

async def _create_agent_embedding(
    data_agent: DataAgent, agent_registry_type: AgentRegistryType
) -> list[float]:
    @cached(name_prefix=f"{data_agent.name} - Agent Embedding")
    async def compute_embedding(data_agent: DataAgent, agent_registry_type: AgentRegistryType) -> list[float]:
        registry_text = await build_agent_registry([data_agent], agent_registry_type)
        return await _create_embedding(registry_text)

    with own_custom_span("Agent Embeddings", input={"agent": data_agent.name}):
        return await compute_embedding(data_agent, agent_registry_type)

@dataclass
class EmbeddingStrategy(StaticSelectionStrategy[float]):
    """
    Ranks agents by L2 distance of their registry embedding to the query.

    ascending=True: lower distance = better rank.
    Precomputed: raw L2 distance per agent (cached on disk).
    """

    registry_type: AgentRegistryType = AgentRegistryType.NATURAL_LANGUAGE
    ascending: bool = True

    async def precompute(
        self,
        agents: list[DataAgent],
        table_name: str | None,
        column_name: str | None,
        query: str = "",
    ) -> dict[str, float]:
        agent_embeddings = await asyncio.gather(
            *[_create_agent_embedding(a, self.registry_type) for a in agents]
        )
        query_embedding = await _create_text_embedding(query)

        selections = [
            AgentSelection(agent=a, table_name=None, column_name=None) for a in agents
        ]
        index = build_index(list(zip(selections, agent_embeddings)))
        ranked = search_index_top_k(index, query_embedding, k=len(agents))
        return {s.agent.name: dist for s, dist in ranked}

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Protocol, cast, overload

from agents import OpenAIProvider
from openai import AsyncOpenAI

from qtypes.data_agent import DataAgent
from stages.common.agent_configuration import EMBEDDING_MODEL, get_multi_provider
from stages.common.data_agent.agent_registry import AgentRegistry
from stages.execution.selection.strategy import StaticSelectionStrategy
from stages.execution.selection.vector_search import build_index, search_index_top_k
from utils.cache import cached
from utils.span import own_custom_span


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


class EmbeddingScorer(Protocol):
    def __call__(self, distances: list[float]) -> float: ...


def min_distance_scorer(distances: list[float]) -> float:
    return min(distances)


@dataclass
class EmbeddingStrategy(StaticSelectionStrategy[float]):
    """
    Ranks agents by embedding distance to the query.

    Each agent's tables are embedded separately; scores are collapsed
    per-agent using `scorer` (default: min distance across tables).
    """

    ascending: bool = True
    scorer: EmbeddingScorer = field(default=min_distance_scorer)

    async def precompute(
        self,
        agents: list[DataAgent],
        table_name: str | None,
        column_name: str | None,
        query: str = "",
    ) -> dict[str, float]:
        per_agent_texts: list[list[str]] = await AgentRegistry.embedding_descriptions(
            agents
        )
        per_agent_vectors: list[list[list[float]]] = list(
            await asyncio.gather(
                *[create_embedding(texts) for texts in per_agent_texts]
            )
        )

        query_embedding = await create_embedding(query)

        flat_data: list[tuple[DataAgent, list[float]]] = []
        for agent, vectors in zip(agents, per_agent_vectors):
            for vec in vectors:
                flat_data.append((agent, vec))

        index = build_index(flat_data)
        ranked = search_index_top_k(index, query_embedding, k=len(flat_data))

        agent_distances: defaultdict[str, list[float]] = defaultdict(list)
        for agent, dist in ranked:
            agent_distances[agent.name].append(dist)

        return {
            name: self.scorer(distances) for name, distances in agent_distances.items()
        }

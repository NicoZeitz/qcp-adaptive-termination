from dataclasses import dataclass, field
from typing import cast

import evoc
import numpy as np
from datasketches import (
    compact_theta_sketch,
    theta_a_not_b,
    update_theta_sketch,
)

import qcp_adaptive_termination.agent_registry as agent_registry
import qcp_adaptive_termination.database as database
from qcp_adaptive_termination.models import DataAgent
from qcp_adaptive_termination.selection_strategy.agent_selection import AgentSelection
from qcp_adaptive_termination.selection_strategy.sketch_strategy import DEFAULT_LG_K
from qcp_adaptive_termination.selection_strategy.strategy import SelectionStrategy

DEFAULT_MIN_SAMPLES = 1
DEFAULT_MIN_CLUSTER_SIZE_RATIO = 0.05


@dataclass(slots=True)
class PosteriorPrecomputedState:
    embedding: list[float]
    theta_sketch: compact_theta_sketch


@dataclass(slots=True)
class PosteriorState:
    theta: update_theta_sketch
    queried_embeddings: list[list[float]] = field(default_factory=list)


def _normalize_probability(values: list[float]) -> list[float]:
    total = sum(values)
    if total == 0:
        return [1.0 / len(values)] * len(values)
    return [v / total for v in values]


@dataclass(slots=True)
class PosteriorStrategy(SelectionStrategy[PosteriorPrecomputedState, PosteriorState]):
    lg_k: int = DEFAULT_LG_K
    clustered_prior: bool = True
    min_samples: int = DEFAULT_MIN_SAMPLES
    min_cluster_size_ratio: float = DEFAULT_MIN_CLUSTER_SIZE_RATIO

    def needs_table(self) -> bool:
        return True

    def needs_column(self) -> bool:
        return True

    async def precompute(
        self,
        agents: list[DataAgent],
        table_per_agent: dict[str, str] | None,
        column_per_agent: dict[str, str] | None,
        query: str = "",
    ) -> dict[str, PosteriorPrecomputedState]:
        assert table_per_agent and column_per_agent

        vectors = await agent_registry.embedding_vector_for_table(agents, table_per_agent)

        result: dict[str, PosteriorPrecomputedState] = {}
        for agent in agents:
            sketch = update_theta_sketch(self.lg_k)
            for v in database.fetch_column_values(agent, table_per_agent[agent.name], column_per_agent[agent.name]):
                sketch.update(v)

            result[agent.name] = PosteriorPrecomputedState(
                embedding=vectors[agent.name],
                theta_sketch=sketch.compact(),
            )

        return result

    def init_state(self) -> PosteriorState:
        return PosteriorState(theta=update_theta_sketch(self.lg_k))

    def calculate_prior(
        self,
        remaining: list[AgentSelection],
        state: PosteriorState,
        precomputed: dict[str, PosteriorPrecomputedState],
    ) -> list[float]:
        if len(state.queried_embeddings) == 0:
            # uniform prior if no agents have been queried yet
            return [1.0 / len(remaining)] * len(remaining)

        if self.clustered_prior:
            return self._calculate_clustered_prior(remaining, state, precomputed)
        else:
            return self._calculate_non_clustered_prior(remaining, state, precomputed)

    def calculate_likelihood(
        self,
        remaining: list[AgentSelection],
        state: PosteriorState,
        precomputed: dict[str, PosteriorPrecomputedState],
    ) -> list[float]:
        likelihood_raw: list[float] = []
        for s in remaining:
            a_not_b = theta_a_not_b().compute(precomputed[s.agent.name].theta_sketch, state.theta)
            likelihood_raw.append(max(0.0, a_not_b.get_estimate()))

        return _normalize_probability(likelihood_raw)

    def rank(
        self,
        remaining: list[AgentSelection],
        precomputed: dict[str, PosteriorPrecomputedState],
        state: PosteriorState,
    ) -> list[AgentSelection]:
        prior = self.calculate_prior(remaining, state, precomputed)
        likelihood = self.calculate_likelihood(remaining, state, precomputed)

        # posterior (unnormalized product) → sort
        for i, s in enumerate(remaining):
            s.score = prior[i] * likelihood[i]

        return sorted(remaining, key=lambda s: cast(float, s.score), reverse=True)

    def update(
        self,
        state: PosteriorState,
        values: list[str],
        agent_names: list[str],
        precomputed: dict[str, PosteriorPrecomputedState],
    ) -> PosteriorState:
        for v in values:
            state.theta.update(v)

        for name in agent_names:
            state.queried_embeddings.append(precomputed[name].embedding)

        return state

    def _calculate_clustered_prior(
        self,
        remaining: list[AgentSelection],
        state: PosteriorState,
        precomputed: dict[str, PosteriorPrecomputedState],
    ) -> list[float]:
        reference_embeddings = np.array(state.queried_embeddings, dtype=float)

        agent_embeddings = np.array([precomputed[s.agent.name].embedding for s in remaining], dtype=float)

        min_cluster_size = max(5, int(len(remaining) * self.min_cluster_size_ratio))
        labels = evoc.EVoC(base_min_cluster_size=min_cluster_size, min_samples=self.min_samples).fit_predict(
            agent_embeddings.astype(np.float32)
        )

        centroids: dict[int, np.ndarray] = {}
        for lbl in set(labels):
            mask = labels == lbl
            centroids[int(lbl)] = agent_embeddings[mask].mean(axis=0)

        prior_raw: list[float] = []
        for i in range(len(remaining)):
            centroid = centroids[labels[i]]
            distances = np.linalg.norm(reference_embeddings - centroid, axis=1)
            prior_raw.append(float(distances.min()))

        return _normalize_probability(prior_raw)

    def _calculate_non_clustered_prior(
        self,
        remaining: list[AgentSelection],
        state: PosteriorState,
        precomputed: dict[str, PosteriorPrecomputedState],
    ) -> list[float]:
        reference_embeddings = np.array(state.queried_embeddings, dtype=float)

        # for each remaining agent
        # compute for the embedding the distance to all reference embeddings, and take the minimum distance
        prior_raw: list[float] = []
        for s in remaining:
            agent_embedding = np.array(precomputed[s.agent.name].embedding, dtype=float)
            distances = np.linalg.norm(reference_embeddings - agent_embedding, axis=1)
            prior_raw.append(float(distances.min()))

        return _normalize_probability(prior_raw)

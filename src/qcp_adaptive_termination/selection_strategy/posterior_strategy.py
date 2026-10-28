"""Bayesian-posterior agent selection strategy over column overlap."""

import asyncio
from dataclasses import dataclass, field
from typing import cast

import numpy as np
from datasketches import (
    compact_theta_sketch,
    theta_a_not_b,
    update_theta_sketch,
)
lazy import evoc

import qcp_adaptive_termination.agent_registry as agent_registry
import qcp_adaptive_termination.database as database
from qcp_adaptive_termination.models import AgentRegistryType, DataAgent, ScoredDataAgent
from qcp_adaptive_termination.selection_strategy.sketch_strategy import DEFAULT_LG_K
from qcp_adaptive_termination.selection_strategy.strategy import SelectionStrategy

DEFAULT_MIN_SAMPLES = 1
DEFAULT_MIN_CLUSTER_SIZE_RATIO = 0.05


@dataclass(slots=True)
class PosteriorAgentPrecomputedState:
    """Precomputed embedding and theta sketch for an agent."""

    embedding: np.ndarray
    """Precomputed embedding vector for the agent"""
    theta_sketch: compact_theta_sketch
    """Precomputed compact theta sketch of the agent's column values"""


@dataclass(slots=True)
class PosteriorPrecomputedState:
    """All per-agent posterior data plus the aligned embedding matrix."""

    agents: dict[str, PosteriorAgentPrecomputedState]
    """Per-agent embedding and theta sketch"""
    embedding_names: tuple[str, ...]
    """Agent names matching rows in the embedding matrix"""
    embedding_matrix: np.ndarray
    """Float64 embedding matrix used for incremental distance updates"""


@dataclass(slots=True)
class PosteriorState:
    """Running posterior state: theta sketch and queried embeddings."""

    theta: update_theta_sketch
    """Running theta sketch of values seen so far"""
    queried_embeddings: list[np.ndarray] = field(default_factory=list)
    """Embeddings of agents queried so far"""
    min_distances: dict[str, float] = field(default_factory=dict)
    """Minimum embedding distance to any queried agent, by candidate name"""


def _normalize_probability(values: np.ndarray) -> np.ndarray:
    """Normalize a numpy array of scores to probabilities (vectorized)."""
    total = values.sum()
    if total == 0:
        return np.ones(values.size, dtype=float) / values.size
    return values / total


@dataclass(slots=True)
class PosteriorStrategy(SelectionStrategy[PosteriorPrecomputedState, PosteriorState]):
    """Ranks agents by a Bayesian posterior over column overlap."""

    lg_k: int = DEFAULT_LG_K
    """Number of bits for the theta sketch precision"""
    prior_w: float = 0.5
    """Exponent weight for the embedding prior (0: likelihood only, 1: prior only)"""
    clustered_prior: bool = False
    """Whether to use the clustered prior"""
    min_samples: int = DEFAULT_MIN_SAMPLES
    """Minimum number of samples to use a distribution"""
    min_cluster_size_ratio: float = DEFAULT_MIN_CLUSTER_SIZE_RATIO
    """Minimum cluster size as a fraction of total agents"""
    registry_type: AgentRegistryType = AgentRegistryType.TECHNICAL
    """Which registry type to embed tables from"""

    def needs_table(self) -> bool:
        """Return True, since this strategy requires a table per agent."""
        return True

    def needs_column(self) -> bool:
        """Return True, since this strategy requires a column per agent."""
        return True

    async def precompute(
        self,
        agents: list[DataAgent],
        table_per_agent: dict[str, str] | None,
        column_per_agent: dict[str, str] | None,
        query: str = "",
    ) -> PosteriorPrecomputedState:
        """Precompute each agent's embedding and theta sketch."""
        assert table_per_agent and column_per_agent

        vectors = await agent_registry.embedding_vector_for_table(
            agents, table_per_agent, registry_type=self.registry_type
        )

        async def _build(agent: DataAgent) -> tuple[str, PosteriorAgentPrecomputedState]:
            """Fetch one agent's column values into its precomputed state."""
            values = await asyncio.to_thread(
                database.fetch_column_values, agent, table_per_agent[agent.name], column_per_agent[agent.name]
            )
            sketch = update_theta_sketch(self.lg_k)
            for v in values:
                sketch.update(v)
            return agent.name, PosteriorAgentPrecomputedState(
                embedding=vectors[agent.name],
                theta_sketch=sketch.compact(),
            )

        agent_precomputed = dict(await asyncio.gather(*[_build(a) for a in agents]))
        embedding_names = tuple(agent.name for agent in agents)
        embedding_matrix = np.asarray(
            [agent_precomputed[name].embedding for name in embedding_names],
            dtype=float,
        )
        return PosteriorPrecomputedState(
            agents=agent_precomputed,
            embedding_names=embedding_names,
            embedding_matrix=embedding_matrix,
        )

    def init_state(self) -> PosteriorState:
        """Return an empty posterior state."""
        return PosteriorState(theta=update_theta_sketch(self.lg_k))

    def calculate_prior(
        self,
        remaining: list[ScoredDataAgent],
        state: PosteriorState,
        precomputed: PosteriorPrecomputedState,
    ) -> np.ndarray:
        """Prior probability over remaining agents from queried embeddings."""
        if len(state.queried_embeddings) == 0:
            # uniform prior if no agents have been queried yet
            return np.ones(len(remaining), dtype=float) / len(remaining)

        if self.clustered_prior:
            return self._calculate_clustered_prior(remaining, state, precomputed)
        else:
            return self._calculate_non_clustered_prior(remaining, state)

    def calculate_likelihood(
        self,
        remaining: list[ScoredDataAgent],
        state: PosteriorState,
        precomputed: PosteriorPrecomputedState,
    ) -> np.ndarray:
        """Likelihood of new distinct values for each remaining agent."""
        likelihood_raw = np.empty(len(remaining), dtype=float)
        for i, s in enumerate(remaining):
            a_not_b = theta_a_not_b().compute(
                precomputed.agents[s.agent.name].theta_sketch,
                state.theta,
            )
            likelihood_raw[i] = max(0.0, a_not_b.get_estimate())

        return _normalize_probability(likelihood_raw)

    def rank(
        self,
        remaining: list[ScoredDataAgent],
        precomputed: PosteriorPrecomputedState,
        state: PosteriorState,
    ) -> list[ScoredDataAgent]:
        """Score and sort remaining agents by the prior * likelihood posterior."""
        if not remaining:
            return []

        prior = self.calculate_prior(remaining, state, precomputed)
        likelihood = self.calculate_likelihood(remaining, state, precomputed)

        posterior = np.power(prior, self.prior_w) * np.power(likelihood, 1.0 - self.prior_w)
        for i, s in enumerate(remaining):
            s.score = float(posterior[i])

        return sorted(remaining, key=lambda s: cast(float, s.score), reverse=True)

    def update(
        self,
        state: PosteriorState,
        values: list[str],
        agent_names: list[str],
        precomputed: PosteriorPrecomputedState,
    ) -> PosteriorState:
        """Fold seen values and queried agent embeddings into the running state."""
        for v in values:
            state.theta.update(v)

        for name in agent_names:
            queried_embedding = precomputed.agents[name].embedding
            state.queried_embeddings.append(queried_embedding)

            distances = np.linalg.norm(precomputed.embedding_matrix - queried_embedding, axis=1)
            for candidate_name, distance in zip(precomputed.embedding_names, distances, strict=True):
                state.min_distances[candidate_name] = min(
                    state.min_distances.get(candidate_name, float("inf")),
                    float(distance),
                )

        return state

    def _calculate_clustered_prior(
        self,
        remaining: list[ScoredDataAgent],
        state: PosteriorState,
        precomputed: PosteriorPrecomputedState,
    ) -> np.ndarray:
        """Prior probability using EVoC-clustered embeddings."""
        # EVoC's knn graph requires n_samples > n_neighbors (default 15); fall back for small sets
        if len(remaining) <= 15:
            return self._calculate_non_clustered_prior(remaining, state)

        reference_embeddings = np.asarray(state.queried_embeddings, dtype=float)  # (Q, D)
        agent_embeddings = np.asarray(
            [precomputed.agents[s.agent.name].embedding for s in remaining],
            dtype=float,  # (R, D)
        )

        min_cluster_size = max(5, int(len(remaining) * self.min_cluster_size_ratio))
        labels = np.asarray(
            evoc.EVoC(
                base_min_cluster_size=min_cluster_size, min_samples=self.min_samples, random_state=42
            ).fit_predict(agent_embeddings.astype(np.float32))
        )

        # Centroid per label, vectorized.
        unique_labels = np.unique(labels)
        centroids: dict[int, np.ndarray] = {
            int(lbl): agent_embeddings[labels == lbl].mean(axis=0) for lbl in unique_labels
        }
        centroid_matrix = np.asarray([centroids[int(lbl)] for lbl in labels], dtype=float)  # (R, D)

        # min L2 distance from each agent's centroid to all queried references (vectorized).
        dists = np.linalg.norm(reference_embeddings[:, None, :] - centroid_matrix[None, :, :], axis=2)
        prior_raw = dists.min(axis=0)
        return _normalize_probability(prior_raw)

    def _calculate_non_clustered_prior(
        self,
        remaining: list[ScoredDataAgent],
        state: PosteriorState,
    ) -> np.ndarray:
        """Prior probability from embedding distances to queried agents."""
        prior_raw = np.fromiter(
            (state.min_distances.get(s.agent.name, float("inf")) for s in remaining),
            dtype=float,
            count=len(remaining),
        )
        return _normalize_probability(prior_raw)

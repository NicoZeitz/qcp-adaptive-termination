"""Sketch-based agent selection strategies (HLL, theta, MinHash)."""

import asyncio
from dataclasses import dataclass, field
from typing import cast

from datasketch import MinHash
from datasketches import (
    compact_theta_sketch,
    hll_sketch,
    hll_union,
    theta_a_not_b,
    update_theta_sketch,
)

import qcp_adaptive_termination.database as database
from qcp_adaptive_termination.models import DataAgent, ScoredDataAgent
from qcp_adaptive_termination.selection_strategy.strategy import (
    SelectionStrategy,
)

DEFAULT_LG_K = 12
DEFAULT_NUM_STD = 2
DEFAULT_MIN_HASH_NUM_PERM = 128


# ---------------------------------------------------------------------------
# HLL
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class HllSketchStrategy(SelectionStrategy[dict[str, hll_sketch], hll_sketch]):
    """Ranks by estimated new unique values each remaining agent would contribute."""

    lg_k: int = DEFAULT_LG_K
    """Number of bits for the HLL sketch precision"""
    use_ucb: bool = False
    """Whether to score using the upper confidence bound"""

    def needs_column(self) -> bool:
        """Return True, since this strategy requires a column per agent."""
        return True

    async def precompute(
        self,
        agents: list[DataAgent],
        table_per_agent: dict[str, str] | None,
        column_per_agent: dict[str, str] | None,
        query: str = "",
    ) -> dict[str, hll_sketch]:
        """Precompute an HLL sketch of each agent's column values."""
        assert table_per_agent and column_per_agent

        async def _build(agent: DataAgent) -> tuple[str, hll_sketch]:
            """Fetch and sketch one agent's column values."""
            values = await asyncio.to_thread(
                database.fetch_column_values, agent, table_per_agent[agent.name], column_per_agent[agent.name]
            )
            sketch = hll_sketch(self.lg_k)
            for v in values:
                sketch.update(v)
            return agent.name, sketch

        return dict(await asyncio.gather(*[_build(a) for a in agents]))

    def init_state(self) -> hll_sketch:
        """Return an empty HLL sketch."""
        return hll_sketch(self.lg_k)

    def rank(
        self,
        remaining: list[ScoredDataAgent],
        precomputed: dict[str, hll_sketch],
        state: hll_sketch,
    ) -> list[ScoredDataAgent]:
        """Score remaining agents by the new distinct values they would add."""

        def score(s: ScoredDataAgent) -> float:
            """Score an agent by the new distinct values it would add."""
            union = hll_union(self.lg_k)
            union.update(state)
            union.update(precomputed[s.agent.name])
            result = union.get_result()
            estimate = result.get_upper_bound(DEFAULT_NUM_STD) if self.use_ucb else result.get_estimate()
            return max(0.0, estimate - state.get_estimate())

        for s in remaining:
            s.score = score(s)

        return sorted(remaining, key=lambda s: cast(float, s.score), reverse=True)

    def update(
        self,
        state: hll_sketch,
        values: list[str],
        agent_names: list[str] | None = None,
        precomputed: dict[str, hll_sketch] | None = None,
    ) -> hll_sketch:
        """Fold the seen column values into the running sketch."""
        for v in values:
            state.update(v)
        return state


# ---------------------------------------------------------------------------
# Theta
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ThetaSketchStrategy(SelectionStrategy[dict[str, compact_theta_sketch], update_theta_sketch]):
    """Ranks by estimated new unique values (agent sketch minus seen state)."""

    lg_k: int = DEFAULT_LG_K
    """Number of bits for the theta sketch precision"""
    use_ucb: bool = False
    """Whether to score using the upper confidence bound"""

    def needs_column(self) -> bool:
        """Return True, since this strategy requires a column per agent."""
        return True

    async def precompute(
        self,
        agents: list[DataAgent],
        table_per_agent: dict[str, str] | None,
        column_per_agent: dict[str, str] | None,
        query: str = "",
    ) -> dict[str, compact_theta_sketch]:
        """Precompute a compact theta sketch of each agent's column values."""
        assert table_per_agent and column_per_agent

        async def _build(agent: DataAgent) -> tuple[str, compact_theta_sketch] | None:
            """Fetch and sketch one agent's column values, or None if unresolved."""
            if agent.name not in table_per_agent or agent.name not in column_per_agent:
                return None
            values = await asyncio.to_thread(
                database.fetch_column_values, agent, table_per_agent[agent.name], column_per_agent[agent.name]
            )
            sketch = update_theta_sketch(self.lg_k)
            for v in values:
                sketch.update(v)
            return agent.name, sketch.compact()

        pairs = [p for p in await asyncio.gather(*[_build(a) for a in agents]) if p is not None]
        return dict(pairs)

    def init_state(self) -> update_theta_sketch:
        """Return an empty update theta sketch."""
        return update_theta_sketch(self.lg_k)

    def rank(
        self,
        remaining: list[ScoredDataAgent],
        precomputed: dict[str, compact_theta_sketch],
        state: update_theta_sketch,
    ) -> list[ScoredDataAgent]:
        """Score remaining agents by the estimated A-NOT-B new distinct values."""

        def score(s: ScoredDataAgent) -> float:
            """Score an agent by its estimated A-not-B new distinct values."""
            a_not_b = theta_a_not_b().compute(precomputed[s.agent.name], state)
            if self.use_ucb:
                return a_not_b.get_upper_bound(DEFAULT_NUM_STD)
            return a_not_b.get_estimate()

        for s in remaining:
            s.score = score(s)

        return sorted(remaining, key=lambda s: cast(float, s.score), reverse=True)

    def update(
        self,
        state: update_theta_sketch,
        values: list[str],
        agent_names: list[str] | None = None,
        precomputed: dict[str, compact_theta_sketch] | None = None,
    ) -> update_theta_sketch:
        """Fold the seen column values into the running sketch."""
        for v in values:
            state.update(v)
        return state


# ---------------------------------------------------------------------------
# MinHash
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MinHashState:
    """Running MinHash and seen-set state for MinHashStrategy."""

    minhash: MinHash
    """Running MinHash over values seen so far"""
    seen: set[str] = field(default_factory=set)
    """Set of distinct values seen so far"""


@dataclass(slots=True)
class MinHashStrategy(SelectionStrategy[dict[str, tuple[MinHash, int]], MinHashState]):
    """Ranks by estimated new unique values via containment estimation."""

    num_perm: int = DEFAULT_MIN_HASH_NUM_PERM
    """Number of permutations for the MinHash"""

    def needs_column(self) -> bool:
        """Return True, since this strategy requires a column per agent."""
        return True

    async def precompute(
        self,
        agents: list[DataAgent],
        table_per_agent: dict[str, str] | None,
        column_per_agent: dict[str, str] | None,
        query: str = "",
    ) -> dict[str, tuple[MinHash, int]]:
        """Precompute a MinHash and distinct-count over each agent's column values."""
        assert table_per_agent and column_per_agent

        async def _build(agent: DataAgent) -> tuple[str, tuple[MinHash, int]] | None:
            """Fetch and build a MinHash and distinct count for one agent."""
            if agent.name not in table_per_agent or agent.name not in column_per_agent:
                return None
            values = await asyncio.to_thread(
                database.fetch_column_values, agent, table_per_agent[agent.name], column_per_agent[agent.name]
            )
            mh = MinHash(num_perm=self.num_perm)
            for v in values:
                mh.update(v.encode("utf8"))
            return agent.name, (mh, len(set(values)))

        pairs = [p for p in await asyncio.gather(*[_build(a) for a in agents]) if p is not None]
        return dict(pairs)

    def init_state(self) -> MinHashState:
        """Return an empty MinHash state."""
        return MinHashState(minhash=MinHash(num_perm=self.num_perm))

    def rank(
        self,
        remaining: list[ScoredDataAgent],
        precomputed: dict[str, tuple[MinHash, int]],
        state: MinHashState,
    ) -> list[ScoredDataAgent]:
        """Score remaining agents by their estimated new distinct values."""

        def score(s: ScoredDataAgent) -> float:
            """Score an agent by its estimated new distinct values."""
            agent_mh, cardinality = precomputed[s.agent.name]
            jaccard = state.minhash.jaccard(agent_mh)
            seen_count = len(state.seen)
            intersection_est = jaccard * (cardinality + seen_count) / (1 + jaccard) if (1 + jaccard) > 0 else 0
            return max(0.0, cardinality - intersection_est)

        for s in remaining:
            s.score = score(s)

        return sorted(remaining, key=lambda s: cast(float, s.score), reverse=True)

    def update(
        self,
        state: MinHashState,
        values: list[str],
        agent_names: list[str] | None = None,
        precomputed: dict[str, tuple[MinHash, int]] | None = None,
    ) -> MinHashState:
        """Fold the seen column values into the running MinHash state."""
        for v in values:
            state.minhash.update(v.encode("utf8"))
            state.seen.add(v)
        return state

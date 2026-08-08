from dataclasses import dataclass, field
from typing import cast

from datasketch import MinHash
from datasketches import (
    compact_theta_sketch,  # ty: ignore[unresolved-import]
    hll_sketch,  # ty: ignore[unresolved-import]
    hll_union,  # ty: ignore[unresolved-import]
    theta_a_not_b,  # ty: ignore[unresolved-import]
    update_theta_sketch,  # ty: ignore[unresolved-import]
)
from sqlalchemy import text

from qcp_adaptive_termination.db import get_db_engine, quote_col, quote_table
from qcp_adaptive_termination.models import AgentSelection, DataAgent
from qcp_adaptive_termination.selection_strategy.strategy import SelectionStrategy

DEFAULT_LG_K = 12
DEFAULT_NUM_STD = 2
DEFAULT_MIN_HASH_NUM_PERM = 128


def fetch_column_values(
    data_agent: DataAgent, table_name: str, column_name: str
) -> list[str]:
    db_engine = get_db_engine(data_agent)
    with db_engine.connect() as conn:
        rows = conn.execute(
            text(
                f"SELECT {quote_col(column_name)} FROM {quote_table(table_name)} WHERE {quote_col(column_name)} IS NOT NULL AND {quote_col(column_name)} <> ''"
            )
        ).fetchall()
    return [str(row[0]) for row in rows]


# ---------------------------------------------------------------------------
# HLL
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class HllSketchStrategy(SelectionStrategy[hll_sketch, hll_sketch]):
    """
    Ranks by estimated new unique values each remaining agent would contribute.

    Precomputed: hll_sketch per agent.
    State: hll_sketch accumulating all values seen so far.
    Score: estimate(union(state, agent)) - estimate(state)  [i.e. new distinct est.]
    UCB variant uses upper_bound instead of point estimate.
    """

    lg_k: int = DEFAULT_LG_K
    use_ucb: bool = False

    async def precompute(
        self,
        agents: list[DataAgent],
        table_per_agent: dict[str, str] | None,
        column_per_agent: dict[str, str] | None,
        query: str = "",
    ) -> dict[str, hll_sketch]:
        assert table_per_agent and column_per_agent
        result: dict[str, hll_sketch] = {}
        for agent in agents:
            sketch = hll_sketch(self.lg_k)
            for v in fetch_column_values(
                agent, table_per_agent[agent.name], column_per_agent[agent.name]
            ):
                sketch.update(v)
            result[agent.name] = sketch
        return result

    def init_state(self) -> hll_sketch:
        return hll_sketch(self.lg_k)

    def rank(
        self,
        remaining: list[AgentSelection],
        precomputed: dict[str, hll_sketch],
        state: hll_sketch,
    ) -> list[AgentSelection]:
        def score(s: AgentSelection) -> float:
            union = hll_union(self.lg_k)
            union.update(state)
            union.update(precomputed[s.agent.name])
            result = union.get_result()
            estimate = (
                result.get_upper_bound(DEFAULT_NUM_STD)
                if self.use_ucb
                else result.get_estimate()
            )
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
        for v in values:
            state.update(v)
        return state


# ---------------------------------------------------------------------------
# Theta
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ThetaSketchStrategy(SelectionStrategy[compact_theta_sketch, update_theta_sketch]):
    """
    Ranks by estimated new unique values (A NOT B: agent sketch minus seen state).

    Precomputed: compact_theta_sketch per agent.
    State: update_theta_sketch accumulating all seen values.
    UCB variant uses upper_bound of the A-NOT-B result.
    """

    lg_k: int = DEFAULT_LG_K
    use_ucb: bool = False

    async def precompute(
        self,
        agents: list[DataAgent],
        table_per_agent: dict[str, str] | None,
        column_per_agent: dict[str, str] | None,
        query: str = "",
    ) -> dict[str, compact_theta_sketch]:
        assert table_per_agent and column_per_agent
        result: dict[str, compact_theta_sketch] = {}
        for agent in agents:
            if agent.name not in table_per_agent or agent.name not in column_per_agent:
                continue
            sketch = update_theta_sketch(self.lg_k)
            for v in fetch_column_values(
                agent, table_per_agent[agent.name], column_per_agent[agent.name]
            ):
                sketch.update(v)
            result[agent.name] = sketch.compact()
        return result

    def init_state(self) -> update_theta_sketch:
        return update_theta_sketch(self.lg_k)

    def rank(
        self,
        remaining: list[AgentSelection],
        precomputed: dict[str, compact_theta_sketch],
        state: update_theta_sketch,
    ) -> list[AgentSelection]:
        def score(s: AgentSelection) -> float:
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
        for v in values:
            state.update(v)
        return state


# ---------------------------------------------------------------------------
# MinHash
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MinHashState:
    minhash: MinHash
    seen: set[str] = field(default_factory=set)


@dataclass(slots=True)
class MinHashStrategy(SelectionStrategy[tuple[MinHash, int], MinHashState]):
    """
    Ranks by estimated new unique values via containment estimation.

    Precomputed: MinHash per agent.
    State: running MinHash of all seen values + seen set for exact cardinality.
    Score: (1 - Jaccard(state, agent)) * agent_cardinality  — estimated new entries.
    """

    num_perm: int = DEFAULT_MIN_HASH_NUM_PERM

    async def precompute(
        self,
        agents: list[DataAgent],
        table_per_agent: dict[str, str] | None,
        column_per_agent: dict[str, str] | None,
        query: str = "",
    ) -> dict[str, tuple[MinHash, int]]:
        assert table_per_agent and column_per_agent
        result: dict[str, tuple[MinHash, int]] = {}
        for agent in agents:
            if agent.name not in table_per_agent or agent.name not in column_per_agent:
                continue
            values = fetch_column_values(
                agent, table_per_agent[agent.name], column_per_agent[agent.name]
            )
            mh = MinHash(num_perm=self.num_perm)
            for v in values:
                mh.update(v.encode("utf8"))
            result[agent.name] = (mh, len(set(values)))
        return result

    def init_state(self) -> MinHashState:
        return MinHashState(minhash=MinHash(num_perm=self.num_perm))

    def rank(
        self,
        remaining: list[AgentSelection],
        precomputed: dict[str, tuple[MinHash, int]],
        state: MinHashState,
    ) -> list[AgentSelection]:
        def score(s: AgentSelection) -> float:
            agent_mh, cardinality = precomputed[s.agent.name]
            jaccard = state.minhash.jaccard(agent_mh)
            seen_count = len(state.seen)
            intersection_est = (
                jaccard * (cardinality + seen_count) / (1 + jaccard)
                if (1 + jaccard) > 0
                else 0
            )
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
        for v in values:
            state.minhash.update(v.encode("utf8"))
            state.seen.add(v)
        return state

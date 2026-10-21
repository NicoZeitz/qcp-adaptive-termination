"""Coverage-curve metrics for selection strategy evaluation."""

from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class EvaluationCurve:
    """Coverage curve for one strategy run."""

    x: list[float]
    y: list[float]
    auc: float


def compute_auc(coverages: list[float]) -> float:
    """
    Mean coverage over the full sequence of queried agents.

    `coverages` holds coverage after each queried agent, including agents queried
    after full coverage. Fixed horizon keeps scores comparable across strategies.
    """
    if not coverages:
        return 0.0
    return float(sum(coverages) / len(coverages))


def compute_relative_auc(auc: float, oracle_auc: float) -> float:
    """
    AUC normalized by the oracle AUC, so the oracle scores exactly 1.0.

    Degenerate case (oracle AUC = 0, i.e. nothing to cover) counts as perfect.
    """
    if oracle_auc <= 0.0:
        return 1.0
    return auc / oracle_auc


def build_curve(
    ordered_agents: list[str],
    data_cache: dict[str, pd.DataFrame | None],
    total_unique: int,
    key_cols: list[str],
) -> EvaluationCurve:
    """
    Build coverage curve from ordered agent query sequence.

    X = fraction of agents queried, Y = fraction of total unique rows covered.
    """
    agents_total = len(ordered_agents)
    if total_unique == 0 or agents_total == 0:
        return EvaluationCurve(x=[0.0, 1.0], y=[0.0, 0.0], auc=0.0)

    x: list[float] = [0.0]
    y: list[float] = [0.0]
    seen_keys: set[tuple] = set()

    for idx, agent_name in enumerate(ordered_agents):
        df = data_cache.get(agent_name)
        if df is not None and not df.empty:
            cols = [c for c in key_cols if c in df.columns]
            if cols:
                seen_keys.update(map(tuple, df[cols].drop_duplicates().values.tolist()))

        x.append((idx + 1) / agents_total)
        y.append(min(len(seen_keys) / total_unique, 1.0))

    area = compute_auc(y[1:])
    return EvaluationCurve(x=x, y=y, auc=area)


def compute_greedy_oracle(
    relevant_agent_names: list[str],
    data_cache: dict[str, pd.DataFrame | None],
    total_unique: int,
    key_cols: list[str],
) -> EvaluationCurve:
    """Greedy oracle: always pick the agent with the most new unique rows next.

    Each agent's key set is precomputed once up front, and the not-yet-seen
    remainder is tracked incrementally per agent, so we never re-build key sets
    nor recompute full cross-products on every greedy step. Same greedy result,
    far less work for large agent sets.
    """
    # Precompute each agent's key set ONCE (the expensive part).
    agent_keys: dict[str, frozenset] = {}
    for name in relevant_agent_names:
        df = data_cache.get(name)
        if df is None or df.empty:
            agent_keys[name] = frozenset()
        else:
            cols = [c for c in key_cols if c in df.columns]
            agent_keys[name] = (
                frozenset(map(tuple, df[cols].drop_duplicates().values.tolist())) if cols else frozenset()
            )

    remaining = list(relevant_agent_names)
    ordered: list[str] = []
    seen_keys: set[tuple] = set()
    # Incremental remainder per agent (monotonically shrinks as seen grows).
    not_seen = {name: set(agent_keys[name]) for name in remaining}
    not_seen_total = {name: len(not_seen[name]) for name in remaining}

    while remaining:
        # Pick the remaining agent contributing the most new keys.
        best_agent = max(remaining, key=lambda name: not_seen_total[name])
        best_keys = not_seen[best_agent]
        if not best_keys:
            # No new unique coverage from anyone left; remaining order is arbitrary.
            ordered.extend(remaining)
            break

        seen_keys.update(best_keys)
        remaining.remove(best_agent)
        ordered.append(best_agent)

        # Only agents whose not-yet-seen set overlaps what we just added change.
        for name in remaining:
            if agent_keys[name] & best_keys:
                not_seen[name].difference_update(best_keys)
                not_seen_total[name] = len(not_seen[name])

    return build_curve(ordered, data_cache, total_unique, key_cols)

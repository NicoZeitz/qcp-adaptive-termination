"""Evaluation utilities for selection strategy benchmarking."""

from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class EvaluationCurve:
    """Coverage curve for one strategy run."""

    x: list[float]
    y: list[float]
    auc: float


@dataclass(slots=True)
class EvaluationResult:
    """Full evaluation result for one strategy run."""

    query: str
    strategy: str
    curve: EvaluationCurve


def compute_auc(coverages: list[float]) -> float:
    """Mean coverage over the full sequence of queried agents.

    `coverages` holds coverage after each queried agent, including agents queried
    after full coverage. Fixed horizon keeps scores comparable across strategies.
    """
    if not coverages:
        return 0.0
    return float(sum(coverages) / len(coverages))


def compute_relative_auc(auc: float, oracle_auc: float) -> float:
    """AUC normalized by the oracle AUC, so the oracle scores exactly 1.0.

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
    """Build coverage curve from ordered agent query sequence.

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
    """Greedy oracle: always pick the agent with the most new unique rows next."""
    remaining = list(relevant_agent_names)
    ordered: list[str] = []
    seen_keys: set[tuple] = set()

    while remaining:
        best_agent: str | None = None
        best_new = -1

        for agent_name in remaining:
            df = data_cache.get(agent_name)
            if df is None or df.empty:
                new_unique = 0
            else:
                cols = [c for c in key_cols if c in df.columns]
                new_unique = len(set(map(tuple, df[cols].drop_duplicates().values.tolist())) - seen_keys) if cols else 0

            if new_unique > best_new:
                best_new = new_unique
                best_agent = agent_name

        if best_agent is None:
            break

        remaining.remove(best_agent)
        ordered.append(best_agent)

        df = data_cache.get(best_agent)
        if df is not None and not df.empty:
            cols = [c for c in key_cols if c in df.columns]
            if cols:
                seen_keys.update(map(tuple, df[cols].drop_duplicates().values.tolist()))

    return build_curve(ordered, data_cache, total_unique, key_cols)

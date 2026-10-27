"""Estimator-trajectory metrics for the k-warmup study.

Builds, for one ordering, the incremental estimator trajectory (N_hat, bounds,
coverage, observed vs true) as each agent's rows are folded in, then extracts
the flatness and error-based stopping indices ``k``.
"""

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from qcp_adaptive_termination.estimator import Estimator
from qcp_adaptive_termination.observability import context
from scripts.utils.canonical import query_agent_canonical

MetaMapping = dict[str, dict[str, dict[str, Any]]]
"""Canonical table -> agent slug -> per-agent table metadata."""


@dataclass(slots=True)
class EstimatorTrajectory:
    """Incremental estimator state after each queried agent for one ordering."""

    agents_ordered: list[str]
    """Agents in the order they were queried"""
    observed: list[int] = field(default_factory=list)
    """Observed unique count after each agent"""
    estimated: list[float] = field(default_factory=list)
    """Estimated total unique count (N_hat) after each agent"""
    lower: list[float] = field(default_factory=list)
    """Estimated count lower bound after each agent"""
    upper: list[float] = field(default_factory=list)
    """Estimated count upper bound after each agent"""
    coverage: list[float] = field(default_factory=list)
    """Good-Turing sample coverage after each agent"""
    error: list[float] = field(default_factory=list)
    """Relative error |N_hat - true| / true after each agent"""
    true_count: int = 0
    """True distinct count across all agents (SELECT * union)"""

    @property
    def agents_total(self) -> int:
        """Total number of agents in the ordering."""
        return len(self.agents_ordered)

    def to_frame(self) -> pd.DataFrame:
        """Serialize the trajectory to a per-agent dataframe."""
        return pd.DataFrame(
            {
                "agents_queried": list(range(1, len(self.observed) + 1)),
                "observed": self.observed,
                "estimated": self.estimated,
                "lower": self.lower,
                "upper": self.upper,
                "coverage": self.coverage,
                "rel_error": self.error,
                "true_count": self.true_count,
            }
        )


def build_estimator_trajectory(
    ordered_agents: list[str],
    data_cache: dict[str, pd.DataFrame | None],
    key_cols: list[str],
    true_count: int,
) -> EstimatorTrajectory:
    """Build the incremental estimator trajectory for an ordered agent list.

    Each agent's SELECT * rows are folded into an Estimator (as one source, like
    the live multi-agent loop), recording N_hat/bounds/coverage/error after each.

    Args:
        ordered_agents: agents in query order
        data_cache: agent name to its SELECT * result (canonical columns)
        key_cols: key columns defining an entity
        true_count: true distinct count across all agents

    Returns:
        (EstimatorTrajectory): the per-agent estimator trajectory
    """
    estimator: Estimator | None = None
    traj = EstimatorTrajectory(agents_ordered=list(ordered_agents), true_count=true_count)

    for agent_name in ordered_agents:
        df = data_cache.get(agent_name)
        if df is not None and not df.empty:
            estimator = _fold_agent(estimator, df, agent_name, key_cols)

        observed = estimator.sample_count() if estimator is not None else 0
        estimated = estimator.estimated_count() if estimator is not None else float("nan")
        lower, upper = estimator.estimated_count_bounds() if estimator is not None else (float("nan"), float("nan"))
        coverage = estimator.sample_coverage() if estimator is not None else float("nan")
        err = abs(estimated - true_count) / true_count if true_count > 0 else float("nan")

        traj.observed.append(observed)
        traj.estimated.append(estimated)
        traj.lower.append(lower)
        traj.upper.append(upper)
        traj.coverage.append(coverage)
        traj.error.append(err)

    return traj


def _fold_agent(
    estimator: Estimator | None,
    df: pd.DataFrame,
    agent_name: str,
    key_cols: list[str],
) -> Estimator:
    """Fold one agent's rows into the estimator, creating it if needed."""
    from qcp_adaptive_termination.stages.execution.multi_agent.estimator_utils import (
        dataframe_to_data_items,
    )

    items = dataframe_to_data_items(df, key_cols, None, agent_name)
    if estimator is None:
        return Estimator([[item] for item in items], context.get_config().completeness.estimator)
    estimator.add_source(items)
    return estimator


def flatness_k(traj: EstimatorTrajectory, rel_threshold: float) -> int | None:
    """First agent index where the next one-step N_hat change is below the threshold.

    Args:
        traj: the estimator trajectory
        rel_threshold: maximum relative one-step change that counts as flat

    Returns:
        (int | None): 1-based agent index, or None if never flat
    """
    for i in range(1, len(traj.estimated)):
        prev, curr = traj.estimated[i - 1], traj.estimated[i]
        if prev > 0 and curr > 0 and abs(curr - prev) / prev <= rel_threshold:
            return i + 1  # 1-based agent index that produced ``curr``
    return None


def error_k(traj: EstimatorTrajectory, rel_threshold: float) -> int | None:
    """First agent index where the relative error vs true count is below the threshold.

    Args:
        traj: the estimator trajectory
        rel_threshold: maximum relative error that counts as converged

    Returns:
        (int | None): 1-based agent index, or None if never converged
    """
    for i, err in enumerate(traj.error):
        if not 0 <= err <= rel_threshold:
            continue
        return i + 1  # 1-based agent index
    return None


def true_count_from_cache(data_cache: dict[str, pd.DataFrame | None], key_cols: list[str]) -> int:
    """Distinct count across all agents' SELECT * results, deduped on the key columns.

    Args:
        data_cache: agent name to its SELECT * result
        key_cols: key columns defining an entity

    Returns:
        (int): the true distinct count
    """
    frames = [df for df in data_cache.values() if df is not None and not df.empty]
    if not frames:
        return 0
    combined = pd.concat(frames, ignore_index=True)
    cols = [c for c in key_cols if c in combined.columns]
    if not cols:
        cols = list(combined.columns)
    return int(len(combined[cols].drop_duplicates()))


def select_all_sql(table: str) -> str:
    """SELECT * statement for a canonical table."""
    return f"SELECT * FROM {table}"


def canonical_key_cols(meta_mapping: MetaMapping, table: str) -> list[str]:
    """Key columns for a table = its first canonical column (the entity id).

    Args:
        meta_mapping: canonical-to-actual schema mapping
        table: the canonical table name

    Returns:
        (list[str]): the key columns
    """
    slugs = meta_mapping.get(table) or {}
    first = next(iter(slugs.values()), None)
    if first is None:
        return []
    cols = list((first or {}).get("columns", {}).keys())
    return cols[:1]


def load_agent_cache(
    relevant_agents,
    canonical_sql: str,
    canonical_table: str,
    meta_mapping: MetaMapping,
    data_dir,
) -> dict[str, pd.DataFrame | None]:
    """Load each relevant agent's SELECT * result into a cache keyed by agent name.

    Args:
        relevant_agents: the agents to query
        canonical_sql: canonical SELECT * SQL
        canonical_table: the canonical table
        meta_mapping: canonical-to-actual schema mapping
        data_dir: data directory

    Returns:
        (dict[str, pd.DataFrame | None]): agent name to its result dataframe
    """
    from concurrent.futures import ThreadPoolExecutor

    cache: dict[str, pd.DataFrame | None] = {}
    with ThreadPoolExecutor() as pool:
        futures = {
            agent.name: pool.submit(
                query_agent_canonical, agent, canonical_sql, canonical_table, meta_mapping, data_dir
            )
            for agent in relevant_agents
        }
        for name, future in futures.items():
            cache[name] = future.result()
    return cache

from dataclasses import dataclass

from sqlalchemy import text

from common.model.agent.data_agent import DataAgent
from common.data_agent.get_db_engine import get_db_engine
from stages.execution.selection.strategy import StaticSelectionStrategy


def scalar(data_agent: DataAgent, query: str) -> float:
    db_engine = get_db_engine(data_agent)
    with db_engine.connect() as conn:
        result = conn.execute(text(query)).scalar()
    return float(result or 0.0)


@dataclass
class NullCoverageStrategy(StaticSelectionStrategy[float]):
    """Fraction of non-null values — favors agents with the most complete data."""

    async def precompute(
        self,
        agents: list[DataAgent],
        table_per_agent: dict[str, str] | None,
        column_per_agent: dict[str, str] | None,
        query: str = "",
    ) -> dict[str, float]:
        assert table_per_agent and column_per_agent
        return {
            a.name: scalar(
                a,
                f'SELECT COUNT("{column_per_agent[a.name]}") * 1.0 / NULLIF(COUNT(*), 0)'
                f' FROM "{table_per_agent[a.name]}"',
            )
            for a in agents
            if a.name in table_per_agent and a.name in column_per_agent
        }


@dataclass
class ValueRangeStrategy(StaticSelectionStrategy[float]):
    """Numeric spread (MAX - MIN) — favors agents whose column spans the widest range."""

    async def precompute(
        self,
        agents: list[DataAgent],
        table_per_agent: dict[str, str] | None,
        column_per_agent: dict[str, str] | None,
        query: str = "",
    ) -> dict[str, float]:
        assert table_per_agent and column_per_agent
        return {
            a.name: scalar(
                a,
                f'SELECT MAX("{column_per_agent[a.name]}") - MIN("{column_per_agent[a.name]}")'
                f' FROM "{table_per_agent[a.name]}"'
                f' WHERE "{column_per_agent[a.name]}" IS NOT NULL',
            )
            for a in agents
            if a.name in table_per_agent and a.name in column_per_agent
        }

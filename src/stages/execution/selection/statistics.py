from dataclasses import dataclass

from sqlalchemy import text

from qtypes.data_agent import DataAgent
from stages.common.data_agent.sql_data_agent.get_db_engine import get_db_engine
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
        table_name: str | None,
        column_name: str | None,
        query: str = "",
    ) -> dict[str, float]:
        assert table_name and column_name
        return {
            a.name: scalar(
                a,
                f'SELECT COUNT("{column_name}") * 1.0 / NULLIF(COUNT(*), 0)'
                f' FROM "{table_name}"',
            )
            for a in agents
        }


@dataclass
class ValueRangeStrategy(StaticSelectionStrategy[float]):
    """Numeric spread (MAX - MIN) — favors agents whose column spans the widest range."""

    async def precompute(
        self,
        agents: list[DataAgent],
        table_name: str | None,
        column_name: str | None,
        query: str = "",
    ) -> dict[str, float]:
        assert table_name and column_name
        return {
            a.name: scalar(
                a,
                f'SELECT MAX("{column_name}") - MIN("{column_name}")'
                f' FROM "{table_name}" WHERE "{column_name}" IS NOT NULL',
            )
            for a in agents
        }

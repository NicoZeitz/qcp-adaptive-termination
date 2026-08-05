from dataclasses import dataclass

from sqlalchemy import text

from model.agent.data_agent import DataAgent
from common.data_agent.get_db_engine import get_db_engine
from common.selection_strategy.strategy import StaticSelectionStrategy


def fetch_count(data_agent: DataAgent, table_name: str) -> int:
    db_engine = get_db_engine(data_agent)
    with db_engine.connect() as conn:
        result = conn.execute(text(f'SELECT COUNT(*) FROM "{table_name}"')).scalar()
    return int(result or 0)


def fetch_distinct_count(
    data_agent: DataAgent, table_name: str, column_name: str
) -> int:
    db_engine = get_db_engine(data_agent)
    with db_engine.connect() as conn:
        result = conn.execute(
            text(f'SELECT COUNT(DISTINCT "{column_name}") FROM "{table_name}"')
        ).scalar()
    return int(result or 0)


@dataclass
class CountStrategy(StaticSelectionStrategy[int]):
    async def precompute(
        self,
        agents: list[DataAgent],
        table_per_agent: dict[str, str] | None,
        column_per_agent: dict[str, str] | None,
        query: str = "",
    ) -> dict[str, int]:
        assert table_per_agent
        return {
            a.name: fetch_count(a, table_per_agent[a.name])
            for a in agents
            if a.name in table_per_agent
        }


@dataclass
class DistinctCountStrategy(StaticSelectionStrategy[int]):
    async def precompute(
        self,
        agents: list[DataAgent],
        table_per_agent: dict[str, str] | None,
        column_per_agent: dict[str, str] | None,
        query: str = "",
    ) -> dict[str, int]:
        assert table_per_agent and column_per_agent
        return {
            a.name: fetch_distinct_count(
                a, table_per_agent[a.name], column_per_agent[a.name]
            )
            for a in agents
            if a.name in table_per_agent and a.name in column_per_agent
        }

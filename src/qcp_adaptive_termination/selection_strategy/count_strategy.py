import qcp_adaptive_termination.database as database
from qcp_adaptive_termination.models import DataAgent
from qcp_adaptive_termination.selection_strategy.strategy import StaticSelectionStrategy


class CountStrategy(StaticSelectionStrategy[int]):
    def needs_table(self) -> bool:
        return True

    async def precompute(
        self,
        agents: list[DataAgent],
        table_per_agent: dict[str, str] | None,
        column_per_agent: dict[str, str] | None,
        query: str = "",
    ) -> dict[str, int]:
        assert table_per_agent

        return {
            a.name: database.fetch_count(a, table_per_agent[a.name]) if a.name in table_per_agent else 0 for a in agents
        }


class DistinctCountStrategy(StaticSelectionStrategy[int]):
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
    ) -> dict[str, int]:
        assert table_per_agent and column_per_agent

        return {
            a.name: database.fetch_distinct_count(a, table_per_agent[a.name], column_per_agent[a.name])
            if a.name in table_per_agent and a.name in column_per_agent
            else 0
            for a in agents
        }

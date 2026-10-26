"""Count-based static selection strategy."""

import asyncio
from dataclasses import dataclass

import qcp_adaptive_termination.database as database
from qcp_adaptive_termination.models import DataAgent
from qcp_adaptive_termination.selection_strategy.strategy import StaticSelectionStrategy


@dataclass(slots=True)
class CountStrategy(StaticSelectionStrategy[int]):
    """Ranks agents by total row count."""

    def needs_table(self) -> bool:
        """Return True, since this strategy requires a table per agent."""
        return True

    async def precompute(
        self,
        agents: list[DataAgent],
        table_per_agent: dict[str, str] | None,
        column_per_agent: dict[str, str] | None,
        query: str = "",
    ) -> dict[str, int]:
        """Precompute per-agent row counts."""
        assert table_per_agent

        mapped_agents = [agent for agent in agents if agent.name in table_per_agent]
        counts = await asyncio.gather(
            *(asyncio.to_thread(database.fetch_count, agent, table_per_agent[agent.name]) for agent in mapped_agents)
        )
        count_by_agent = dict(zip((agent.name for agent in mapped_agents), counts, strict=True))
        return {agent.name: count_by_agent.get(agent.name, 0) for agent in agents}


@dataclass(slots=True)
class DistinctCountStrategy(StaticSelectionStrategy[int]):
    """Ranks agents by distinct value count."""

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
    ) -> dict[str, int]:
        """Precompute per-agent distinct value counts."""
        assert table_per_agent and column_per_agent

        mapped_agents = [agent for agent in agents if agent.name in table_per_agent and agent.name in column_per_agent]
        counts = await asyncio.gather(
            *(
                asyncio.to_thread(
                    database.fetch_distinct_count,
                    agent,
                    table_per_agent[agent.name],
                    column_per_agent[agent.name],
                )
                for agent in mapped_agents
            )
        )
        count_by_agent = dict(zip((agent.name for agent in mapped_agents), counts, strict=True))
        return {agent.name: count_by_agent.get(agent.name, 0) for agent in agents}

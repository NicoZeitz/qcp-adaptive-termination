from common.model.agent.data_agent import DataAgent
from stages.execution.selection.strategy import StaticSelectionStrategy


class SequentialStrategy(StaticSelectionStrategy[float]):
    ascending: bool = True

    async def precompute(
        self,
        agents: list[DataAgent],
        table_per_agent: dict[str, str] | None,
        column_per_agent: dict[str, str] | None,
        query: str = "",
    ) -> dict[str, float]:
        return {a.name: float(i) for i, a in enumerate(agents)}

from qtypes.data_agent import DataAgent
from stages.execution.selection.strategy import StaticSelectionStrategy


class SequentialStrategy(StaticSelectionStrategy[float]):
    ascending: bool = True

    async def precompute(
        self,
        agents: list[DataAgent],
        table_name: str | None,
        column_name: str | None,
        query: str = "",
    ) -> dict[str, float]:
        return {a.name: float(i) for i, a in enumerate(agents)}

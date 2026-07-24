from qtypes.data_agent import DataAgent


from dataclasses import dataclass


@dataclass
class AgentSelection:
    agent: DataAgent
    table_name: str | None
    column_name: str | None

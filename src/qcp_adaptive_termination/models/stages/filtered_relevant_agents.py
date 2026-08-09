from pydantic import BaseModel, ConfigDict

from qcp_adaptive_termination.models.agent import DataAgent


class FilteredRelevantAgents(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    data_agents: list[DataAgent]
    relevant_data_agents: list[DataAgent]

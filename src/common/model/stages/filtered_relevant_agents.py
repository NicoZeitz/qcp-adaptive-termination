from common.model.agent.data_agent import DataAgent


from pydantic import BaseModel, ConfigDict


class FilteredRelevantAgents(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    data_agents: list[DataAgent]
    relevant_data_agents: list[DataAgent]

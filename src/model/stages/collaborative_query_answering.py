from model.agent.data_agent import DataAgent


from pydantic import BaseModel, ConfigDict


class CollaborativeQueryAnswering(BaseModel):
    model_config = ConfigDict(extra="allow")

    query: str
    data_agents: list[DataAgent]

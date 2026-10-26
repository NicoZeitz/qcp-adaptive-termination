"""Input model for the collaborative query answering stage."""

from pydantic import BaseModel, ConfigDict, Field

from qcp_adaptive_termination.models.agent import DataAgent


class CollaborativeQueryAnswering(BaseModel):
    """Input to the collaborative query answering stage."""

    model_config = ConfigDict(extra="allow")

    query: str = Field(description="The user's query to answer.")
    data_agents: list[DataAgent] = Field(description="The data agents to query.")

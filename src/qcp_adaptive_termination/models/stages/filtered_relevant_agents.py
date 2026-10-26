"""Model for the output of the agent filtering stage."""

from pydantic import BaseModel, ConfigDict, Field

from qcp_adaptive_termination.models.agent import DataAgent


class FilteredRelevantAgents(BaseModel):
    """Relevant agents filtered from the full data agent list."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(description="The user's query.")
    data_agents: list[DataAgent] = Field(description="All data agents considered.")
    relevant_data_agents: list[DataAgent] = Field(description="The subset of agents deemed relevant.")

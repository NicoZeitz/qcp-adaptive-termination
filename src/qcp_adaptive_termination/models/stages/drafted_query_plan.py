"""Model for a drafted query plan with its relevant agents."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from qcp_adaptive_termination.models.agent import DataAgent
from qcp_adaptive_termination.models.query_plan import QueryPlan


class DraftedQueryPlan(BaseModel):
    """Drafted query plan together with the relevant data agents."""

    model_config = ConfigDict(extra="forbid")

    relevant_data_agents: list[DataAgent] = Field(description="Agents relevant to answering the query")
    # All available data agents.
    data_agents: list[DataAgent] = Field(description="All available data agents")
    # The drafted query plan.
    query_plan: QueryPlan = Field(description="The drafted query plan")
    planning_continuation: list[Any] | None = Field(
        default=None,
        description="Raw conversation continuation from the initial plan draft, threaded into replan history",
    )

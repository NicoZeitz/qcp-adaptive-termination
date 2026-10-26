from typing import Any

from pydantic import BaseModel, ConfigDict

from qcp_adaptive_termination.models.agent import DataAgent
from qcp_adaptive_termination.models.query_plan import QueryPlan


class DraftedQueryPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relevant_data_agents: list[DataAgent]
    data_agents: list[DataAgent]
    query_plan: QueryPlan
    planning_continuation: list[Any] | None = None
    """Raw conversation continuation from the initial plan draft, threaded into replan history."""

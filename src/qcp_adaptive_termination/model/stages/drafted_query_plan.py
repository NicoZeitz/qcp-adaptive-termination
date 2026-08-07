from qcp_adaptive_termination.model.agent.data_agent import DataAgent
from qcp_adaptive_termination.model.query_plan.query_plan import QueryPlan


from pydantic import BaseModel, ConfigDict


class DraftedQueryPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relevant_data_agents: list[DataAgent]
    data_agents: list[DataAgent]
    query_plan: QueryPlan

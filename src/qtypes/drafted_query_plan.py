from qtypes.data_agent import DataAgent
from qtypes.query_plan import QueryPlan


from pydantic import BaseModel, ConfigDict


class DraftedQueryPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relevant_data_agents: list[DataAgent]
    data_agents: list[DataAgent]
    query_plan: QueryPlan

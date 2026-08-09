from pydantic import BaseModel, ConfigDict, Field

from qcp_adaptive_termination.models.query_plan.query_dependency import QueryDependency
from qcp_adaptive_termination.models.result_type import SerializedOptionalResultType


class BaseQueryPlanStep(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    id: str
    query: str
    dependencies: list[QueryDependency]
    answer: SerializedOptionalResultType = None


class SingleAgentQueryPlanStep(BaseQueryPlanStep):
    to_agent: str

    @property
    def has_multiple_to_agents(self) -> bool:
        return False

    @property
    def to_agents(self) -> list[str]:
        return [self.to_agent]


class MultiAgentQueryPlanStep(BaseQueryPlanStep):
    to_agent: list[str] = Field(min_length=2)
    merge_task: str

    @property
    def has_multiple_to_agents(self) -> bool:
        return True

    @property
    def to_agents(self) -> list[str]:
        return self.to_agent


type QueryPlanStep = SingleAgentQueryPlanStep | MultiAgentQueryPlanStep

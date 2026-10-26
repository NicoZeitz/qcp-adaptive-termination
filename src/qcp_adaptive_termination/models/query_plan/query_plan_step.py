"""Models for query plan steps."""

from pydantic import BaseModel, ConfigDict, Field

from qcp_adaptive_termination.models.query_plan.query_dependency import QueryDependency
from qcp_adaptive_termination.models.result_type import SerializedOptionalResultType


class BaseQueryPlanStep(BaseModel):
    """Base model for a query plan step."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    id: str = Field(description="Unique id of the step.")
    query: str = Field(description="The query to execute in this step.")
    dependencies: list[QueryDependency] = Field(description="Dependencies on other steps' results.")
    answer: SerializedOptionalResultType = Field(default=None, description="The answer produced by the step, if any.")


class SingleAgentQueryPlanStep(BaseQueryPlanStep):
    """A query plan step addressed to a single agent."""

    to_agent: str = Field(description="The single agent the step is addressed to.")

    @property
    def has_multiple_to_agents(self) -> bool:
        """Return whether the step targets multiple agents."""
        return False

    @property
    def to_agents(self) -> list[str]:
        """Return the agents the step is addressed to."""
        return [self.to_agent]


class MultiAgentQueryPlanStep(BaseQueryPlanStep):
    """A query plan step addressed to multiple agents with a merge task."""

    to_agent: list[str] = Field(min_length=2, description="The agents the step is addressed to.")
    merge_task: str = Field(description="Instruction for merging the agents' outputs.")

    @property
    def has_multiple_to_agents(self) -> bool:
        """Return whether the step targets multiple agents."""
        return True

    @property
    def to_agents(self) -> list[str]:
        """Return the agents the step is addressed to."""
        return self.to_agent


type QueryPlanStep = SingleAgentQueryPlanStep | MultiAgentQueryPlanStep

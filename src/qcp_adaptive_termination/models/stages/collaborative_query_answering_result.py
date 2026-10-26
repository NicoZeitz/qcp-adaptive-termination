"""Result model for the collaborative query answering stage."""

from pydantic import BaseModel, ConfigDict, Field

from qcp_adaptive_termination.models.query_plan import QueryPlan
from qcp_adaptive_termination.models.result_type import SerializedResultType
from qcp_adaptive_termination.models.uncertainty import StepUncertainty


class CollaborativeQueryAnsweringResult(BaseModel):
    """Result of the collaborative query answering stage."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    result: SerializedResultType = Field(description="The final result of the stage.")
    query_plan: QueryPlan = Field(description="The query plan that was executed.")
    step_uncertainties: dict[str, StepUncertainty] = Field(
        default_factory=dict, description="Recorded uncertainty per sub-step, keyed by step id."
    )

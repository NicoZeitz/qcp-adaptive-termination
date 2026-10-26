"""Result models for executing a single query plan step."""

from pydantic import BaseModel, ConfigDict, Field

from qcp_adaptive_termination.models.execution.query_plan_step_execution_result import (
    FailedExecution,
)
from qcp_adaptive_termination.models.result_type import SerializedOptionalResultType
from qcp_adaptive_termination.models.uncertainty import StepUncertainty


class ExecutionResult(BaseModel):
    """Result of executing a single query plan step."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    success: bool = Field(description="Whether the step executed successfully.")
    result: SerializedOptionalResultType = Field(default=None, description="The data produced by the step, if any.")
    failed_execution: FailedExecution | None = Field(
        default=None, description="Details of the failure, if the step failed."
    )
    step_uncertainties: dict[str, StepUncertainty] = Field(
        default_factory=dict, description="Recorded uncertainty per sub-step, keyed by step id."
    )

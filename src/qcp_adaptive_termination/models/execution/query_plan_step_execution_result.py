"""Outcome models for executing a query plan step."""

from pydantic import BaseModel, ConfigDict, Field

from qcp_adaptive_termination.models.need_type import SerializedNeedList
from qcp_adaptive_termination.models.query_plan import QueryPlanStep
from qcp_adaptive_termination.models.result_type import (
    SerializedOptionalResultType,
    SerializedResultType,
)
from qcp_adaptive_termination.models.uncertainty import StepUncertainty


class SuccessfulExecution(BaseModel):
    """Outcome of a query plan step that executed successfully."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    success: bool = Field(default=True, description="Whether the step executed successfully.")
    step: QueryPlanStep = Field(description="The query plan step that ran.")
    result: SerializedResultType = Field(description="The data produced by the step.")
    execution_finished: bool = Field(
        default=False, description="Whether the step finished executing without interruption."
    )
    step_uncertainties: list[StepUncertainty] = Field(
        default_factory=list, description="Recorded uncertainty per sub-step."
    )


class FailedExecution(BaseModel):
    """Outcome of a query plan step that failed and needs more information."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    success: bool = Field(default=False, description="Always false for a failed execution.")
    step: QueryPlanStep = Field(description="The query plan step that failed.")
    needs: SerializedNeedList = Field(description="What information is missing to complete the step.")
    partial_answer: SerializedOptionalResultType = Field(
        default=None, description="Partial answer produced before the failure."
    )
    has_partial_answer: bool = Field(default=False, description="Whether a partial answer is available.")


QueryPlanStepExecutionResult = SuccessfulExecution | FailedExecution

from pydantic import BaseModel, ConfigDict, Field

from qcp_adaptive_termination.models.execution.query_plan_step_execution_result import (
    FailedExecution,
)
from qcp_adaptive_termination.models.result_type import SerializedOptionalResultType
from qcp_adaptive_termination.models.uncertainty import StepUncertainty


class ExecutionResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    success: bool
    result: SerializedOptionalResultType = None
    failed_execution: FailedExecution | None = None
    step_uncertainties: list[StepUncertainty] = Field(default_factory=list)

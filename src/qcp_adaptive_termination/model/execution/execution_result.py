from pydantic import BaseModel, ConfigDict

from qcp_adaptive_termination.model.execution.query_plan_step_execution_result import (
    FailedExecution,
)
from qcp_adaptive_termination.model.serializers import SerializedOptionalResultType


class ExecutionResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    success: bool
    result: SerializedOptionalResultType = None
    failed_execution: FailedExecution | None = None

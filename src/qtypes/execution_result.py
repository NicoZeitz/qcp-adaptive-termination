from pydantic import BaseModel, ConfigDict

from qtypes.query_plan_step_execution_result import FailedExecution
from qtypes.serializers import SerializedOptionalResultType


class ExecutionResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    success: bool
    result: SerializedOptionalResultType = None
    failed_execution: FailedExecution | None = None

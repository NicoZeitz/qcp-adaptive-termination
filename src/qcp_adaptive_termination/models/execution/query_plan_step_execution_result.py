from pydantic import BaseModel, ConfigDict

from qcp_adaptive_termination.models.need_type import SerializedNeedList
from qcp_adaptive_termination.models.query_plan import QueryPlanStep
from qcp_adaptive_termination.models.result_type import (
    SerializedOptionalResultType,
    SerializedResultType,
)


class SuccessfulExecution(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    success: bool = True
    step: QueryPlanStep
    result: SerializedResultType
    execution_finished: bool = False


class FailedExecution(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    success: bool = False
    step: QueryPlanStep
    needs: SerializedNeedList
    partial_answer: SerializedOptionalResultType = None
    has_partial_answer: bool = False


QueryPlanStepExecutionResult = SuccessfulExecution | FailedExecution

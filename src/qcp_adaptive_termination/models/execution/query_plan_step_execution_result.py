from abc import ABC

from pydantic import BaseModel, ConfigDict

from qcp_adaptive_termination.models.query_plan.query_plan_step import QueryPlanStep
from qcp_adaptive_termination.models.serializers import (
    SerializedNeedList,
    SerializedOptionalResultType,
    SerializedResultType,
)


class QueryPlanStepExecutionResult(ABC, BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    success: bool
    step: QueryPlanStep


class SuccessfulExecution(QueryPlanStepExecutionResult, BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    success: bool = True
    result: SerializedResultType
    execution_finished: bool = False


class FailedExecution(QueryPlanStepExecutionResult, BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    success: bool = False
    needs: SerializedNeedList
    partial_answer: SerializedOptionalResultType = None
    has_partial_answer: bool = False

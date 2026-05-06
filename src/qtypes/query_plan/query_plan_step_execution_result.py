from dataclasses import dataclass


from qtypes.query_plan.need import NeedType
from qtypes.query_plan.query_plan import QueryPlanStep
from qtypes.query_plan.result import ResultType


@dataclass
class QueryPlanStepExecutionResult:
    step: QueryPlanStep


@dataclass
class SuccessfulExecution(QueryPlanStepExecutionResult):
    result: ResultType
    execution_finished: bool = False


@dataclass
class FailedExecution(QueryPlanStepExecutionResult):
    needs: list[NeedType]
    partial_answer: ResultType | None = None
    has_partial_answer: bool = False

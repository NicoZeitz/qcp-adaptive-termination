from abc import ABC, abstractmethod
from dataclasses import dataclass

from pydantic import JsonValue


from qtypes.query_plan.need import NeedType, needs_to_span_data
from qtypes.query_plan.query_plan import QueryPlanStep
from qtypes.query_plan.result import ResultType, result_to_span_data


@dataclass
class QueryPlanStepExecutionResult(ABC):
    step: QueryPlanStep

    @abstractmethod
    def to_span_data(self) -> JsonValue:
        pass


@dataclass
class SuccessfulExecution(QueryPlanStepExecutionResult):
    result: ResultType
    execution_finished: bool = False

    def to_span_data(self) -> JsonValue:
        return {
            "step": self.step.to_span_data(),
            "result": result_to_span_data(self.result),
            "execution_finished": self.execution_finished,
        }


@dataclass
class FailedExecution(QueryPlanStepExecutionResult):
    needs: list[NeedType]
    partial_answer: ResultType | None = None
    has_partial_answer: bool = False

    def to_span_data(self) -> JsonValue:
        return {
            "step": self.step.to_span_data(),
            "needs": needs_to_span_data(self.needs),
            "has_partial_answer": self.has_partial_answer,
            "partial_answer": result_to_span_data(self.partial_answer)
            if self.partial_answer is not None
            else None,
        }

"""
Execution result and outcome models.

Defines the outcome types (`StopDecision`, `RaiseNeedOutcome`), per-step
execution results (`SuccessfulExecution`, `FailedExecution`), and the final
`ExecutionResult`.
"""

from qcp_adaptive_termination.models.execution.additional_information import (
    AdditionalInformation,
)
from qcp_adaptive_termination.models.execution.execution_result import ExecutionResult
from qcp_adaptive_termination.models.execution.outcome import (
    RaiseNeedOutcome,
    StopDecision,
)
from qcp_adaptive_termination.models.execution.partial_query_answer import (
    PartialQueryAnswer,
)
from qcp_adaptive_termination.models.execution.query_plan_step_execution_result import (
    FailedExecution,
    QueryPlanStepExecutionResult,
    SuccessfulExecution,
)

__all__ = [
    "AdditionalInformation",
    "ExecutionResult",
    "FailedExecution",
    "PartialQueryAnswer",
    "QueryPlanStepExecutionResult",
    "RaiseNeedOutcome",
    "StopDecision",
    "SuccessfulExecution",
]

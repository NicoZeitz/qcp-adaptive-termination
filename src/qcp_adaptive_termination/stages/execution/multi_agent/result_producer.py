"""Result-producing agent: turns the merged result list into a final answer."""

import math
from typing import Any

import pandas as pd
from agents import (
    FunctionToolResult,
    RunContextWrapper,
    ToolsToFinalOutputResult,
)
from pydantic import BaseModel, ConfigDict, Field

import qcp_adaptive_termination.hooks as hooks
from qcp_adaptive_termination.code_execution.code_execution_environment import (
    CodeExecutionEnvironment,
)
from qcp_adaptive_termination.estimator import Estimator
from qcp_adaptive_termination.hooks import HookEvent
from qcp_adaptive_termination.llm import runner
from qcp_adaptive_termination.models import (
    AgentRole,
    AlignedResultList,
    SerializedOptionalResultType,
    StopReason,
    result_type,
)
from qcp_adaptive_termination.observability import tracing


class ResultProducerResult(BaseModel):
    """The final answer produced by the result-producing agent."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    result: SerializedOptionalResultType = Field(default=None, description="Serialized final produced answer.")


def _completeness(estimator: Estimator | None, stop_reason: StopReason) -> float:
    """Overall completeness of the collected result list.

    Args:
        estimator: the current estimator
        stop_reason: why execution stopped

    Returns:
        (float): completeness in [0, 1]
    """
    if stop_reason == StopReason.ALL_AGENTS or (stop_reason == StopReason.LLM and estimator is None):
        return 1.0
    if estimator is None:
        return 1.0
    observed = estimator.sample_count()
    estimated = estimator.estimated_count()
    if not math.isfinite(estimated) or estimated <= 0:
        return 1.0
    return min(1.0, observed / estimated)


@hooks.hook(HookEvent.EXECUTION_EXECUTE_QUERY_PLAN_EXECUTE_STEP_ANSWER_STEP_MULTIPLE_AGENTS_RESULT_PRODUCER)
async def produce_result(
    aligned_result_list: AlignedResultList,
    estimator: Estimator | None,
    query: str,
    merge_task: str,
    stop_reason: StopReason,
) -> SerializedOptionalResultType:
    """Produce the final answer via the result-producing agent.

    Args:
        aligned_result_list: the aligned result list
        estimator: the current estimator
        query: the step's query
        merge_task: how agent results should be merged
        stop_reason: why execution stopped

    Returns:
        (SerializedOptionalResultType): the final produced answer
    """
    result = ResultProducerResult()
    completeness = _completeness(estimator, stop_reason)
    use_estimator = completeness < 1.0 and estimator is not None

    def final_result(result_value: Any):
        """Set the final produced answer."""
        result.result = result_type.try_parse_value(result_value)

    variables: dict[str, Any] = {
        "query": query,
        "result_list": aligned_result_list.data,
        "pd": pd,
    }
    if use_estimator:
        variables["estimator"] = estimator

    code_environment = CodeExecutionEnvironment(
        variables=variables,
        tools={
            "final_result": final_result,
        },
    )

    def custom_tool_handler(
        _context: RunContextWrapper[None],
        _tool_results: list[FunctionToolResult],
    ) -> ToolsToFinalOutputResult:
        """Expose the final result as output once produced."""
        return ToolsToFinalOutputResult(
            is_final_output=result.result is not None,
            final_output=result,
        )

    result_list_preview = result_type.serialize(aligned_result_list.data, context="preview")

    with tracing.own_span(
        "Coordinator - Result Producer",
        input={
            "query": query,
            "result_list_preview": result_list_preview,
            "completeness": completeness,
            "use_estimator": use_estimator,
            "stop_reason": stop_reason.value,
        },
    ):
        result = (
            await runner.run_agent(
                role=AgentRole.EXECUTION_COORDINATOR_RESULT_PRODUCER,
                prompt_parameters={
                    "query": query,
                    "result_list_preview": result_list_preview,
                    "merge_task": merge_task,
                    "use_estimator": use_estimator,
                },
                tools=[code_environment.as_function_tool()],
                tool_use_behavior=custom_tool_handler,
                reset_tool_choice=False,
                parallel_tool_calls=False,
                tool_choice="required",
                output_type=ResultProducerResult,
            )
        ).output
        tracing.append_to_span_output({"result": result.result})

    if result.result is None:
        raise RuntimeError("Result producer returned no result")
    return result.result

"""Result-producing agent: turns the merged result list into a final answer."""

from typing import Any

import pandas as pd
from agents import (
    FunctionToolResult,
    RunContextWrapper,
    ToolsToFinalOutputResult,
)
from pydantic import BaseModel, ConfigDict

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
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    result: SerializedOptionalResultType = None


@hooks.hook(HookEvent.EXECUTION_EXECUTE_QUERY_PLAN_EXECUTE_STEP_ANSWER_STEP_MULTIPLE_AGENTS_RESULT_PRODUCER)
async def produce_result(
    aligned_result_list: AlignedResultList,
    estimator: Estimator | None,
    query: str,
    merge_task: str,
    stop_reason: StopReason,
) -> SerializedOptionalResultType:
    """Produce the final answer via the result-producing agent."""
    result = ResultProducerResult()

    def final_result(result_value: Any):
        result.result = result_type.try_parse_value(result_value)

    code_environment = CodeExecutionEnvironment(
        variables={
            "query": query,
            "result_list": aligned_result_list.data,
            "estimator": estimator,
            "pd": pd,
        },
        tools={
            "final_result": final_result,
        },
    )

    def custom_tool_handler(
        _context: RunContextWrapper[None],
        _tool_results: list[FunctionToolResult],
    ) -> ToolsToFinalOutputResult:
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
            "has_estimator": estimator is not None,
        },
    ):
        result = (
            await runner.run_agent(
                role=AgentRole.EXECUTION_COORDINATOR_RESULT_PRODUCER,
                prompt_parameters={
                    "query": query,
                    "result_list_preview": result_list_preview,
                    "merge_task": merge_task,
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

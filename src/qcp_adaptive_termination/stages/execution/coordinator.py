from typing import Any

import pandas as pd
from agents import (
    FunctionToolResult,
    RunContextWrapper,
    ToolsToFinalOutputResult,
)

import qcp_adaptive_termination.hooks as hooks
from qcp_adaptive_termination.code_execution.code_execution_environment import (
    CodeExecutionEnvironment,
)
from qcp_adaptive_termination.hooks import HookEvent
from qcp_adaptive_termination.llm import runner
from qcp_adaptive_termination.models import (
    AgentRole,
    need_type,
    result_type,
)
from qcp_adaptive_termination.models.execution import (
    AdditionalInformation,
    PartialQueryAnswer,
)
from qcp_adaptive_termination.models.query_plan import QueryPlanStep
from qcp_adaptive_termination.observability import tracing


async def _answer_step_coordinator_via_llm(
    query_plan_step: QueryPlanStep,
    additional_information: list[AdditionalInformation],
) -> PartialQueryAnswer:
    query = query_plan_step.query

    with tracing.own_span(
        "Coordinator - answer query",
        input={"query": query, "additional_information": additional_information},
    ):
        partial_answer = PartialQueryAnswer()

        def raise_needs(new_needs: Any):
            partial_answer.add_needs(need_type.try_parse_value(new_needs))

        def final_answer(new_answer: Any):
            partial_answer.set_answer(result_type.try_parse_value(new_answer))

        code_environment = CodeExecutionEnvironment(
            variables={
                "query": query,
                "additional_information": additional_information,
                "pd": pd,
            },
            tools={
                "final_answer": final_answer,
                "raise_needs": raise_needs,
            },
        )

        def custom_tool_handler(
            _context: RunContextWrapper[None],
            _tool_results: list[FunctionToolResult],
        ) -> ToolsToFinalOutputResult:
            return ToolsToFinalOutputResult(
                is_final_output=partial_answer.is_final,
                final_output=partial_answer,
            )

        result = await runner.run_agent(
            role=AgentRole.COORDINATOR_CODE,
            prompt_parameters={
                "query": query,
                "additional_information_preview": AdditionalInformation.preview_additional_information(
                    additional_information
                ),
            },
            tools=[code_environment.as_function_tool()],
            tool_use_behavior=custom_tool_handler,
            reset_tool_choice=False,
            parallel_tool_calls=False,
            tool_choice="required",
            output_type=PartialQueryAnswer,
        )

        tracing.append_to_span_output(
            {
                "code": code_environment.to_span_data(),
                "partial_answer": result,
            }
        )
        return result


async def answer_step_coordinator(
    query_plan_step: QueryPlanStep,
    additional_information: list[AdditionalInformation],
) -> PartialQueryAnswer:
    return await hooks.call_override(
        HookEvent.ANSWER_STEP_COORDINATOR_OVERRIDE,
        default=lambda: _answer_step_coordinator_via_llm(query_plan_step, additional_information),
        query_plan_step=query_plan_step,
        additional_information=additional_information,
    )

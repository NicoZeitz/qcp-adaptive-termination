from typing import Any

from agents import (
    FunctionToolResult,
    RunContextWrapper,
    ToolsToFinalOutputResult,
)
import pandas as pd

from qcp_adaptive_termination.code_execution.code_execution_environment import (
    CodeExecutionEnvironment,
)
from qcp_adaptive_termination.model.execution.additional_information import (
    AdditionalInformation,
)
from qcp_adaptive_termination.model.need import try_parse_value_to_need_type
from qcp_adaptive_termination.model.execution.partial_query_answer import (
    PartialQueryAnswer,
)
from qcp_adaptive_termination.model.query_plan.query_plan_step import QueryPlanStep
from qcp_adaptive_termination.model.result import try_parse_value_to_result_type
from qcp_adaptive_termination.agents.configuration import (
    AgentRole,
    get_agent_configuration,
)
from qcp_adaptive_termination.preview import preview_additional_information
from qcp_adaptive_termination.agents.runner import run_agent
from qcp_adaptive_termination.tracing.span import append_to_span_output, own_custom_span


async def answer_step_coordinator(
    query_plan_step: QueryPlanStep,
    additional_information: list[AdditionalInformation],
) -> PartialQueryAnswer:
    query = query_plan_step.query

    with own_custom_span(
        "Coordinator - answer query",
        input={"query": query, "additional_information": additional_information},
    ):
        partial_answer = PartialQueryAnswer()

        def raise_needs(new_needs: Any):
            partial_answer.add_needs(try_parse_value_to_need_type(new_needs))

        def final_answer(new_answer: Any):
            partial_answer.set_answer(try_parse_value_to_result_type(new_answer))

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

        result = await run_agent(
            agent_configuration=get_agent_configuration(AgentRole.COORDINATOR_CODE),
            prompt_name="execution-coordinator-aggregate",
            prompt_parameters={
                "query": query,
                "additional_information_preview": preview_additional_information(
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

        append_to_span_output(
            {
                "code": code_environment.to_span_data(),
                "partial_answer": result,
            }
        )
        return result

from typing import Any, cast

from agents import (
    FunctionToolResult,
    RunContextWrapper,
    ToolsToFinalOutputResult,
)
import pandas as pd

from code_execution.code_execution_environment import CodeExecutionEnvironment
from qtypes.additional_information import AdditionalInformation
from qtypes.need import try_parse_value_to_need_type
from qtypes.partial_query_answer import PartialQueryAnswer, PartialQueryAnswerDummy
from qtypes.query_plan_step import QueryPlanStep
from qtypes.result import try_parse_value_to_result_type
from stages.common.agent_configuration import get_coordinator_agent_configuration
from utils.run_agent import run_agent
from utils.span import append_to_span_output, own_custom_span


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
                # pre-import some common libraries for the agent to use in the code tool
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

        result = cast(
            PartialQueryAnswer,
            await run_agent(
                name_postfix="Code",
                agent_configuration=get_coordinator_agent_configuration(),
                prompt_name="Query Coordinator/Aggregate",
                prompt_parameters={"query": query},
                tools=[code_environment.as_function_tool()],
                tool_use_behavior=custom_tool_handler,
                reset_tool_choice=False,
                parallel_tool_calls=False,
                tool_choice="required",
                output_type=PartialQueryAnswerDummy,  # See: https://github.com/openai/openai-agents-python/blob/main/src/agents/run_internal/turn_resolution.py#L189-L190
                get_actual_result=lambda: partial_answer,
            ),
        )

        append_to_span_output(
            {
                "code": code_environment.to_span_data(),
                "partial_answer": result,
            }
        )
        return result

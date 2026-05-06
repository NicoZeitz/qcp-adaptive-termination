from typing import Any, cast

from agents import FunctionToolResult, RunContextWrapper, RunResult, ToolsToFinalOutputResult
import pandas as pd

from code_execution.code_execution_environment import CodeExecutionEnvironment
from query_coordinator.query_coordinator import QueryCoordinator
from qtypes.partial_query_answer import PartialQueryAnswer, PartialQueryAnswerDummy
from qtypes.query_plan.need import try_parse_value_to_need_type
from qtypes.query_plan.query_plan import QueryPlanStep
from qtypes.query_plan.result import (
    ResultType,
    result_to_span_data,
    try_parse_value_to_result_type,
)
from utils.prompt import load_prompt
from utils.run_agent import run_agent
from utils.span import append_to_span_input, append_to_span_output, own_custom_span


async def answer_step_coordinator(
    coordinator: QueryCoordinator,
    query_plan_step: QueryPlanStep,
    additional_information: dict[str, ResultType],
) -> PartialQueryAnswer:
    query = query_plan_step.query

    with own_custom_span("Coordinator - answer query") as span:
        append_to_span_input(
            span,
            {
                "query": query,
                "additional_information": {
                    k: result_to_span_data(v) if v is not None else None
                    for k, v in additional_information.items()
                },
            },
        )

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

        result, _ = cast(
            tuple[PartialQueryAnswer, RunResult],
            await run_agent(
                name_postfix="Code",
                agent_configuration=coordinator.agent_configuration,
                instructions=load_prompt(
                    "Query Coordinator/Code/Instructions",
                    {},
                ),
                prompt=load_prompt("Query Coordinator/Code/Prompt", {"query": query}),
                tools=[code_environment.as_function_tool()],
                tool_use_behavior=custom_tool_handler,
                reset_tool_choice=False,
                parallel_tool_calls=False,
                tool_choice="required",
                output_type=PartialQueryAnswerDummy, # See: https://github.com/openai/openai-agents-python/blob/main/src/agents/run_internal/turn_resolution.py#L189-L190
            ),
        )

        append_to_span_output(
            span,
            {
                "code": code_environment.to_span_data(),
                "partial_answer": result.to_span_data(),
            },
        )
        return result

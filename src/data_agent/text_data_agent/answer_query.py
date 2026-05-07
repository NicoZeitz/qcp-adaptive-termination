from typing import Any, cast

from agents import (
    FunctionToolResult,
    RunContextWrapper,
    RunResult,
    ToolsToFinalOutputResult,
)
import pandas as pd

from data_agent.data_agent import DataAgent
from qtypes.AdditionalInformation import AdditionalInformation
from qtypes.partial_query_answer import PartialQueryAnswer, PartialQueryAnswerDummy
from code_execution.code_execution_environment import CodeExecutionEnvironment
from qtypes.query_plan.need import (
    try_parse_value_to_need_type,
)
from qtypes.query_plan.result import (
    try_parse_value_to_result_type,
)
from utils.prompt import load_prompt
from utils.run_agent import run_agent
from utils.span import append_to_span_input, append_to_span_output, own_custom_span


async def answer_query(
    data_agent: DataAgent,
    text: str,
    query: str,
    additional_information: list[AdditionalInformation],
) -> PartialQueryAnswer:
    with own_custom_span("Text Data Agent - answer_query") as span:
        append_to_span_input(
            span,
            {
                "query": query,
                "additional_information": [
                    information.to_span_data() for information in additional_information
                ],
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
                "text": text,
                "summary": await data_agent.data_description(),
                "additional_information": additional_information,
                # pre-import some common libraries for the agent to use in the code tool
                "pd": pd,
            },
            tools={
                "raise_needs": raise_needs,
                "final_answer": final_answer,
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
                agent_configuration=data_agent.agent_configuration,
                instructions=load_prompt(
                    "Text Data Agent/Code/Instructions",
                    {"summary": await data_agent.data_description()},
                ),
                prompt=load_prompt("Text Data Agent/Code/Prompt", {"query": query}),
                tools=[code_environment.as_function_tool()],
                tool_use_behavior=custom_tool_handler,
                reset_tool_choice=False,
                parallel_tool_calls=False,
                tool_choice="required",
                output_type=PartialQueryAnswerDummy,  # See: https://github.com/openai/openai-agents-python/blob/main/src/agents/run_internal/turn_resolution.py#L189-L190
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

from code_execution.code_execution_environment import CodeExecutionEnvironment
from qtypes.aligned_result_list import (
    AggregationType,
    AlignedResultList,
    AlignedSchemaMetadata,
)
from qtypes.result import ResultType
from stages.common.agent_configuration import get_coordinator_agent_configuration
from stages.execution.answer_step_multiple_agents.validation import (
    SOURCE_AGENT_COLUMN,
    validate_aligned_schema_metadata,
    validate_and_parse_result_type,
)
from utils.preview import preview_dataframe
from utils.run_agent import run_agent
from utils.span import append_to_span_output, own_custom_span


import pandas as pd
from agents import FunctionToolResult, RunContextWrapper, ToolsToFinalOutputResult


from typing import Any


async def align_schema_first(
    query: str,
    agent_name: str,
    agent_result: ResultType,
) -> AlignedResultList:
    result_list = AlignedResultList()

    def set_schema(
        result_df: Any,
        aggregation_type: AggregationType,
        keys: list[str],
        value_column: str | None = None,
    ):
        nonlocal result_list

        aligned_schema_metadata = AlignedSchemaMetadata(
            aggregation_type=aggregation_type,
            keys=list(keys),
            value_column=value_column,
        )
        dataframe = validate_and_parse_result_type(result_df)
        validate_aligned_schema_metadata(dataframe, aligned_schema_metadata)

        result_list.schema = aligned_schema_metadata
        dataframe[SOURCE_AGENT_COLUMN] = agent_name
        result_list.data = dataframe

    code_environment = CodeExecutionEnvironment(
        variables={"query": query, "agent_result": agent_result, "pd": pd},
        tools={"set_schema": set_schema},
    )

    def custom_tool_handler(
        _context: RunContextWrapper[None],
        _tool_results: list[FunctionToolResult],
    ) -> ToolsToFinalOutputResult:
        nonlocal result_list
        return ToolsToFinalOutputResult(
            is_final_output=result_list.schema is not None, final_output=result_list
        )

    agent_result_preview = preview_dataframe(agent_result)

    with own_custom_span(
        "Query Coordinator - Align Schema First",
        input={
            "query": query,
            "agent_name": agent_name,
            "agent_result_preview": agent_result_preview,
            "is_first": True,
        },
    ):
        result_list = await run_agent(
            agent_configuration=get_coordinator_agent_configuration(
                "Align Schema First"
            ),
            prompt_name="execution-coordinator-align-schema-first",
            prompt_parameters={
                "query": query,
                "agent_name": agent_name,
                "force_result": False,
                "agent_result_preview": agent_result_preview,
            },
            tools=[code_environment.as_function_tool()],
            tool_use_behavior=custom_tool_handler,
            reset_tool_choice=False,
            parallel_tool_calls=False,
            tool_choice="required",
            output_type=AlignedResultList,
        )

        append_to_span_output({"result_list": result_list})

        return result_list

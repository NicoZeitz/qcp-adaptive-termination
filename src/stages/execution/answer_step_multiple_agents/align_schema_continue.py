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
    validate_source_agent_column,
)
from utils.preview import preview_dataframe
from utils.run_agent import run_agent
from utils.span import append_to_span_output, own_custom_span


import pandas as pd
from agents import FunctionToolResult, RunContextWrapper, ToolsToFinalOutputResult


from typing import Any, cast


async def align_schema_continue(
    query: str,
    agent_name: str,
    agent_result: ResultType,
    aligned: AlignedResultList,
) -> AlignedResultList:
    is_done = False
    agent_result[SOURCE_AGENT_COLUMN] = agent_name

    def merge_result(
        merged_df: Any,
        aggregation_type: AggregationType | None = None,
        keys: list[str] | None = None,
        value_column: str | None = None,
    ):
        nonlocal is_done, aligned
        dataframe = validate_and_parse_result_type(merged_df)

        new_aligned_schema_metadata = AlignedSchemaMetadata(
            aggregation_type=aggregation_type
            if aggregation_type is not None
            else cast(AlignedSchemaMetadata, aligned.schema).aggregation_type,
            keys=list(keys)
            if keys is not None
            else cast(AlignedSchemaMetadata, aligned.schema).keys,
            value_column=value_column
            if value_column is not None
            else cast(AlignedSchemaMetadata, aligned.schema).value_column,
        )

        validate_aligned_schema_metadata(dataframe, new_aligned_schema_metadata)
        validate_source_agent_column(aligned.data, dataframe, agent_name)

        aligned.data = dataframe
        aligned.schema = new_aligned_schema_metadata
        is_done = True

    code_environment = CodeExecutionEnvironment(
        variables={
            "query": query,
            "existing_result_list": aligned,
            "agent_result": agent_result,
            "pd": pd,
        },
        tools={"merge_result": merge_result},
    )

    def custom_tool_handler(
        _context: RunContextWrapper[None],
        _tool_results: list[FunctionToolResult],
    ) -> ToolsToFinalOutputResult:
        nonlocal is_done, aligned
        return ToolsToFinalOutputResult(is_final_output=is_done, final_output=aligned)

    agent_result_preview = preview_dataframe(agent_result)
    existing_result_preview = preview_dataframe(aligned.data)

    with own_custom_span(
        "Query Coordinator - Align Schema Continue",
        input={
            "query": query,
            "agent_name": agent_name,
            "is_first": False,
            "agent_result_preview": agent_result_preview,
            "existing_schema": aligned.schema if aligned.schema else None,
            "existing_result_preview": existing_result_preview,
        },
    ):
        new_aligned = await run_agent(
            agent_configuration=get_coordinator_agent_configuration(
                "Align Schema Continue"
            ),
            prompt_name="execution-coordinator-align-schema-continue",
            prompt_parameters={
                "query": query,
                "agent_name": agent_name,
                "agent_result_preview": agent_result_preview,
                "existing_schema": cast(
                    AlignedSchemaMetadata, aligned.schema
                ).model_dump_json(indent=2),
                "existing_result_preview": existing_result_preview,
            },
            tools=[code_environment.as_function_tool()],
            tool_use_behavior=custom_tool_handler,
            reset_tool_choice=False,
            parallel_tool_calls=False,
            tool_choice="required",
            output_type=AlignedResultList,
        )

        append_to_span_output({"result_list": new_aligned})
        return new_aligned

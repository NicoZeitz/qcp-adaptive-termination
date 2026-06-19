from code_execution.code_execution_environment import CodeExecutionEnvironment
from qtypes.aligned_result_list import (
    AggregationType,
    AlignedResultList,
    AlignedSchemaMetadata,
)
from qtypes.partial_query_answer import PartialQueryAnswer, PartialQueryAnswerDummy
from qtypes.result import ResultType, try_parse_value_to_result_type
from stages.common.agent_configuration import get_coordinator_agent_configuration
from utils.preview import preview_dataframe
from utils.run_agent import run_agent
from utils.span import append_to_span_output, indented_logging, own_custom_span


import pandas as pd
from agents import FunctionToolResult, RunContextWrapper, ToolsToFinalOutputResult


from typing import Any, cast


async def align_schema_first(
    query: str,
    agent_name: str,
    raw_data: pd.DataFrame,
) -> tuple[PartialQueryAnswer | None, AlignedSchemaMetadata | None]:
    partial_answer: PartialQueryAnswer | None = None
    captured_schema: AlignedSchemaMetadata | None = None

    def set_schema(
        result_df: Any,
        aggregation_type: AggregationType,
        keys: list,
        value_column: str | None = None,
    ):
        """Commit the schema for the unified result list and provide the transformed result.

        Transform agent_result into result_df before calling — rename, drop, or add
        columns as needed so the DataFrame matches the schema you are declaring.
        A `source_agent` column is added automatically; do not include it yourself.
        """
        nonlocal partial_answer, captured_schema
        captured_schema = AlignedSchemaMetadata(
            aggregation_type=aggregation_type,
            keys=list(keys),
            value_column=value_column,
        )
        df = try_parse_value_to_result_type(result_df).copy()
        df["source_agent"] = agent_name
        partial_answer = PartialQueryAnswer()
        partial_answer.set_answer(df)

    code_environment = CodeExecutionEnvironment(
        variables={"query": query, "agent_result": raw_data, "pd": pd},
        tools={"set_schema": set_schema},
    )

    def custom_tool_handler(
        _context: RunContextWrapper[None],
        _tool_results: list[FunctionToolResult],
    ) -> ToolsToFinalOutputResult:
        is_done = partial_answer is not None and partial_answer.is_final
        return ToolsToFinalOutputResult(
            is_final_output=is_done, final_output=partial_answer
        )

    with own_custom_span(
        "Query Coordinator - Align Schema First",
        input={
            "query": query,
            "agent_name": agent_name,
            "is_first": True,
            "agent_result_rows": len(raw_data),
            "agent_result_columns": list(raw_data.columns),
            "agent_result_preview": preview_dataframe(
                raw_data, top_rows=5, bottom_rows=0
            ),
        },
    ):
        await run_agent(
            agent_configuration=get_coordinator_agent_configuration(
                "Align Schema First"
            ),
            prompt_name="execution-coordinator-align-schema-first",
            prompt_parameters={
                "query": query,
                "agent_name": agent_name,
                "agent_result_columns": str(list(raw_data.columns)),
                "agent_result_preview": preview_dataframe(
                    raw_data, top_rows=5, bottom_rows=0
                ),
            },
            tools=[code_environment.as_function_tool()],
            tool_use_behavior=custom_tool_handler,
            reset_tool_choice=False,
            parallel_tool_calls=False,
            tool_choice="required",
            output_type=PartialQueryAnswerDummy,
            get_actual_result=lambda: partial_answer,
        )
        has_answer = partial_answer is not None and partial_answer.has_answer
        result_rows = (
            len(partial_answer.answer)
            if has_answer
            and partial_answer is not None
            and partial_answer.answer is not None
            else 0
        )
        append_to_span_output(
            {
                "has_answer": has_answer,
                "result_rows": result_rows,
                "captured_schema": captured_schema.model_dump()
                if captured_schema is not None
                else None,
                "fallback_used": not has_answer,
            }
        )
        indented_logging(
            f"Align Schema First: has_answer={has_answer}, rows={result_rows}, "
            f"schema={captured_schema.model_dump() if captured_schema else 'inherited'}"
        )

    return partial_answer, captured_schema


async def align_schema_continue(
    query: str,
    agent_name: str,
    raw_data: pd.DataFrame,
    aligned: AlignedResultList,
) -> tuple[PartialQueryAnswer | None, pd.DataFrame]:
    partial_answer: PartialQueryAnswer | None = None

    new_data = raw_data.copy()
    new_data["source_agent"] = agent_name

    def merge_result(merged_df: Any):
        """Provide the merged DataFrame combining existing_result_list and the transformed new agent result.

        Transform agent_result to align its columns to the existing schema before merging.
        Rename, drop, or add columns as needed. The `source_agent` column must be preserved
        in both parts — do not drop it.
        """
        nonlocal partial_answer
        if partial_answer is None:
            partial_answer = PartialQueryAnswer()
        partial_answer.set_answer(try_parse_value_to_result_type(merged_df))

    code_environment = CodeExecutionEnvironment(
        variables={
            "query": query,
            "existing_result_list": aligned.data,
            "agent_result": new_data,
            "pd": pd,
        },
        tools={"merge_result": merge_result},
    )

    def custom_tool_handler(
        _context: RunContextWrapper[None],
        _tool_results: list[FunctionToolResult],
    ) -> ToolsToFinalOutputResult:
        is_done = partial_answer is not None and partial_answer.is_final
        return ToolsToFinalOutputResult(
            is_final_output=is_done, final_output=partial_answer
        )

    with own_custom_span(
        "Query Coordinator - Align Schema Continue",
        input={
            "query": query,
            "agent_name": agent_name,
            "is_first": False,
            "agent_result_rows": len(new_data),
            "agent_result_columns": list(new_data.columns),
            "agent_result_preview": preview_dataframe(
                new_data, top_rows=5, bottom_rows=0
            ),
            "existing_result_rows": len(aligned.data),
            "existing_result_columns": list(aligned.data.columns),
            "existing_schema": aligned.schema.model_dump() if aligned.schema else None,
            "existing_result_preview": preview_dataframe(
                aligned.data, top_rows=10, bottom_rows=0
            ),
        },
    ):
        await run_agent(
            agent_configuration=get_coordinator_agent_configuration(
                "Align Schema Continue"
            ),
            prompt_name="execution-coordinator-align-schema-continue",
            prompt_parameters={
                "query": query,
                "agent_name": agent_name,
                "agent_result_columns": str(list(new_data.columns)),
                "agent_result_preview": preview_dataframe(
                    new_data, top_rows=5, bottom_rows=0
                ),
                "existing_result_columns": str(list(aligned.data.columns)),
                "existing_result_preview": preview_dataframe(
                    aligned.data, top_rows=10, bottom_rows=0
                ),
                "schema": aligned.schema.model_dump_json() if aligned.schema else None,
            },
            tools=[code_environment.as_function_tool()],
            tool_use_behavior=custom_tool_handler,
            reset_tool_choice=False,
            parallel_tool_calls=False,
            tool_choice="required",
            output_type=PartialQueryAnswerDummy,
            get_actual_result=lambda: partial_answer,
        )
        has_answer = partial_answer is not None and partial_answer.has_answer
        result_rows = (
            len(partial_answer.answer)
            if has_answer
            and partial_answer is not None
            and partial_answer.answer is not None
            else 0
        )
        append_to_span_output(
            {
                "has_answer": has_answer,
                "result_rows": result_rows,
                "captured_schema": None,
                "fallback_used": not has_answer,
            }
        )
        indented_logging(
            f"Align Schema Continue: has_answer={has_answer}, rows={result_rows}, schema=inherited"
        )

    return partial_answer, new_data


async def align_schema(
    aligned: AlignedResultList,
    agent_name: str,
    agent_result: ResultType,
    query: str,
) -> AlignedResultList:
    is_first = aligned.schema is None
    raw_data = agent_result.copy()

    if is_first:
        partial_answer, captured_schema = await align_schema_first(
            query, agent_name, raw_data
        )
        if partial_answer is None or not partial_answer.has_answer:
            fallback = raw_data.copy()
            fallback["source_agent"] = agent_name
            return AlignedResultList(data=fallback, schema=None)
        return AlignedResultList(
            data=cast(pd.DataFrame, partial_answer.answer),
            schema=captured_schema,
        )
    else:
        partial_answer, new_data = await align_schema_continue(
            query, agent_name, raw_data, aligned
        )
        if partial_answer is None or not partial_answer.has_answer:
            merged = (
                pd.concat([aligned.data, new_data], ignore_index=True)
                if not aligned.data.empty
                else new_data
            )
            return AlignedResultList(data=merged, schema=aligned.schema)
        return AlignedResultList(
            data=cast(pd.DataFrame, partial_answer.answer),
            schema=aligned.schema,
        )

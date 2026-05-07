from dataclasses import dataclass
from typing import Any, cast

from agents import (
    FunctionToolResult,
    RunContextWrapper,
    RunResult,
    ToolsToFinalOutputResult,
)
import pandas as pd
from pandas import DataFrame
from pydantic import JsonValue
from sqlalchemy import Engine

from data_agent.data_agent import DataAgent
from qtypes.AdditionalInformation import AdditionalInformation
from qtypes.partial_query_answer import PartialQueryAnswer, PartialQueryAnswerDummy
from code_execution.code_execution_environment import CodeExecutionEnvironment
from qtypes.query_plan.need import try_parse_value_to_need_type
from qtypes.query_plan.result import (
    try_parse_value_to_result_type,
)
from utils.serialize_value import serialize_value
from utils.prompt import load_prompt
from utils.preview import preview_dataframe
from utils.run_agent import run_agent
from utils.span import append_to_span_input, append_to_span_output, own_custom_span


@dataclass
class SqlQuery:
    query: str
    error: Exception | None = None
    result: DataFrame | None = None

    def to_span_data(self) -> JsonValue:
        span_data: dict[str, JsonValue] = {"query": self.query}

        if self.result is not None:
            span_data["result_preview"] = preview_dataframe(self.result)
        if self.error is not None:
            span_data["error"] = serialize_value(self.error)

        return span_data


async def answer_query(
    data_agent: DataAgent,
    db_engine: Engine,
    query: str,
    additional_information: list[AdditionalInformation],
) -> PartialQueryAnswer:
    with own_custom_span("SQL Data Agent - answer_query") as span:
        append_to_span_input(
            span,
            {
                "query": query,
                "additional_information": [
                    information.to_span_data() for information in additional_information
                ],
            },
        )

        sql_queries: list[SqlQuery] = []
        partial_answer = PartialQueryAnswer()

        def raise_needs(new_needs: Any):
            partial_answer.add_needs(try_parse_value_to_need_type(new_needs))

        def final_answer(new_answer: Any):
            partial_answer.set_answer(try_parse_value_to_result_type(new_answer))

        def query_sql(query: str) -> DataFrame:
            sql_query = SqlQuery(query=query)

            try:
                result = pd.read_sql_query(
                    query, db_engine, dtype_backend="numpy_nullable"
                )
                sql_query.result = result
                return result
            except Exception as e:
                sql_query.error = e
                raise
            finally:
                sql_queries.append(sql_query)

        code_environment = CodeExecutionEnvironment(
            variables={
                "query": query,
                "additional_information": additional_information,
                # pre-import some common libraries for the agent to use in the code tool
                "pd": pd,
            },
            tools={
                "query_sql": query_sql,
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
                prompt_name="SQL Data Agent/Code",
                prompt_parameters={
                    "m_schema": await data_agent.data_description(),
                    "query": query
                },
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
                "sql_queries": [query.to_span_data() for query in sql_queries],
                "code": code_environment.to_span_data(),
                "partial_answer": result.to_span_data(),
            },
        )

        return result

from dataclasses import dataclass
from typing import Any

import pandas as pd
from agents import FunctionToolResult, RunContextWrapper, ToolsToFinalOutputResult
from pydantic import JsonValue

import qcp_adaptive_termination.agent_registry as agent_registry
from qcp_adaptive_termination.code_execution.code_execution_environment import (
    CodeExecutionEnvironment,
)
from qcp_adaptive_termination.db import get_db_engine
from qcp_adaptive_termination.llm import runner
from qcp_adaptive_termination.models import (
    AgentRegistryType,
    AgentRole,
    AlignedSchemaMetadata,
    DataAgent,
    try_parse_value_to_need_type,
    try_parse_value_to_result_type,
)
from qcp_adaptive_termination.models.execution import (
    AdditionalInformation,
    PartialQueryAnswer,
)
from qcp_adaptive_termination.observability import tracing
from qcp_adaptive_termination.preview import (
    preview_additional_information,
    result_to_json_preview,
)
from qcp_adaptive_termination.serialize_value import serialize_value
from qcp_adaptive_termination.stages.execution.validation import validate_agent_output


@dataclass(slots=True)
class SqlQuery:
    query: str
    error: Exception | None = None
    result: pd.DataFrame | None = None

    def to_span_data(self) -> JsonValue:
        span_data: dict[str, JsonValue] = {"query": self.query}

        if self.result is not None:
            span_data["result_preview"] = result_to_json_preview(self.result)
        if self.error is not None:
            span_data["error"] = serialize_value(self.error)

        return span_data


async def answer_query_data_agent(
    data_agent: DataAgent,
    query: str,
    additional_information: list[AdditionalInformation],
    schema_metadata: AlignedSchemaMetadata | None = None,
) -> PartialQueryAnswer:
    with tracing.own_span(
        f"Data Agent {data_agent.name} - answer_query",
        input={
            "query": query,
            "additional_information": additional_information,
            "schema_metadata": schema_metadata,
        },
    ):
        db_engine = get_db_engine(data_agent)
        m_schema = await agent_registry.build_registry_string(
            data_agent, AgentRegistryType.TECHNICAL, include_capabilities=False
        )
        sql_queries: list[SqlQuery] = []
        partial_answer = PartialQueryAnswer()

        def query_sql(query: str) -> pd.DataFrame:
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

        def raise_needs(new_needs: Any):
            partial_answer.add_needs(try_parse_value_to_need_type(new_needs))

        def final_answer(new_answer: Any):
            answer_df = try_parse_value_to_result_type(new_answer)
            if schema_metadata is not None:
                validate_agent_output(answer_df, schema_metadata)
            partial_answer.set_answer(answer_df)

        code_environment = CodeExecutionEnvironment(
            variables={
                "query": query,
                "additional_information": additional_information,
                "pd": pd,
            },
            tools={
                "raise_needs": raise_needs,
                "final_answer": final_answer,
                "query_sql": query_sql,
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
            role=AgentRole.DATA_AGENT_CODE,
            agent_name=data_agent.name,
            prompt_parameters={
                "m_schema": m_schema,
                "query": query,
                "additional_information_preview": preview_additional_information(
                    additional_information
                ),
                "columns": ", ".join(schema_metadata.columns)
                if schema_metadata is not None
                else None,
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
                "sql_queries": [sql_query.to_span_data() for sql_query in sql_queries],
            }
        )
        return result


async def answer_step_data_agent(
    data_agents: list[DataAgent],
    query: str,
    to_agent: str,
    additional_information: list[AdditionalInformation],
    schema_metadata: AlignedSchemaMetadata | None = None,
) -> PartialQueryAnswer:
    responsible_data_agent = next(
        (agent for agent in data_agents if agent.name == to_agent)
    )
    return await answer_query_data_agent(
        responsible_data_agent, query, additional_information, schema_metadata
    )

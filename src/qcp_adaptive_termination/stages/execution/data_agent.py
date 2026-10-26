"""Data agent answer step: execute SQL against a data agent and return a partial answer."""

from dataclasses import dataclass
from typing import Any

import pandas as pd
from agents import FunctionToolResult, RunContextWrapper, ToolsToFinalOutputResult
from pydantic import JsonValue

import qcp_adaptive_termination.agent_registry as agent_registry
import qcp_adaptive_termination.database as database
import qcp_adaptive_termination.hooks as hooks
from qcp_adaptive_termination.code_execution.code_execution_environment import (
    CodeExecutionEnvironment,
)
from qcp_adaptive_termination.hooks import HookEvent
from qcp_adaptive_termination.llm import runner
from qcp_adaptive_termination.models import (
    AgentRegistryType,
    AgentRole,
    AlignedSchemaMetadata,
    DataAgent,
    need_type,
    result_type,
)
from qcp_adaptive_termination.models.execution import (
    AdditionalInformation,
    PartialQueryAnswer,
)
from qcp_adaptive_termination.observability import tracing
from qcp_adaptive_termination.stages.execution.validation import validate_agent_output


@dataclass(slots=True)
class SqlQuery:
    """A single SQL query executed against a data agent."""

    query: str
    """SQL query text executed against the agent"""
    error: Exception | None = None
    """Error raised during execution, if any"""
    result: pd.DataFrame | None = None
    """Dataframe returned by the query, if any"""

    def to_span_data(self) -> JsonValue:
        """Serialize the query for tracing output."""
        span_data: dict[str, JsonValue] = {"query": self.query}

        if self.result is not None:
            span_data["result_preview"] = result_type.serialize(self.result, context="preview")
        if self.error is not None:
            span_data["error"] = tracing.serialize_data_for_tracing(self.error)

        return span_data


@hooks.hook(HookEvent.EXECUTION_QUERY_DATA_AGENT)
async def answer_query_data_agent(
    data_agent: DataAgent,
    query: str,
    additional_information: list[AdditionalInformation],
    schema_metadata: AlignedSchemaMetadata | None = None,
) -> PartialQueryAnswer:
    """Answer a query against one data agent.

    Args:
        data_agent: the agent to query
        query: the query to answer
        additional_information: the satisfying dependency answers
        schema_metadata: the schema the output must conform to

    Returns:
        (PartialQueryAnswer): the data agent's partial answer
    """
    with tracing.own_span(
        f"Data Agent {data_agent.name} - answer_query",
        input={
            "query": query,
            "additional_information": additional_information,
            "schema_metadata": schema_metadata,
        },
    ):
        db_engine = database.get_database_engine(data_agent)
        m_schema = await agent_registry.build_registry_string(
            data_agent, AgentRegistryType.TECHNICAL, include_capabilities=False
        )
        sql_queries: list[SqlQuery] = []
        partial_answer = PartialQueryAnswer()

        def query_sql(query: str) -> pd.DataFrame:
            """Run one SQL query against the data agent."""
            sql_query = SqlQuery(query=query)

            try:
                result = pd.read_sql_query(query, db_engine, dtype_backend="numpy_nullable")
                sql_query.result = result
                return result
            except Exception as e:
                sql_query.error = e
                raise
            finally:
                sql_queries.append(sql_query)

        def raise_needs(new_needs: Any):
            """Register needs raised by the data agent."""
            partial_answer.add_needs(need_type.try_parse_value(new_needs))

        def final_answer(new_answer: Any):
            """Validate and set the data agent's final answer."""
            answer_df = result_type.try_parse_value(new_answer)
            if schema_metadata is not None:
                answer_df = validate_agent_output(answer_df, schema_metadata)
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
            """Expose the partial answer as final output when complete."""
            return ToolsToFinalOutputResult(
                is_final_output=partial_answer.is_final,
                final_output=partial_answer,
            )

        result = (
            await runner.run_agent(
                role=AgentRole.EXECUTION_DATA_AGENT_QUERY,
                agent_name=data_agent.name,
                prompt_parameters={
                    "m_schema": m_schema,
                    "query": query,
                    "additional_information_preview": AdditionalInformation.preview_additional_information(
                        additional_information
                    ),
                    "columns": ", ".join(schema_metadata.columns) if schema_metadata is not None else None,
                },
                tools=[code_environment.as_function_tool()],
                tool_use_behavior=custom_tool_handler,
                reset_tool_choice=False,
                parallel_tool_calls=False,
                tool_choice="required",
                output_type=PartialQueryAnswer,
            )
        ).output

        tracing.append_to_span_output(
            {
                "code": code_environment.to_span_data(),
                "partial_answer": result,
                "sql_queries": [sql_query.to_span_data() for sql_query in sql_queries],
            }
        )
        return result


@hooks.hook(HookEvent.EXECUTION_EXECUTE_QUERY_PLAN_EXECUTE_STEP_ANSWER_STEP_DATA_AGENT)
async def answer_step_data_agent(
    data_agents: list[DataAgent],
    query: str,
    to_agent: str,
    additional_information: list[AdditionalInformation],
    schema_metadata: AlignedSchemaMetadata | None = None,
) -> PartialQueryAnswer:
    """Answer one query plan step as a named data agent.

    Args:
        data_agents: the available data agents
        query: the query to answer
        to_agent: the name of the agent answering
        additional_information: the satisfying dependency answers
        schema_metadata: the schema the output must conform to

    Returns:
        (PartialQueryAnswer): the data agent's partial answer
    """
    responsible_data_agent = next(agent for agent in data_agents if agent.name == to_agent)
    return await answer_query_data_agent(responsible_data_agent, query, additional_information, schema_metadata)

from typing import Any, Callable, Optional, cast

from agents import FunctionToolResult, RunContextWrapper, ToolsToFinalOutputResult
from attr import dataclass
import pandas as pd
from pydantic import JsonValue

from code_execution.code_execution_environment import CodeExecutionEnvironment
from qtypes.additional_information import AdditionalInformation
from qtypes.data_agent import DataAgent
from qtypes.need import try_parse_value_to_need_type
from qtypes.partial_query_answer import PartialQueryAnswer, PartialQueryAnswerDummy
from qtypes.result import try_parse_value_to_result_type
from stages.common.agent_configuration import get_data_agent_agent_configuration
from stages.common.data_agent.data_description import data_agent_data_description
from stages.common.data_agent.sql_data_agent.get_db_connection import get_db_engine
from stages.common.data_agent.text_data_agent.get_text import get_text
from utils.serialize_value import serialize_value
from utils.preview import preview_dataframe
from utils.run_agent import run_agent
from utils.span import append_to_span_output, own_custom_span


@dataclass
class Parameters:
    prompt_name: str
    prompt_parameters: dict[str, Any] = {}
    additional_variables: dict[str, str] = {}
    additional_tools: dict[str, Any] = {}
    additional_span_data: Optional[Callable[[], dict[str, Any]]] = None


@dataclass
class SqlQuery:
    query: str
    error: Exception | None = None
    result: pd.DataFrame | None = None

    def to_span_data(self) -> JsonValue:
        span_data: dict[str, JsonValue] = {"query": self.query}

        if self.result is not None:
            span_data["result_preview"] = preview_dataframe(self.result)
        if self.error is not None:
            span_data["error"] = serialize_value(self.error)

        return span_data


async def get_text_data_agent_parameters(
    data_agent: DataAgent, query: str
) -> Parameters:
    summary = await data_agent_data_description(data_agent)
    return Parameters(
        prompt_name="Text Data Agent/Code",
        prompt_parameters={"summary": summary, "query": query},
        additional_variables={
            "text": get_text(data_agent),
            "summary": summary,
        },
    )


async def get_sql_data_agent_parameters(
    data_agent: DataAgent,
    query: str,
    additional_information: list[AdditionalInformation],
) -> Parameters:
    db_engine = get_db_engine(data_agent)

    sql_queries: list[SqlQuery] = []

    def query_sql(query: str) -> pd.DataFrame:
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

    def additional_span_data_func() -> dict[str, Any]:
        return {"sql_queries": [sql_query.to_span_data() for sql_query in sql_queries]}

    m_schema = await data_agent_data_description(data_agent)

    return Parameters(
        prompt_name="SQL Data Agent/Code",
        prompt_parameters={"m_schema": m_schema, "query": query},
        additional_tools={"query_sql": query_sql},
        additional_span_data=additional_span_data_func,
    )


async def answer_step_data_agent(
    data_agents: list[DataAgent],
    query: str,
    to_agent: str,
    additional_information: list[AdditionalInformation],
) -> PartialQueryAnswer:
    responsible_data_agent = next(
        (agent for agent in data_agents if agent.name == to_agent)
    )

    if responsible_data_agent.type == "text":
        parameters = await get_text_data_agent_parameters(responsible_data_agent, query)
    elif responsible_data_agent.type == "sql":
        parameters = await get_sql_data_agent_parameters(
            responsible_data_agent, query, additional_information
        )
    else:
        raise ValueError(f"Unsupported data agent type: {responsible_data_agent.type}")

    return await answer_query(
        data_agent=responsible_data_agent,
        query=query,
        additional_information=additional_information,
        prompt_name=parameters.prompt_name,
        prompt_parameters=parameters.prompt_parameters,
        additional_variables=parameters.additional_variables,
        additional_tools=parameters.additional_tools,
        additional_span_data=parameters.additional_span_data,
    )


async def answer_query(
    data_agent: DataAgent,
    query: str,
    additional_information: list[AdditionalInformation],
    prompt_name: str,
    prompt_parameters: dict[str, str],
    additional_variables: dict[str, Any],
    additional_tools: dict[str, Any],
    additional_span_data: Callable[[], dict[str, Any]] | None = None,
) -> PartialQueryAnswer:

    with own_custom_span(
        "Data Agent - answer_query",
        input={"query": query, "additional_information": additional_information},
    ):
        # sql_queries: list[SqlQuery] = []
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
                **additional_variables,
            },
            tools={
                "raise_needs": raise_needs,
                "final_answer": final_answer,
                **additional_tools,
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
                agent_configuration=get_data_agent_agent_configuration(data_agent),
                prompt_name=prompt_name,
                prompt_parameters=prompt_parameters,
                tools=[code_environment.as_function_tool()],
                tool_use_behavior=custom_tool_handler,
                reset_tool_choice=False,
                parallel_tool_calls=False,
                tool_choice="required",
                output_type=PartialQueryAnswerDummy,  # See: https://github.com/openai/openai-agents-python/blob/main/src/agents/run_internal/turn_resolution.py#L189-L190,
                get_actual_result=lambda: partial_answer,
            ),
        )

        append_to_span_output(
            {
                "code": code_environment.to_span_data(),
                "partial_answer": result,
                **(additional_span_data() if additional_span_data is not None else {}),
            }
        )
        return result

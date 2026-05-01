import dataclasses
from typing import Any, Sequence, TypedDict, cast
from agents import (
    Agent,
    AgentOutputSchema,
    FunctionToolResult,
    RunContextWrapper,
    Runner,
    ToolsToFinalOutputResult,
)
from sqlalchemy import Engine, Row, text
from modules.code_execution_environment import (
    CodeExecutionEnvironment,
    CodeExecutionResult,
)
from modules.model import (
    CHAR_LIMIT_SPAN,
    Need,
    PartialQueryAnswer,
    AgentContext,
    DataAgent,
    Symbol,
    own_custom_span,
    preview_text,
    serialize_value,
)
from modules.mschema.schema_engine import SchemaEngine
from pydantic import Field, create_model
from modules.prompt import load_prompt


class CodeExecution(TypedDict):
    code: str
    result: CodeExecutionResult
    formatted: str


class SqlDataAgent(DataAgent):
    def __init__(self, agent: Agent[AgentContext], db_engine: Engine):
        super().__init__(agent)
        self.db_engine = db_engine

        self.schema_engine = SchemaEngine(
            engine=self.db_engine, db_name=self.agent.name
        )
        self.schema_initialized = False
        self.natural_language_description: str = ""
        self.m_schema: str = ""

    async def get_data_and_query_capabilities(self) -> str:
        await self._initialize_schema()
        return self.natural_language_description

    async def get_schema_for_query(self, query: str) -> str | None:
        await self._initialize_schema()

        instructions = load_prompt(
            "SQL Data Agent/Relevance/Instructions",
            {"m_schema": self.m_schema},
        )
        prompt = load_prompt("SQL Data Agent/Relevance/Prompt", {"query": query})

        relevance_agent = self.agent.clone(
            name=f"{self.agent.name}_Relevance_Agent",
            instructions=instructions,
            output_type=bool,
        )

        is_relevant = cast(
            bool, (await Runner.run(relevance_agent, prompt)).final_output
        )

        if not is_relevant:
            return None

        return self.m_schema

    async def answer_query(
        self, query: str, answered_needs: list[Need]
    ) -> PartialQueryAnswer:
        await self._initialize_schema()

        with own_custom_span("SQL Data Agent - answer_query") as span:
            needs: list[Any] = []
            answer: Any | Symbol = Symbol()
            sql_queries: list[str] = []

            additional_information = [
                {"information": answered_need.need, "data": answered_need.answer}
                for answered_need in answered_needs
            ]

            def raise_needs(new_needs: list[Any]):
                needs.extend(new_needs)

            def final_answer(new_answer: Any):
                nonlocal answer
                answer = new_answer

            def query_sql(query: str) -> Sequence[Row[Any]]:
                with self.db_engine.connect() as conn:
                    sql_queries.append(query)
                    result = conn.execute(text(query)).fetchall()
                    return result

            code_environment = CodeExecutionEnvironment(
                variables={
                    "query": query,
                    "additional_information": additional_information,
                },
                tools={
                    "query_sql": query_sql,
                    "raise_needs": raise_needs,
                    "final_answer": final_answer,
                },
            )

            def custom_tool_handler(
                _context: RunContextWrapper[AgentContext],
                _tool_results: list[FunctionToolResult],
            ) -> ToolsToFinalOutputResult:
                nonlocal answer, needs
                if len(needs) == 0 and isinstance(answer, Symbol):
                    return ToolsToFinalOutputResult(
                        is_final_output=False, final_output=None
                    )

                answer = "" if isinstance(answer, Symbol) else serialize_value(answer)
                needs = [serialize_value(need) for need in needs]

                return ToolsToFinalOutputResult(
                    is_final_output=True,
                    final_output=PartialQueryAnswer(answer=answer, needs=needs),
                )

            code_agent = self.agent.clone(
                name=f"{self.agent.name}_Code_Agent",
                instructions=load_prompt(
                    "SQL Data Agent/Code/Instructions",
                    {"m_schema": self.m_schema},
                ),
                tools=[code_environment.as_function_tool()],
                tool_use_behavior=custom_tool_handler,
                reset_tool_choice=False,
                model_settings=dataclasses.replace(
                    self.agent.model_settings,
                    parallel_tool_calls=False,
                    tool_choice="required",
                ),
                output_type=AgentOutputSchema(
                    PartialQueryAnswer, strict_json_schema=False
                ),
            )

            code_agent_prompt = load_prompt(
                "SQL Data Agent/Code/Prompt", {"query": query}
            )

            result = cast(
                PartialQueryAnswer,
                (await Runner.run(code_agent, code_agent_prompt)).final_output,
            )

            code_environment.append_data_to_span(span)
            span.span_data.data["sql_queries"] = [str(query) for query in sql_queries]
            span.span_data.data["partial_answer"] = {
                "answer": preview_text(str(result.answer), CHAR_LIMIT_SPAN),
                "needs": [
                    preview_text(str(need), CHAR_LIMIT_SPAN)
                    for need in result.needs or []
                ],
            }

            return result

    async def _initialize_schema(self):
        if self.schema_initialized:
            return

        m_schema_without_comments = self.schema_engine.mschema.to_mschema()

        schema_dict = {}
        for table_name, table_info in self.schema_engine.mschema.tables.items():
            fields = {}
            for col_name in table_info["fields"].keys():
                fields[col_name] = (str, Field(...))
            schema_dict[table_name] = (
                create_model(f"{table_name}ColumnDescriptions", **fields),
                Field(...),
            )
        schema_dict["database_description"] = (str, Field(...))

        DatabaseDescriptionModel = create_model(
            "DatabaseDescriptionModel", **schema_dict
        )

        schema_description_agent = self.agent.clone(
            name=f"{self.agent.name}_Schema_Description_Agent",
            instructions=load_prompt(
                "SQL Data Agent/Schema Description/Instructions", {}
            ),
            output_type=DatabaseDescriptionModel,
        )

        schema_prompt = load_prompt(
            "SQL Data Agent/Schema Description/Prompt",
            {"m_schema_without_comments": m_schema_without_comments},
        )

        result = (
            await Runner.run(schema_description_agent, schema_prompt)
        ).final_output

        # Update the schema with the generated descriptions
        for table_name, table_info in self.schema_engine.mschema.tables.items():
            for col_name, field_info in table_info["fields"].items():
                description = getattr(getattr(result, table_name, {}), col_name, None)
                if description:
                    field_info["comment"] = description

        self.natural_language_description = (
            result.database_description + " | Capabilities: Text-to-SQL"
        )
        self.m_schema = self.schema_engine.mschema.to_mschema()
        self.schema_initialized = True

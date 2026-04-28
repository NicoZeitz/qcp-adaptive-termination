import dataclasses
from typing import Any, Literal, Sequence, TypedDict
from agents import (
    Agent,
    AgentOutputSchema,
    FunctionToolResult,
    RunContextWrapper,
    Runner,
    ToolsToFinalOutputResult,
    custom_span,
    function_tool,
)
from sqlalchemy import Engine, Row, text
from modules.cache import Cache
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
    def __init__(
        self,
        agent: Agent[AgentContext],
        db_engine: Engine,
        type: Literal["code", "sql"] = "code",
    ):
        self.agent = agent
        self.db_engine = db_engine
        self.type = type

        self.schema_engine = SchemaEngine(
            engine=self.db_engine, db_name=self.agent.name
        )
        self.schema_initialized = False
        self.natural_language_description: str = ""
        self.m_schema: str = ""

    def name(self) -> str:
        return self.agent.name

    async def get_data_and_query_capabilities(self) -> str:
        await self._initialize_schema()
        return self.natural_language_description

    async def get_schema_for_query(self, query: str) -> str | None:
        await self._initialize_schema()

        instructions = load_prompt(
            "sql_data_agent_relevance_agent_instructions",
            None,
            {"m_schema": self.m_schema},
        )
        prompt = load_prompt(
            "sql_data_agent_relevance_agent_prompt", None, {"query": query}
        )

        relevance_agent = self.agent.clone(
            name=f"{self.agent.name}_Relevance_Agent",
            instructions=instructions,
            output_type=bool,
        )

        async def compute_is_relevant():
            return (await Runner.run(relevance_agent, prompt)).final_output

        is_relevant = await Cache.instance().get_or_compute(
            relevance_agent, prompt, bool, compute_is_relevant
        )

        if not is_relevant:
            return None

        return self.m_schema

    async def answer_query(
        self, query: str, answered_needs: list[Need]
    ) -> PartialQueryAnswer:
        if self.type == "code":
            return await self.answer_query_code(query, answered_needs)
        else:
            return await self.answer_query_sql(query, answered_needs)

    async def answer_query_code(
        self, query: str, answered_needs: list[Need]
    ) -> PartialQueryAnswer:
        await self._initialize_schema()

        with custom_span("SQL Data Agent - answer_query_code") as span:
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
                name=f"{self.agent.name}_SQL_Agent",
                instructions=load_prompt(
                    "sql_data_agent_code_agent_instructions",
                    None,
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

            async def compute_code_answer() -> PartialQueryAnswer:
                return (
                    await Runner.run(
                        code_agent,
                        load_prompt(
                            "sql_data_agent_code_agent_prompt", None, {"query": query}
                        ),
                    )
                ).final_output

            result = await compute_code_answer()

            span.span_data.data["answer"] = preview_text(
                str(result.answer), CHAR_LIMIT_SPAN
            )
            span.span_data.data["needs"] = [
                preview_text(str(need), CHAR_LIMIT_SPAN) for need in result.needs or []
            ]
            span.span_data.data["sql_queries"] = [str(query) for query in sql_queries]
            code_environment.append_data_to_span(span)

            return result

    async def answer_query_sql(
        self, query: str, answered_needs: list[Need]
    ) -> PartialQueryAnswer:
        await self._initialize_schema()

        additional_information = "\n".join(
            [f"{need.need}: {need.answer}" for need in answered_needs]
        )
        additional_information_section = (
            f"【Additional Information】\n{additional_information}"
            if additional_information
            else ""
        )

        @function_tool()
        async def execute_sql(ctx: RunContextWrapper[AgentContext], query: str) -> str:
            """Execute a SQL query and return the results.

            Args:
                query: The SQL query to execute.
            """
            print(f"        Executing SQL query: {query}")
            with self.db_engine.connect() as conn:
                result = conn.execute(text(query)).fetchall()
                return str(result)

        instructions = load_prompt(
            "sql_data_agent_sql_agent_instructions", None, {"m_schema": self.m_schema}
        )
        prompt = load_prompt(
            "sql_data_agent_sql_agent_prompt",
            None,
            {
                "query": query,
                "additional_information_section": additional_information_section,
            },
        )

        sql_agent = self.agent.clone(
            name=f"{self.agent.name}_SQL_Agent",
            instructions=instructions,
            tools=[execute_sql],
            output_type=PartialQueryAnswer,
        )

        async def compute_sql_answer() -> PartialQueryAnswer:
            return (await Runner.run(sql_agent, prompt)).final_output

        return await Cache.instance().get_or_compute(
            sql_agent,
            prompt,
            PartialQueryAnswer,
            compute_sql_answer,
        )

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
                "sql_data_agent_schema_description_agent_instructions", None, {}
            ),
            output_type=DatabaseDescriptionModel,
        )

        schema_prompt = load_prompt(
            "sql_data_agent_schema_description_agent_prompt",
            None,
            {"m_schema_without_comments": m_schema_without_comments},
        )

        async def compute_schema_description():
            return (
                await Runner.run(schema_description_agent, schema_prompt)
            ).final_output

        result = await Cache.instance().get_or_compute(
            schema_description_agent,
            schema_prompt,
            DatabaseDescriptionModel,
            compute_schema_description,
        )

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

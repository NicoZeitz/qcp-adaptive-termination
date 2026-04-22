import textwrap
from agents import Agent, RunContextWrapper, Runner, function_tool
from sqlalchemy import Engine, text
from modules.cache import Cache
from modules.model import PartialQueryAnswer, AgentContext, DataAgent
from mschema import m_schema
from mschema.schema_engine import SchemaEngine
from pydantic import Field, create_model

def schema_description_agent_instructions() -> str:
    return textwrap.dedent(f"""
You are a database documentation expert. For each column in the following database schema, write a concise description in natural language of what that column likely represents. Example values are not needed. Return the result as a JSON object matching the specified output type. Also generate a concise description of the whole database, including what kind of information it contains (tables, example columns) in natural language without technical details. Include this description in the "database_description" field of the output JSON.
""").strip()

def relevance_agent_instructions(m_schema: str) -> str:
    return textwrap.dedent(f"""
You are given a natural language query and a database schema. Determine if the query is relevant to the database. A query is relevant if any part of it could be answered using the database, even in a partial, indirect, or abstract way. Even loose or inferred connections count. Only return False if there is no connection between the query and the schema.

{m_schema}
""").strip()

def relevance_agent_prompt(query: str) -> str:
    return textwrap.dedent(f"""
Determine if the following natural language query is relevant to the database schema provided earlier. A query is relevant if any part of it could be answered using the database, even in a partial, indirect, or abstract way. Even loose or inferred connections count. Only return False if there is no connection between the query and the schema.

【Question】
{query}
""").strip()


# TODO: answer can be empty (currently it returns needs instead)
def sql_agent_instructions(m_schema: str) -> str:
    return textwrap.dedent(f"""
You are an expert at translating natural language questions into SQL queries. Your task is to generate SQL queries that can be executed on the given database schema to answer the user's question. The database schema is as follows:
{m_schema}

When generating SQL queries, follow these guidelines:
- Generate SQL queries that produce as little output as possible, but that are sufficient to answer the user's question.
- If the question can be answered with a simple query, generate that query. If the question requires multiple steps, break it down into sub-queries and generate those. You can use the results of sub-queries to build up to the final answer.
- Only generate queries that can be executed on the given schema. Do not make assumptions about the data that is not reflected in the schema.
- If the question cannot be answered with the given schema, return a string describing what information is needed to answer the question.
- The user may provide you with additional information that can help you generate the correct SQL query. Use this information if it is relevant.

Please follow the following steps:
1. Generate SQL query, e.g. SELECT …
2. Execute SQL query and check results:
If the query returns a result that answers the question, return the answer.
If the query contains an error, fix the error and try again.
If the query contains an error that cannot be fixed or a valid query answering the question cannot be generated with the current schema, return a string describing what information is needed to fix the query.
Try to answer the query first. Recheck and retry to be confident in your answer. If you are confident in your answer, return it. If after multiple retries or when you are sure information is missing, return the needs.

It may be the case that the question can still be answered partially with the current schema. In that case, return the answer that can be obtained with the current schema and also return what information is needed to get the full answer.
""").strip()

def sql_agent_prompt(query: str, additional_information: str) -> str:
    additional_information_section = f"【Additional Information】\n{additional_information}" if additional_information else ""

    return textwrap.dedent(f"""Please generate SQL query and execute that SQL query to answer the following question: {query}

{additional_information_section}
""").strip()

class SqlDataAgent(DataAgent):
    def __init__(
        self,
        agent: Agent[AgentContext],
        db_engine: Engine,
    ):
        self.agent = agent
        self.db_engine = db_engine
        self.schema_engine = SchemaEngine(engine=self.db_engine, db_name=self.agent.name)
        self.schema_initialized = False
        self.natural_language_description: str = ""
        self.m_schema: str = ""

    def name(self) -> str:
        return self.agent.name

    async def get_data_and_query_capabilities(self) -> str:
        if not self.schema_initialized:
            await self._initialize_schema()

        return self.natural_language_description

    async def get_schema_for_query(self, query: str) -> str | None:
        if not self.schema_initialized:
            await self._initialize_schema()

        relevance_agent = self.agent.clone(
            name=f"{self.agent.name}_Relevance_Agent",
            instructions=relevance_agent_instructions(self.m_schema),
            output_type=bool
        )

        async def compute_is_relevant():
            return (await Runner.run(
                relevance_agent,
                relevance_agent_prompt(query)
            )).final_output

        is_relevant = await Cache.instance().get_or_compute(relevance_agent, relevance_agent_prompt(query), bool, compute_is_relevant)

        if not is_relevant:
            return None

        return self.m_schema

    async def _initialize_schema(self):
        m_schema_without_comments = self.schema_engine.mschema.to_mschema()

        schema_dict = {}
        for table_name, table_info in self.schema_engine.mschema.tables.items():
            fields = {}
            for col_name in table_info["fields"].keys():
                fields[col_name] = (str, Field(...))
            schema_dict[table_name] = (create_model(f"{table_name}ColumnDescriptions", **fields), Field(...))
        schema_dict["database_description"] = (str, Field(...))

        DatabaseDescriptionModel = create_model("DatabaseDescriptionModel", **schema_dict)

        schema_description_agent = self.agent.clone(
            name=f"{self.agent.name}_Schema_Description_Agent",
            instructions=schema_description_agent_instructions(),
            output_type=DatabaseDescriptionModel
        )

        async def compute_schema_description():
            return (await Runner.run(
                schema_description_agent,
                m_schema_without_comments
            )).final_output

        result = await Cache.instance().get_or_compute(schema_description_agent, m_schema_without_comments, DatabaseDescriptionModel, compute_schema_description)

        # Update the schema with the generated descriptions
        for table_name, table_info in self.schema_engine.mschema.tables.items():
            for col_name, field_info in table_info["fields"].items():
                description = getattr(getattr(result, table_name, {}), col_name, None)
                if description:
                    field_info["comment"] = description

        self.natural_language_description = result.database_description + " | Capabilities: Text-to-SQL"
        self.m_schema = self.schema_engine.mschema.to_mschema()
        self.schema_initialized = True

    async def answer_query(self, query: str, additional_information: str) -> PartialQueryAnswer:
        @function_tool()
        async def execute_sql(ctx: RunContextWrapper[AgentContext], query: str) -> str:
            """Execute a SQL query and return the results.

            Args:
                query: The SQL query to execute.
            """
            print(f"        Executing SQL query: {query}") # TODO: Block non read queries)
            with self.db_engine.connect() as conn:
                result = conn.execute(text(query)).fetchall()
                return str(result)

        sql_agent = self.agent.clone(
            name=f"{self.agent.name}_SQL_Agent",
            instructions=sql_agent_instructions(self.m_schema),
            tools=[execute_sql],
            output_type=PartialQueryAnswer
        )

        async def compute_sql_answer() -> PartialQueryAnswer:
            return (await Runner.run(
                sql_agent,
                sql_agent_prompt(query, additional_information)
            )).final_output

        return await Cache.instance().get_or_compute(sql_agent, sql_agent_prompt(query, additional_information), PartialQueryAnswer, compute_sql_answer)

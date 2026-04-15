import asyncio
from agents import Agent, FunctionTool, Runner, RunContextWrapper, function_tool
from typing_extensions import TypedDict, Any
import os
from dotenv import load_dotenv
import sqlite3
load_dotenv()
# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def create_data_agent(
        conn: sqlite3.Connection,
        agent_name: str,
        agent_tool_name: str,
        specific_agent_instructions: str) -> tuple[Agent, FunctionTool]:
    def create_instructions_for_data_agent(conn: sqlite3.Connection, specific_agent_instructions: str) -> str:
        def query_schema(conn: sqlite3.Connection) -> str:
            table_names = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall()
            output = ""
            for table in table_names:
                table_name = table[0]
                columns = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
                column_names = [f"{column[1]} {column[2]}" for column in columns]

                output += f"{table_name}({', '.join(column_names)})\n"

                example_rows = conn.execute(f"SELECT * FROM {table_name} LIMIT 3").fetchall()
                for row in example_rows:
                    output += str(row) + "\n"
            return output

        schema = query_schema(conn)
        instructions = f"""
You are an agent with information about {specific_agent_instructions}. You have a local database and are capable of Text-to-SQL. You can use the tool 'execute_sql' to execute SQL queries on your local database. Produce SQL queries that produce as little output as possible, but that are sufficient to answer the user's question. Your database schema together with example values are:
{schema}
        """.strip()
        return instructions

    instructions = create_instructions_for_data_agent(conn, specific_agent_instructions)

    @function_tool()
    async def execute_sql(ctx: RunContextWrapper[Any], query: str) -> str:
        """Execute a SQL query and return the results.

        Args:
            query: The SQL query to execute.
        """
        print(f"Executing SQL query: {query}")
        result = conn.execute(query).fetchall()
        return str(result)

    data_agent = Agent(
        name=agent_name,
        instructions=instructions,
        tools=[execute_sql],
    )
    data_agent_tool = data_agent.as_tool(
        tool_name=agent_tool_name,
        tool_description=f"""I am an agent with information about {specific_agent_instructions}. I have a local database and am capable of Text-to-SQL""".strip(),
    )
    return data_agent, data_agent_tool

agent1, agent_1_tool = create_data_agent(
    conn=sqlite3.connect('./data/agent1.db'),
    agent_name="People Agent",
    agent_tool_name="agent_1",
    specific_agent_instructions="people, such as their id or name",
)
agent2, agent_2_tool = create_data_agent(
    conn=sqlite3.connect('./data/agent2.db'),
    agent_name="Crew Agent",
    agent_tool_name="agent_2",
    specific_agent_instructions="crew members, such the movie and role they played in and the actual person behind it",
)
agent3, agent_3_tool = create_data_agent(
    conn=sqlite3.connect('./data/agent3.db'),
    agent_name="Ratings and Episodes Agent",
    agent_tool_name="agent_3",
    specific_agent_instructions="movie ratings and episodes, such as the rating value, the number of votes, or the season and episode number",
)
agent4, agent_4_tool = create_data_agent(
    conn=sqlite3.connect('./data/agent4.db'),
    agent_name="Titles Agent",
    agent_tool_name="agent_4",
    specific_agent_instructions="movie titles, such as their id or name",
)
agent5, agent_5_tool = create_data_agent(
    conn=sqlite3.connect('./data/agent5.db'),
    agent_name="Titles and AKAs Agent",
    agent_tool_name="agent_5",
    specific_agent_instructions="movie titles, such as their id or name, as well as alternative titles (AKAs), such as the region, language, or type of the aka",
)

orchestrator = Agent(
    name="Orchestrator",
    instructions="""
You are an orchestrator agent. Your task is to answer user queries by calling the appropriate tools.
    """.strip(),
    tools=[
        agent_1_tool,
        agent_2_tool,
        agent_3_tool,
        agent_4_tool,
        agent_5_tool,
    ],
)

async def main():
    query = "What's the actors name who played Rocket in Guardians of the Galaxy?"
    result = await Runner.run(orchestrator, query)
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
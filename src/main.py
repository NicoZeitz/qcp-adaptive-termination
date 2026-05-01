from agents import trace

from modules.config import create_sql_data_agents_from_folder, create_text_agents_from_folder, get_default_query_coordinator_agent_config, load_query
from modules.initialization import initialize
from modules.cache import Cache
from modules.model import ReplanMode
from modules.query_coordinator import QueryCoordinator


async def main(
    sql_data_agents_path: str,
    text_data_agents_path: str,
    query_path: str,
    replan_mode: ReplanMode,
):
    user_query, _sql_query, expected_result = load_query(query_path)

    with trace("NZ: Collaborative Query Answering", metadata={"query": user_query}):
        print(
            "================================================== REGISTERING ======================================================"
        )
        queryCoordinator = QueryCoordinator(
            agent=get_default_query_coordinator_agent_config("Query Coordinator"),
            replan_mode=replan_mode,
        )

        sql_data_agents = create_sql_data_agents_from_folder(sql_data_agents_path)
        for data_agent in sql_data_agents:
            queryCoordinator.register_data_agent(data_agent)

        text_data_agents = create_text_agents_from_folder(text_data_agents_path)
        for data_agent in text_data_agents:
            queryCoordinator.register_data_agent(data_agent)

        print(
            "================================================== USER QUERY ======================================================="
        )
        print(user_query)
        print(
            "================================================== ANSWERING ========================================================"
        )
        result = await queryCoordinator.collaborative_query_answering(user_query)
        print(
            "================================================== RESULT ==========================================================="
        )
        print(result)
        print(
            "================================================== EXPECTED RESULT =================================================="
        )
        print(expected_result)


if __name__ == "__main__":
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(prog="Collaborative Query Answering")
    parser.add_argument("--sql-data-agents", type=str, default="../data/handmade")
    parser.add_argument("--text-data-agents", type=str, default="../data/text")
    parser.add_argument(
        "--query",
        type=str,
        default="../test/What is the name of the person with id 42?.json",
    )
    parser.add_argument(
        "--replan-mode",
        type=str,
        choices=["global", "local"],
        default="local",
    )
    args = parser.parse_args()

    initialize()
    Cache.instance().set_mode("DISABLED") # TODO: Remove cache
    asyncio.run(
        main(
            args.sql_data_agents,
            args.text_data_agents,
            args.query,
            ReplanMode(args.replan_mode),
        )
    )

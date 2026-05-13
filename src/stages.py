from io import TextIOWrapper
import os
from typing import Literal
import json

from agents import trace

from data_agent.data_agent import DataAgent
from qtypes.query_plan.query_plan import QueryPlan
from qtypes.query_plan.result import result_to_json
from query_coordinator.default_query_coordinator import DefaultQueryCoordinator
from query_coordinator.query_coordinator import QueryCoordinator
from utils.initialization import initialize, langfuse
from utils.load_agents import (
    create_sql_data_agents_from_folder,
    create_text_agents_from_folder,
)

stages_mapping = {
    0: "All stages",
    1: "Filter relevant agents",
    2: "Draft query plan",
    3: "Execute query plan",
}


def print_to_log_file(log_file: TextIOWrapper, message: str):
    log_file.write(message + "\n")
    print(message)


async def main(
    sql_data_agents_path: str,
    text_data_agents_path: str,
    test_file: str,
    stage: int,
    provider: Literal["University", "Work"],
):
    print(
        f"Running stage {stage}: {stages_mapping[stage]} for test file {test_file} with provider {provider}"
    )

    user = "NZ" if os.getlogin() == "I550817" else "FW"

    with open(test_file, "r") as f:
        test_data = json.load(f)

    query = test_data["query"]
    answer = test_data["answer"]
    sql_query = test_data["sql_query"]
    relevant_agents = test_data["relevant_data_agents"]
    query_plan = QueryPlan.load_from_json(test_data["query_plan"])

    log_file_path = os.path.join(
        os.path.dirname(test_file),
        "logs",
        f"{os.path.splitext(os.path.basename(test_file))[0]}_{stage}.log",
    )
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    with (
        trace(
            f"{user}: Stage {stage} - {stages_mapping[stage]}",
            metadata={"test_file": test_file, "query": query, "provider": provider},
        ),
        open(log_file_path, "w") as log_file,
    ):
        data_agents: list[DataAgent] = []
        data_agents.extend(create_sql_data_agents_from_folder(sql_data_agents_path))
        data_agents.extend(create_text_agents_from_folder(text_data_agents_path))
        query_coordinator: QueryCoordinator = DefaultQueryCoordinator()

        print_to_log_file(log_file, f"QUERY: {query}")
        print_to_log_file(log_file, f"SQL QUERY: {sql_query}")

        if stage == 0:
            # All stages
            # Filter relevant agents
            relevant_data_agents_with_reasoning = (
                await query_coordinator.filter_relevant_agents(data_agents, query)
            )
            data_agents = await query_coordinator.ask_relevant_agents(
                relevant_data_agents_with_reasoning, query
            )

            print_to_log_file(
                log_file,
                f"RELEVANT AGENTS FROM COORDINATOR: {sorted([agent.name for agent, _ in relevant_data_agents_with_reasoning])}",
            )
            print_to_log_file(
                log_file,
                f"RELEVANT AGENTS: {sorted([agent.name for agent in data_agents])}",
            )
            print_to_log_file(
                log_file, f"EXPECTED RELEVANT AGENTS: {sorted(relevant_agents)}"
            )
            print_to_log_file(
                log_file,
                f"RELEVANT AGENTS ARE {'TOO FEW' if len(data_agents) < len(relevant_agents) else 'TOO MANY' if len(data_agents) > len(relevant_agents) else 'EQUAL'}\n",
            )
            print_to_log_file(
                log_file,
                f"RELEVANT AGENTS WITH REASONING:\n{'\n'.join([f'{agent.name}: {reasoning}' for agent, reasoning in relevant_data_agents_with_reasoning])}\n",
            )

            # Draft query plan
            draft_query_plan = await query_coordinator.draft_query_plan(
                data_agents, query
            )

            print_to_log_file(
                log_file,
                f"DRAFTED QUERY PLAN:\n{json.dumps(draft_query_plan.to_json(), indent=2)}\n",
            )
            print_to_log_file(
                log_file,
                f"EXPECTED QUERY PLAN:\n{json.dumps(query_plan.to_json(), indent=2)}\n",
            )

            # Execute Query Plan
            result = await query_coordinator.execute_query_plan(
                data_agents, draft_query_plan
            )

            print_to_log_file(
                log_file, f"RESULT:\n{json.dumps(result_to_json(result), indent=2)}\n"
            )
            print_to_log_file(
                log_file, f"EXPECTED ANSWER:\n{json.dumps(answer, indent=2)}\n"
            )
        elif stage == 1:
            # Filter relevant agents
            relevant_data_agents_with_reasoning = (
                await query_coordinator.filter_relevant_agents(data_agents, query)
            )

            data_agents = await query_coordinator.ask_relevant_agents(
                relevant_data_agents_with_reasoning, query
            )

            print_to_log_file(
                log_file,
                f"RELEVANT AGENTS FROM COORDINATOR: {sorted([agent.name for agent, _ in relevant_data_agents_with_reasoning])}",
            )
            print_to_log_file(
                log_file,
                f"RELEVANT AGENTS: {sorted([agent.name for agent in data_agents])}",
            )
            print_to_log_file(
                log_file, f"EXPECTED RELEVANT AGENTS: {sorted(relevant_agents)}"
            )
            print_to_log_file(
                log_file,
                f"RELEVANT AGENTS ARE {'TOO FEW' if len(data_agents) < len(relevant_agents) else 'TOO MANY' if len(data_agents) > len(relevant_agents) else 'EQUAL'}\n",
            )
            print_to_log_file(
                log_file,
                f"RELEVANT AGENTS WITH REASONING:\n{'\n'.join([f'{agent.name}: {reasoning}' for agent, reasoning in relevant_data_agents_with_reasoning])}\n",
            )
        elif stage == 2:
            # Draft query plan
            data_agents = [
                agent for agent in data_agents if agent.name in relevant_agents
            ]

            print_to_log_file(
                log_file,
                f"RELEVANT AGENTS: {sorted([agent.name for agent in data_agents])}",
            )

            draft_query_plan = await query_coordinator.draft_query_plan(
                data_agents, query
            )

            print_to_log_file(
                log_file,
                f"DRAFTED QUERY PLAN:\n{json.dumps(draft_query_plan.to_json(), indent=2)}\n",
            )
            print_to_log_file(
                log_file,
                f"EXPECTED QUERY PLAN:\n{json.dumps(query_plan.to_json(), indent=2)}\n",
            )
        elif stage == 3:
            # Execute query plan
            data_agents = [
                agent for agent in data_agents if agent.name in relevant_agents
            ]

            print_to_log_file(log_file, f"QUERY: {query}")
            print_to_log_file(
                log_file,
                f"RELEVANT AGENTS: {sorted([agent.name for agent in data_agents])}",
            )
            print_to_log_file(
                log_file,
                f"EXECUTED QUERY PLAN:\n{json.dumps(query_plan.to_json(), indent=2)}\n",
            )

            result = await query_coordinator.execute_query_plan(data_agents, query_plan)

            print_to_log_file(
                log_file, f"RESULT:\n{json.dumps(result_to_json(result), indent=2)}\n"
            )
            print_to_log_file(
                log_file, f"EXPECTED ANSWER:\n{json.dumps(answer, indent=2)}\n"
            )
        else:
            raise ValueError("Invalid stage")

    langfuse.flush()


async def multiple_main(
    sql_data_agents_path: str,
    text_data_agents_path: str,
    test_dir: str,
    stage: int,
    provider: Literal["University", "Work"],
):
    test_files = [
        os.path.join(test_dir, file)
        for file in os.listdir(test_dir)
        if file.endswith(".json")
    ]
    for test_file in test_files:
        await main(
            sql_data_agents_path, text_data_agents_path, test_file, stage, provider
        )


if __name__ == "__main__":
    import argparse
    import asyncio

    # uv run stages.py --test-file "../test/Who was the cinematographer of Donald Trump's favorite movie?.json" --stage 1
    parser = argparse.ArgumentParser(prog="Collaborative Query Answering")
    parser.add_argument("--stage", type=int, choices=[0, 1, 2, 3], default=1)
    parser.add_argument("--test-dir", type=str, default="../test/")
    parser.add_argument("--test-file", type=str)
    parser.add_argument("--sql-data-agents", type=str, default="../data/handmade")
    parser.add_argument("--text-data-agents", type=str, default="../data/text")
    parser.add_argument(
        "--provider", type=str, choices=["University", "Work"], default="University"
    )
    args = parser.parse_args()

    initialize(args.provider)
    if args.test_file:
        asyncio.run(
            main(
                args.sql_data_agents,
                args.text_data_agents,
                args.test_file,
                args.stage,
                args.provider,
            )
        )
    else:
        asyncio.run(
            multiple_main(
                args.sql_data_agents,
                args.text_data_agents,
                args.test_dir,
                args.stage,
                args.provider,
            )
        )

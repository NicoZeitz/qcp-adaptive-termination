from typing import Literal

from pydantic import BaseModel, create_model

from query_coordinator.build_agent_registry import build_agent_registry
from qtypes.query_plan.query_plan import (
    FieldDefinitions,
    QueryDependency,
    QueryPlan,
    QueryPlanStep,
)
from data_agent.data_agent import DataAgent
from query_coordinator.query_coordinator import QueryCoordinator
from utils.prompt import load_prompt
from utils.run_agent import run_agent
from utils.span import append_to_span_input, append_to_span_output, own_custom_span


def create_draft_query_plan_model(data_agents: list[DataAgent]) -> type[BaseModel]:
    data_agent_names = [agent.name for agent in data_agents]

    QueryDependency = create_model(
        "QueryDependency",
        step_id=(str, FieldDefinitions.dependency_step_id),
        dependency=(str, FieldDefinitions.dependency_description),
    )

    QueryPlanStep = create_model(
        "QueryPlanStep",
        id=(str, FieldDefinitions.step_id),
        query=(str, FieldDefinitions.step_query),
        to_agent=(
            Literal["Coordinator", *data_agent_names],
            FieldDefinitions.step_to_agents,
        ),
        dependencies=(list[QueryDependency], FieldDefinitions.step_dependencies),
    )

    QueryPlan = create_model(
        "QueryPlan", steps=(list[QueryPlanStep], FieldDefinitions.plan_steps)
    )
    return QueryPlan


def map_query_plan_model_to_query_plan(
    query_plan_model: BaseModel, query: str
) -> QueryPlan:
    step_mapping: dict[str, QueryPlanStep] = {}
    steps = []

    for step_model in getattr(query_plan_model, "steps", []):
        if getattr(step_model, "id", None) in step_mapping:
            continue  # TODO:DRAFT_ERROR Ignore duplicate step ids, only the first one will be considered valid

        dependencies = []
        for dependency_model in getattr(step_model, "dependencies", []):
            if getattr(dependency_model, "step_id", None) not in step_mapping:
                continue  # TODO:DRAFT_ERROR Ignore invalid dependencies that refer to steps that are not defined in the plan

            dependencies.append(
                QueryDependency(
                    step=step_mapping[dependency_model.step_id],
                    dependency=dependency_model.dependency,
                )
            )

        query_plan_step = QueryPlanStep(
            id=step_model.id,
            query=step_model.query,
            to_agent=step_model.to_agent,
            dependencies=dependencies,
            answer=None,
        )

        step_mapping[step_model.id] = query_plan_step
        steps.append(query_plan_step)

    query_plan = QueryPlan(query=query, steps=steps)
    return query_plan


async def draft_query_plan(
    coordinator: QueryCoordinator, data_agents: list[DataAgent], query: str
) -> QueryPlan:
    with own_custom_span("Coordinator - Draft Query Plan") as span:
        append_to_span_input(span, {"query": query})

        agent_registry = await build_agent_registry(data_agents, span)
        QueryPlanModel = create_draft_query_plan_model(data_agents)

        result = await run_agent(
            name_postfix="Draft Query Plan",
            agent_configuration=coordinator.agent_configuration,
            system_prompt=load_prompt(
                "Query Coordinator/Query Plan/Instructions",
                {"agent_registry": agent_registry},  # TODO: change prompt
            ),
            prompt=load_prompt("Query Coordinator/Query Plan/Prompt", {"query": query}),
            output_type=QueryPlanModel,
        )

        query_plan = map_query_plan_model_to_query_plan(result, query)
        append_to_span_output(span, {"query_plan": query_plan.to_span_data()})
        return query_plan

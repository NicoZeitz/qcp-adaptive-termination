from typing import Literal

from pydantic import BaseModel, create_model

from qtypes.drafted_query_plan import DraftedQueryPlan
from qtypes.data_agent import DataAgent
from qtypes.query_dependency import QueryDependency
from qtypes.query_plan import FieldDefinitions, QueryPlan
from qtypes.query_plan_step import QueryPlanStep
from stages.common.agent_configuration import get_coordinator_agent_configuration
from stages.common.data_agent.agent_registry import (
    AgentRegistryType,
    build_agent_registry,
)
from utils.cache import cached
from utils.run_agent import run_agent
from utils.span import append_to_span_output, own_custom_span


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
                    step_id=dependency_model.step_id,
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


@cached(name_prefix="Stage 2: Draft Query Plan", mode="DISABLED")
async def draft_query_plan(
    query: str, data_agents: list[DataAgent], relevant_data_agents: list[DataAgent]
) -> DraftedQueryPlan:
    with own_custom_span(
        "Coordinator - Draft Query Plan",
        input={
            "query": query,
            "data_agents": data_agents,
            "relevant_data_agents": relevant_data_agents,
        },
    ):
        agent_registry = await build_agent_registry(
            relevant_data_agents, agent_registry_type=AgentRegistryType.TECHNICAL
        )
        QueryPlanModel = create_draft_query_plan_model(relevant_data_agents)

        result = await run_agent(
            name_postfix="Draft Query Plan",
            agent_configuration=get_coordinator_agent_configuration(),
            prompt_name="Query Coordinator/Query Plan",
            prompt_parameters={
                "agent_registry": agent_registry,
                "query": query,
            },
            output_type=QueryPlanModel,
        )

        query_plan = map_query_plan_model_to_query_plan(result, query)
        append_to_span_output({"query_plan": query_plan})

        return DraftedQueryPlan(
            relevant_data_agents=relevant_data_agents,
            data_agents=data_agents,
            query_plan=query_plan,
        )

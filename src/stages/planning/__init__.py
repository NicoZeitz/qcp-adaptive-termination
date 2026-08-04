from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from qtypes.drafted_query_plan import DraftedQueryPlan
from qtypes.data_agent import DataAgent
from qtypes.query_dependency import QueryDependency
from qtypes.query_plan import FieldDefinitions, QueryPlan
from qtypes.query_plan_step import (
    MultiAgentQueryPlanStep,
    QueryPlanStep,
    SingleAgentQueryPlanStep,
)
from stages.common.agent_configuration import get_coordinator_agent_configuration
from stages.common.data_agent.agent_registry import (
    AgentRegistry,
    AgentRegistryType,
)
from utils.cache import cached
from utils.run_agent import run_llm
from utils.span import append_to_span_output, own_custom_span

STAGE_NAME = "Stage 2: Planning"


def create_draft_query_plan_model(data_agents: list[DataAgent]) -> type[BaseModel]:
    data_agent_names = [agent.name for agent in data_agents]
    AgentName = Literal["Coordinator", *data_agent_names]
    DataAgentName = Literal[*data_agent_names]
    StrictModelConfig = ConfigDict(extra="forbid")

    QueryDependency = create_model(
        "QueryDependency",
        step_id=(str, FieldDefinitions.dependency_step_id),
        dependency=(str, FieldDefinitions.dependency_description),
    )

    SingleAgentQueryPlanStepModel = create_model(
        "SingleAgentQueryPlanStep",
        id=(str, FieldDefinitions.step_id),
        query=(str, FieldDefinitions.step_query),
        to_agent=(AgentName, FieldDefinitions.step_to_agents),
        dependencies=(list[QueryDependency], FieldDefinitions.step_dependencies),
        __config__=StrictModelConfig,
    )

    MultiAgentQueryPlanStepModel = create_model(
        "MultiAgentQueryPlanStep",
        id=(str, FieldDefinitions.step_id),
        query=(str, FieldDefinitions.step_query),
        to_agent=(
            list[DataAgentName],
            Field(
                description=FieldDefinitions.step_to_agents.description,
                min_length=2,
            ),
        ),
        merge_task=(str, FieldDefinitions.step_merge_task),
        dependencies=(list[QueryDependency], FieldDefinitions.step_dependencies),
        table_name=(
            str,
            Field(
                description="The table name agents will primarily query in this step. ",
            ),
        ),
        column_name=(
            str,
            Field(
                description="The column name agents will primarily query in this step in the table specified by table_name. ONLY INCLUDE A SINGLE COLUMN NAME",
            ),
        ),
        __config__=StrictModelConfig,
    )

    QueryPlan = create_model(
        "QueryPlan",
        steps=(
            list[SingleAgentQueryPlanStepModel | MultiAgentQueryPlanStepModel],
            FieldDefinitions.plan_steps,
        ),
        __config__=StrictModelConfig,
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

        to_agent = step_model.to_agent
        if isinstance(to_agent, list):
            query_plan_step = MultiAgentQueryPlanStep(
                id=step_model.id,
                query=step_model.query,
                to_agent=to_agent,
                merge_task=step_model.merge_task,
                dependencies=dependencies,
                answer=None,
                table_name=step_model.table_name,
                column_name=step_model.column_name,
            )
        else:
            query_plan_step = SingleAgentQueryPlanStep(
                id=step_model.id,
                query=step_model.query,
                to_agent=to_agent,
                dependencies=dependencies,
                answer=None,
            )

        step_mapping[step_model.id] = query_plan_step
        steps.append(query_plan_step)

    query_plan = QueryPlan(query=query, steps=steps)
    return query_plan


@cached(name_prefix=STAGE_NAME, mode="WRITE_ONLY")
async def draft_query_plan(
    query: str, data_agents: list[DataAgent], relevant_data_agents: list[DataAgent]
) -> DraftedQueryPlan:
    with own_custom_span(
        STAGE_NAME,
        input={
            "query": query,
            "data_agents": data_agents,
            "relevant_data_agents": relevant_data_agents,
        },
    ):
        agent_registry = await AgentRegistry.build_registry_string(
            relevant_data_agents, registry_type=AgentRegistryType.TECHNICAL
        )
        QueryPlanModel = create_draft_query_plan_model(relevant_data_agents)

        result = await run_llm(
            agent_configuration=get_coordinator_agent_configuration("Planning"),
            prompt_name="planning-coordinator-planning",
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

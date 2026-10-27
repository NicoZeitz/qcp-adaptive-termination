"""Planning stage: draft a query plan for the selected data agents."""

import logging
from collections.abc import Callable, Mapping
from typing import Literal

from agents.items import TResponseInputItem
from agents.run_config import DEFAULT_MAX_TURNS
from pydantic import BaseModel, ConfigDict, Field, create_model

import qcp_adaptive_termination.agent_registry as agent_registry
from qcp_adaptive_termination import hooks
from qcp_adaptive_termination.hooks import HookEvent
from qcp_adaptive_termination.llm import runner
from qcp_adaptive_termination.models import AgentRegistryType, AgentRole, CacheMode, DataAgent
from qcp_adaptive_termination.models.query_plan import (
    FieldDefinitions,
    MultiAgentQueryPlanStep,
    QueryDependency,
    QueryPlan,
    QueryPlanStep,
    SingleAgentQueryPlanStep,
)
from qcp_adaptive_termination.models.stages import DraftedQueryPlan, Stage
from qcp_adaptive_termination.observability import cache, tracing
from qcp_adaptive_termination.observability import context as obs_context


def correction_input(error: ValueError) -> TResponseInputItem:
    """Build a user message asking a plan agent to fix an invalid plan."""
    return {
        "type": "message",
        "role": "user",
        "content": [
            {
                "type": "input_text",
                "text": f"Your query plan was invalid: {error}. Please re-draft a corrected query plan.",
            }
        ],
    }


async def run_plan_agent(
    *,
    role: AgentRole,
    prompt_parameters: Mapping[str, str | int | float | bool | None],
    output_type: type[BaseModel],
    map_output: Callable[[BaseModel], QueryPlan],
    error_kind: Literal["drafting", "replanning"],
    history: list[TResponseInputItem] | None = None,
) -> tuple[QueryPlan, list[TResponseInputItem]]:
    """Run a plan agent, retrying with correction feedback until a valid plan is produced.

    Args:
        role: the agent role to run
        prompt_parameters: prompt parameters for the role
        output_type: the query plan output model
        map_output: converts a model output to a validated QueryPlan (raises ValueError if invalid)
        error_kind: short action noun for logging and error messages
        history: prior conversation to continue from

    Returns:
        (QueryPlan, list[TResponseInputItem]): the validated plan and its continuation
    """
    input_items = list(history) if history is not None else []
    first = True
    for _ in range(obs_context.get_config().execution.llm_max_turns or DEFAULT_MAX_TURNS):
        run_result = await runner.run_llm(
            role=role,
            prompt_parameters=prompt_parameters,
            output_type=output_type,
            previous_input=input_items or None,
            render_prompt_after_previous_history=first,
        )

        first = False

        try:
            query_plan = map_output(run_result.output)
            return query_plan, run_result.continuation_input
        except ValueError as error:
            logging.warning("Invalid query plan output, retrying: %s", error)
            input_items = [*run_result.continuation_input, correction_input(error)]

    raise RuntimeError(f"Query plan {error_kind} exceeded max_turns")


def create_query_plan_model(data_agents: list[DataAgent]) -> type[BaseModel]:
    """Build a Pydantic model describing the query plan's steps.

    Args:
        data_agents: the data agents the plan may target

    Returns:
        (type[BaseModel]): the dynamic query plan model
    """
    data_agent_names = [agent.name for agent in data_agents]
    AgentName = Literal["Coordinator", *data_agent_names]  # ty: ignore[invalid-type-form]
    DataAgentName = Literal[*data_agent_names]  # ty: ignore[invalid-type-form]
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
        to_agent=(AgentName, FieldDefinitions.step_to_agents),  # pyright: ignore[reportArgumentType]
        dependencies=(
            list[QueryDependency],  # ty: ignore[invalid-type-form]  # pyright: ignore[reportInvalidTypeForm]
            FieldDefinitions.step_dependencies,
        ),
        __config__=StrictModelConfig,
    )

    MultiAgentQueryPlanStepModel = create_model(
        "MultiAgentQueryPlanStep",
        id=(str, FieldDefinitions.step_id),
        query=(str, FieldDefinitions.step_query),
        to_agent=(
            list[DataAgentName],  # pyright: ignore[reportInvalidTypeForm]
            Field(
                description=FieldDefinitions.step_to_agents.description,
                min_length=2,
            ),
        ),
        merge_task=(str, FieldDefinitions.step_merge_task),
        dependencies=(
            list[QueryDependency],  # ty: ignore[invalid-type-form]  # pyright: ignore[reportInvalidTypeForm]
            FieldDefinitions.step_dependencies,
        ),
        __config__=StrictModelConfig,
    )

    PlanSteps = list[SingleAgentQueryPlanStepModel | MultiAgentQueryPlanStepModel]  # ty: ignore[invalid-type-form]

    QueryPlan = create_model(
        "QueryPlan",
        steps=(
            PlanSteps,  # pyright: ignore[reportInvalidTypeForm]
            FieldDefinitions.plan_steps,
        ),
        __config__=StrictModelConfig,
    )
    return QueryPlan


def map_query_plan_model_to_query_plan(query_plan_model: BaseModel, query: str) -> QueryPlan:
    """Convert a drafted query plan model into a validated QueryPlan.

    Args:
        query_plan_model: the model output by the planning LLM
        query: the user query

    Returns:
        (QueryPlan): the parsed query plan
    """
    step_mapping: dict[str, QueryPlanStep] = {}
    steps = []

    for step_model in getattr(query_plan_model, "steps", []):
        if getattr(step_model, "id", None) in step_mapping:
            raise ValueError(f"Duplicate step id found: {step_model.id}")

        dependencies = []
        for dependency_model in getattr(step_model, "dependencies", []):
            if getattr(dependency_model, "step_id", None) not in step_mapping:
                raise ValueError(f"Invalid dependency: step {dependency_model.step_id} not found in the plan")

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


@hooks.hook(HookEvent.PLANNING_STAGE, stage_context=True)
@cache.cached(mode=CacheMode.WRITE_ONLY)
async def planning_stage(
    query: str, data_agents: list[DataAgent], relevant_data_agents: list[DataAgent]
) -> DraftedQueryPlan:
    """Plan the query steps for the relevant data agents.

    Args:
        query: the user query
        data_agents: all known data agents
        relevant_data_agents: the selected data agents

    Returns:
        (DraftedQueryPlan): the drafted query plan
    """
    return await draft_query_plan(query, data_agents, relevant_data_agents)


@hooks.hook(HookEvent.PLANNING_DRAFT_QUERY_PLAN)
async def draft_query_plan(
    query: str, data_agents: list[DataAgent], relevant_data_agents: list[DataAgent]
) -> DraftedQueryPlan:
    """Draft a query plan against the relevant data agents.

    Args:
        query: the user query
        data_agents: all known data agents
        relevant_data_agents: the selected data agents

    Returns:
        (DraftedQueryPlan): the drafted query plan
    """
    with tracing.own_span(
        Stage.PLANNING.label,
        input={
            "query": query,
            "data_agents": data_agents,
            "relevant_data_agents": relevant_data_agents,
        },
    ):
        registry_string = await agent_registry.build_registry_string(
            relevant_data_agents, registry_type=AgentRegistryType.TECHNICAL
        )
        QueryPlanModel = create_query_plan_model(relevant_data_agents)

        query_plan, _ = await run_plan_agent(
            role=AgentRole.PLANNING_COORDINATOR_DRAFT_QUERY_PLAN,
            prompt_parameters={
                "agent_registry": registry_string,
                "query": query,
            },
            output_type=QueryPlanModel,
            map_output=lambda out: map_query_plan_model_to_query_plan(out, query),
            error_kind="drafting",
        )

        tracing.append_to_span_output({"query_plan": query_plan})

        drafted_query_plan = DraftedQueryPlan(
            relevant_data_agents=relevant_data_agents,
            data_agents=data_agents,
            query_plan=query_plan,
        )
        return drafted_query_plan

import json

from pydantic import BaseModel

from data_agent.data_agent import DataAgent
from query_coordinator.build_agent_registry import build_agent_registry
from query_coordinator.planning.draft_query_plan import (
    create_draft_query_plan_model,
    map_query_plan_model_to_query_plan,
)
from query_coordinator.query_coordinator import QueryCoordinator
from qtypes.query_plan.need import needs_to_json_value
from qtypes.query_plan.query_plan import (
    QueryPlan,
)
from qtypes.query_plan.query_plan_step_execution_result import FailedExecution
from qtypes.query_plan.result import (
    ResultType,
    result_to_json_preview,
)
from utils.run_agent import run_agent


def create_answer_map_from_query_plan(query_plan: QueryPlan) -> dict[str, ResultType]:
    answer_map = {}
    for step in query_plan.steps:
        if step.answer is not None:
            answer_map[step.id] = step.answer
    return answer_map


def create_query_plan_prompt_context(query_plan: QueryPlan) -> str:
    return json.dumps(
        {
            "steps": [
                {
                    "id": step.id,
                    "query": step.query,
                    "to_agent": step.to_agent,
                    "dependencies": [
                        {
                            "step_id": dependency.step.id,
                            "dependency": dependency.dependency,
                        }
                        for dependency in step.dependencies
                    ],
                    "answer_preview": result_to_json_preview(step.answer)
                    if step.answer is not None
                    else None,
                }
                for step in query_plan.steps
            ]
        },
        indent=2,
    )


def create_reason_prompt_context(reason: FailedExecution) -> str:
    reason_data = {
        "raised_needs": needs_to_json_value(reason.needs),
        "failed_step_id": reason.step.id,
        "failed_step_query": reason.step.query,
        "failed_step_agent": reason.step.to_agent,
    }
    if reason.partial_answer is not None:
        reason_data["partial_answer"] = result_to_json_preview(reason.partial_answer)

    return json.dumps(reason_data, indent=2)


def create_query_replan_model(data_agents: list[DataAgent]) -> type[BaseModel]:
    # Use same model as draft query plan since the structure of the query plan is the same, we just need to re-use the step ids to restore the answers in the new query plan
    return create_draft_query_plan_model(data_agents)


def map_query_replan_model_to_query_plan(
    query_replan_model: BaseModel, query: str, answer_map: dict[str, ResultType]
) -> QueryPlan:
    # The mapping from the query replan model to the query plan is the same as the mapping from the draft query plan model to the query plan since they have the same structure
    query_plan = map_query_plan_model_to_query_plan(query_replan_model, query)

    # restore answers
    for step in query_plan.steps:
        if step.id in answer_map:
            step.answer = answer_map[step.id]

    return query_plan


async def replan_query_plan(
    coordinator: QueryCoordinator,
    data_agents: list[DataAgent],
    query_plan: QueryPlan,
    reason: FailedExecution,
) -> QueryPlan:
    agent_registry = await build_agent_registry(data_agents, span=None)
    answer_map = create_answer_map_from_query_plan(query_plan)
    QueryReplanModel = create_query_replan_model(data_agents)

    result, _ = await run_agent(
        name_postfix="Draft Query Plan",
        agent_configuration=coordinator.agent_configuration,
        prompt_name="Query Coordinator/Query Replan",
        prompt_parameters={
            "agent_registry": agent_registry,
            "old_query_plan": create_query_plan_prompt_context(query_plan),
            "query": query_plan.query,
            "reason": create_reason_prompt_context(reason),
        },
        output_type=QueryReplanModel,
    )

    query_plan = map_query_replan_model_to_query_plan(
        result, query_plan.query, answer_map
    )
    return query_plan

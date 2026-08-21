import json

from pydantic import BaseModel

import qcp_adaptive_termination.agent_registry as agent_registry
import qcp_adaptive_termination.hooks as hooks
from qcp_adaptive_termination.hooks import HookEvent
from qcp_adaptive_termination.llm import runner
from qcp_adaptive_termination.models import AgentRegistryType, AgentRole, DataAgent, ResultType, need_type, result_type
from qcp_adaptive_termination.models.execution import FailedExecution
from qcp_adaptive_termination.models.query_plan import (
    MultiAgentQueryPlanStep,
    QueryPlan,
)
from qcp_adaptive_termination.stages.planning import (
    create_draft_query_plan_model,
    map_query_plan_model_to_query_plan,
)


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
                    **(
                        {
                            "merge_task": step.merge_task,
                        }
                        if isinstance(step, MultiAgentQueryPlanStep)
                        else {}
                    ),
                    "dependencies": [
                        {
                            "step_id": dependency.step_id,
                            "dependency": dependency.dependency,
                        }
                        for dependency in step.dependencies
                    ],
                    "answer_preview": result_type.serialize(step.answer, context="preview")
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
        "raised_needs": [need_type.serialize(need, context="preview") for need in reason.needs],
        "failed_step_id": reason.step.id,
        "failed_step_query": reason.step.query,
        "failed_step_agent": reason.step.to_agent,
    }
    if reason.partial_answer is not None:
        reason_data["partial_answer"] = result_type.serialize(reason.partial_answer, context="preview")

    return json.dumps(reason_data, indent=2)


def create_query_replan_model(data_agents: list[DataAgent]) -> type[BaseModel]:
    return create_draft_query_plan_model(data_agents)


def map_query_replan_model_to_query_plan(
    query_replan_model: BaseModel, query: str, answer_map: dict[str, ResultType]
) -> QueryPlan:
    query_plan = map_query_plan_model_to_query_plan(query_replan_model, query)

    for step in query_plan.steps:
        if step.id in answer_map:
            step.answer = answer_map[step.id]

    return query_plan


async def replan_query_plan(
    data_agents: list[DataAgent],
    relevant_data_agents: list[DataAgent],
    query_plan: QueryPlan,
    reason: FailedExecution,
) -> QueryPlan:
    return await hooks.call_override(
        HookEvent.REPLAN_QUERY_PLAN_OVERRIDE,
        default=lambda: _replan_via_llm(data_agents, relevant_data_agents, query_plan, reason),
        data_agents=data_agents,
        relevant_data_agents=relevant_data_agents,
        query_plan=query_plan,
        reason=reason,
    )


async def _replan_via_llm(
    data_agents: list[DataAgent],
    relevant_data_agents: list[DataAgent],
    query_plan: QueryPlan,
    reason: FailedExecution,
) -> QueryPlan:
    registry_string = await agent_registry.build_registry_string(relevant_data_agents, AgentRegistryType.TECHNICAL)
    answer_map = create_answer_map_from_query_plan(query_plan)
    QueryReplanModel = create_query_replan_model(relevant_data_agents)

    result = await runner.run_llm(
        role=AgentRole.COORDINATOR_REPLANNING,
        prompt_parameters={
            "agent_registry": registry_string,
            "old_query_plan": create_query_plan_prompt_context(query_plan),
            "query": query_plan.query,
            "reason": create_reason_prompt_context(reason),
        },
        output_type=QueryReplanModel,
    )

    query_plan = map_query_replan_model_to_query_plan(result, query_plan.query, answer_map)
    return query_plan

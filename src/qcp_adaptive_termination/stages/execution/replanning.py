"""Replanning: revise the query plan after a failed execution."""

import json
import logging
from typing import Any

from agents.items import TResponseInputItem
from pydantic import BaseModel

import qcp_adaptive_termination.agent_registry as agent_registry
import qcp_adaptive_termination.hooks as hooks
from qcp_adaptive_termination.hooks import HookEvent
from qcp_adaptive_termination.models import AgentRegistryType, AgentRole, DataAgent, ResultType, need_type, result_type
from qcp_adaptive_termination.models.execution import FailedExecution
from qcp_adaptive_termination.models.query_plan import (
    MultiAgentQueryPlanStep,
    QueryPlan,
)
from qcp_adaptive_termination.models.stages import Stage
from qcp_adaptive_termination.observability import tracing
from qcp_adaptive_termination.stages.discovery import filter_relevant_agents
from qcp_adaptive_termination.stages.planning import (
    create_query_plan_model,
    map_query_plan_model_to_query_plan,
    run_plan_agent,
)


def create_answer_map_from_query_plan(query_plan: QueryPlan) -> dict[str, ResultType]:
    """Map each already-answered step id to its answer."""
    answer_map = {}
    for step in query_plan.steps:
        if step.answer is not None:
            answer_map[step.id] = step.answer
    return answer_map


def create_query_plan_prompt_context(query_plan: QueryPlan) -> str:
    """Serialize a query plan for the replanning LLM prompt."""
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
    """Serialize a failed execution for the replanning LLM prompt."""
    reason_data = {
        "raised_needs": [need_type.serialize(need, context="preview") for need in reason.needs],
        "failed_step_id": reason.step.id,
        "failed_step_query": reason.step.query,
        "failed_step_agent": reason.step.to_agent,
    }
    if reason.partial_answer is not None:
        reason_data["partial_answer"] = result_type.serialize(reason.partial_answer, context="preview")

    return json.dumps(reason_data, indent=2)


def map_query_replan_model_to_query_plan(
    query_replan_model: BaseModel, query: str, answer_map: dict[str, ResultType]
) -> QueryPlan:
    """Convert a replan model to a QueryPlan, preserving prior answers.

    Args:
        query_replan_model: the model output by the replanning LLM
        query: the user query
        answer_map: answers of steps already executed

    Returns:
        (QueryPlan): the parsed replanned query plan
    """
    query_plan = map_query_plan_model_to_query_plan(query_replan_model, query)

    for step in query_plan.steps:
        if step.id in answer_map:
            step.answer = answer_map[step.id]

    return query_plan


@hooks.hook(HookEvent.EXECUTION_REPLAN_DRAFT_QUERY_PLAN)
async def replan_query_plan(
    data_agents: list[DataAgent],
    relevant_data_agents: list[DataAgent],
    query_plan: QueryPlan,
    reason: FailedExecution,
    history: list[TResponseInputItem],
) -> QueryPlan:
    """Revise the query plan after a failed execution.

    Args:
        data_agents: all known data agents
        relevant_data_agents: the agents considered for the replan
        query_plan: the current query plan
        reason: the failed execution to recover from
        history: conversation history threaded through replans

    Returns:
        (QueryPlan): the revised query plan
    """
    registry_string = await agent_registry.build_registry_string(relevant_data_agents, AgentRegistryType.TECHNICAL)
    answer_map = create_answer_map_from_query_plan(query_plan)
    QueryReplanModel = create_query_plan_model(relevant_data_agents)

    prompt_parameters: dict[str, Any] = {
        "agent_registry": registry_string,
        "old_query_plan": create_query_plan_prompt_context(query_plan),
        "query": query_plan.query,
        "reason": create_reason_prompt_context(reason),
    }
    if history:
        prompt_parameters["is_continuation"] = True

    query_plan, continuation = await run_plan_agent(
        role=AgentRole.EXECUTION_COORDINATOR_REPLAN,
        prompt_parameters=prompt_parameters,
        output_type=QueryReplanModel,
        map_output=lambda out: map_query_replan_model_to_query_plan(out, query_plan.query, answer_map),
        error_kind="replanning",
        history=history,
    )

    history[:] = continuation
    return query_plan


@hooks.hook(HookEvent.EXECUTION_REPLAN, stage_context=True)
async def replan(
    attempt: int,
    current_data_agents: list[DataAgent],
    data_agents: list[DataAgent],
    query_plan: QueryPlan,
    failed_execution: FailedExecution,
    history: list[TResponseInputItem],
) -> tuple[list[DataAgent], QueryPlan]:
    """Replan: expand the agent set if needed and revise the query plan.

    Args:
        attempt: the current replan attempt index
        current_data_agents: the agents in the current attempt
        data_agents: all known data agents
        query_plan: the current query plan
        failed_execution: the failed execution to recover from
        history: conversation history threaded through replans

    Returns:
        (tuple[list[DataAgent], QueryPlan]): the updated agents and query plan
    """
    with tracing.own_span(
        f"{attempt} - {Stage.EXECUTION.label} (Replan)",
        input={
            "attempt": attempt,
            "current_data_agents": current_data_agents,
            "data_agents": data_agents,
            "query_plan": query_plan,
            "failed_execution": failed_execution,
        },
    ):
        updated_data_agents = current_data_agents

        if failed_execution.needs:
            current_agent_names = {agent.name for agent in current_data_agents}
            remaining_agents = [agent for agent in data_agents if agent.name not in current_agent_names]
            if remaining_agents:
                result = await filter_relevant_agents(
                    query=query_plan.query,
                    data_agents=remaining_agents,
                )
                updated_data_agents = current_data_agents + result.relevant_data_agents
            else:
                logging.warning("Replan triggered with needs but no remaining agents to expand into")

        new_query_plan = await replan_query_plan(
            data_agents=data_agents,
            relevant_data_agents=updated_data_agents,
            query_plan=query_plan,
            reason=failed_execution,
            history=history,
        )

        tracing.append_to_span_output(
            {
                "updated_data_agents": updated_data_agents,
                "new_query_plan": new_query_plan,
            }
        )

        return updated_data_agents, new_query_plan

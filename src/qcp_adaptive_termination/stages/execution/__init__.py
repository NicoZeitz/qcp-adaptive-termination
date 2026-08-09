"""
Execution stage: runs a query plan against selected data agents.

Drives the step-by-step execution loop, handles replanning on failure
(up to `MAX_REPLAN_ATTEMPTS`), and returns a `CollaborativeQueryAnsweringResult`.
"""

from typing import cast

from qcp_adaptive_termination.cache import CacheMode, cached
from qcp_adaptive_termination.models import (
    AgentRegistryType,
    DataAgent,
    ResultType,
    get_failed_result,
    serialize_need,
)
from qcp_adaptive_termination.models.execution import (
    ExecutionResult,
    FailedExecution,
    SuccessfulExecution,
)
from qcp_adaptive_termination.models.query_plan import QueryPlan
from qcp_adaptive_termination.models.stages import CollaborativeQueryAnsweringResult
from qcp_adaptive_termination.observability import tracing
from qcp_adaptive_termination.stages.discovery import (
    FilterType,
    filter_relevant_agents_internal,
)
from qcp_adaptive_termination.stages.execution.execute_query_plan_next_step import (
    execute_query_plan_next_step,
)
from qcp_adaptive_termination.stages.replanning import replan_query_plan

STAGE_NAME = "Stage 3: Execution"
REPLAN_STAGE_NAME = "Stage 4: Replan"


async def execute_query_plan_attempt(
    replan_attempt: int,
    query_plan: QueryPlan,
    current_data_agents: list[DataAgent],
) -> ExecutionResult:
    with tracing.own_span(
        f"{replan_attempt} - {STAGE_NAME}",
        input={
            "query_plan": query_plan,
            "current_data_agents": current_data_agents,
        },
    ):
        while True:
            execution_result = await execute_query_plan_next_step(
                current_data_agents,
                query_plan,
            )

            if isinstance(execution_result, SuccessfulExecution) and execution_result.execution_finished:
                return ExecutionResult(success=execution_result.success, result=execution_result.result)
            elif isinstance(execution_result, FailedExecution):
                return ExecutionResult(success=execution_result.success, failed_execution=execution_result)


async def replan(
    replan_attempt: int,
    current_data_agents: list[DataAgent],
    data_agents: list[DataAgent],
    query_plan: QueryPlan,
    failed_execution: FailedExecution,
) -> tuple[list[DataAgent], QueryPlan]:
    with tracing.own_span(
        f"{replan_attempt} - {REPLAN_STAGE_NAME}",
        input={
            "replan_attempt": replan_attempt,
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
            needs_context = "\n".join(f"- {serialize_need(need)}" for need in failed_execution.needs)
            if remaining_agents:
                result = await filter_relevant_agents_internal(
                    query=query_plan.query,
                    data_agents=remaining_agents,
                    registry_type=AgentRegistryType.NATURAL_LANGUAGE,
                    filter_type=FilterType.REASONING,
                    additional_context=needs_context,
                )
                updated_data_agents = current_data_agents + result.relevant_data_agents

        new_query_plan = await replan_query_plan(
            data_agents=data_agents,
            relevant_data_agents=updated_data_agents,
            query_plan=query_plan,
            reason=failed_execution,
        )

        tracing.append_to_span_output(
            {
                "updated_data_agents": updated_data_agents,
                "new_query_plan": new_query_plan,
            }
        )

        return updated_data_agents, new_query_plan


@cached(name_prefix=STAGE_NAME, mode=CacheMode.WRITE_ONLY)
async def execute_query_plan(
    query_plan: QueryPlan,
    relevant_data_agents: list[DataAgent],
    data_agents: list[DataAgent],
) -> CollaborativeQueryAnsweringResult:
    MAX_REPLAN_ATTEMPTS = 2

    for replan_attempt in range(MAX_REPLAN_ATTEMPTS + 1):
        execution_result = await execute_query_plan_attempt(
            replan_attempt=replan_attempt,
            query_plan=query_plan,
            current_data_agents=relevant_data_agents,
        )

        if execution_result.success:
            return CollaborativeQueryAnsweringResult(
                result=cast(ResultType, execution_result.result),
            )

        if replan_attempt >= MAX_REPLAN_ATTEMPTS:
            return CollaborativeQueryAnsweringResult(
                result=get_failed_result(),
            )

        relevant_data_agents, query_plan = await replan(
            replan_attempt=replan_attempt,
            current_data_agents=relevant_data_agents,
            data_agents=data_agents,
            query_plan=query_plan,
            failed_execution=cast(FailedExecution, execution_result.failed_execution),
        )

    return CollaborativeQueryAnsweringResult(
        result=get_failed_result(),
    )

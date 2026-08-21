"""
Execution stage: runs a query plan against selected data agents.

Drives the step-by-step execution loop, handles replanning on failure
(up to `MAX_REPLAN_ATTEMPTS`), and returns a `CollaborativeQueryAnsweringResult`.
"""

import logging
from typing import cast

from qcp_adaptive_termination import hooks
from qcp_adaptive_termination.hooks import HookEvent
from qcp_adaptive_termination.models import (
    AgentRegistryType,
    CacheMode,
    DataAgent,
    ResultType,
    need_type,
    result_type,
)
from qcp_adaptive_termination.models.execution import (
    ExecutionResult,
    FailedExecution,
    SuccessfulExecution,
)
from qcp_adaptive_termination.models.query_plan import QueryPlan
from qcp_adaptive_termination.models.stages import CollaborativeQueryAnsweringResult
from qcp_adaptive_termination.observability import cache, tracing
from qcp_adaptive_termination.stages.discovery import (
    filter_relevant_agents_internal,
)
from qcp_adaptive_termination.stages.execution.execute_query_plan_next_step import (
    execute_query_plan_next_step,
)
from qcp_adaptive_termination.stages.replanning import replan_query_plan

STAGE_NAME = "Stage 3: Execution"
REPLAN_STAGE_NAME = "Stage 4: Replan"


async def execute_query_plan_attempt(
    attempt: int,
    query_plan: QueryPlan,
    current_data_agents: list[DataAgent],
) -> ExecutionResult:
    with (
        hooks.stage_context(),
        tracing.own_span(
            f"{attempt} - {STAGE_NAME}",
            input={
                "query_plan": query_plan,
                "current_data_agents": current_data_agents,
            },
        ),
    ):
        hooks.emit(
            HookEvent.EXECUTE_QUERY_PLAN_START,
            attempt=attempt,
            query_plan=query_plan,
            current_data_agents=current_data_agents,
        )
        while True:
            execution_result = await execute_query_plan_next_step(
                current_data_agents,
                query_plan,
            )

            if isinstance(execution_result, SuccessfulExecution) and execution_result.execution_finished:
                result = ExecutionResult(success=execution_result.success, result=execution_result.result)
                break
            if isinstance(execution_result, FailedExecution):
                result = ExecutionResult(success=execution_result.success, failed_execution=execution_result)
                break

        hooks.emit(HookEvent.EXECUTE_QUERY_PLAN_END, attempt=attempt, result=result)
        return result


async def replan(
    attempt: int,
    current_data_agents: list[DataAgent],
    data_agents: list[DataAgent],
    query_plan: QueryPlan,
    failed_execution: FailedExecution,
) -> tuple[list[DataAgent], QueryPlan]:
    with (
        hooks.stage_context(),
        tracing.own_span(
            f"{attempt} - {REPLAN_STAGE_NAME}",
            input={
                "attempt": attempt,
                "current_data_agents": current_data_agents,
                "data_agents": data_agents,
                "query_plan": query_plan,
                "failed_execution": failed_execution,
            },
        ),
    ):
        hooks.emit(
            HookEvent.REPLAN_START,
            attempt=attempt,
            current_data_agents=current_data_agents,
            data_agents=data_agents,
            query_plan=query_plan,
            failed_execution=failed_execution,
        )
        updated_data_agents = current_data_agents

        if failed_execution.needs:
            current_agent_names = {agent.name for agent in current_data_agents}
            remaining_agents = [agent for agent in data_agents if agent.name not in current_agent_names]
            needs_context = "\n".join(f"- {need_type.serialize(need)}" for need in failed_execution.needs)
            if remaining_agents:
                result = await filter_relevant_agents_internal(
                    query=query_plan.query,
                    data_agents=remaining_agents,
                    registry_type=AgentRegistryType.NATURAL_LANGUAGE,
                    additional_context=needs_context,
                    pre_filter=False,
                )
                updated_data_agents = current_data_agents + result.relevant_data_agents
            else:
                logging.warning("Replan triggered with needs but no remaining agents to expand into")

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

        hooks.emit(
            HookEvent.REPLAN_END,
            attempt=attempt,
            updated_data_agents=updated_data_agents,
            new_query_plan=new_query_plan,
        )
        return updated_data_agents, new_query_plan


@cache.cached(name_prefix=STAGE_NAME, mode=CacheMode.WRITE_ONLY)
async def execute_query_plan(
    query_plan: QueryPlan,
    relevant_data_agents: list[DataAgent],
    data_agents: list[DataAgent],
) -> CollaborativeQueryAnsweringResult:
    MAX_REPLAN_ATTEMPTS = 2

    with hooks.stage_context():
        hooks.emit(
            HookEvent.EXECUTE_QUERY_PLAN_LOOP_START,
            query_plan=query_plan,
            current_data_agents=relevant_data_agents,
        )

        final_result: CollaborativeQueryAnsweringResult | None = None
        for attempt in range(MAX_REPLAN_ATTEMPTS + 1):
            execution_result = await execute_query_plan_attempt(
                attempt=attempt,
                query_plan=query_plan,
                current_data_agents=relevant_data_agents,
            )

            if execution_result.success:
                final_result = CollaborativeQueryAnsweringResult(
                    result=cast(ResultType, execution_result.result),
                )
                break

            if attempt >= MAX_REPLAN_ATTEMPTS:
                final_result = CollaborativeQueryAnsweringResult(
                    result=result_type.failed_result(),
                )
                break

            relevant_data_agents, query_plan = await replan(
                attempt=attempt,
                current_data_agents=relevant_data_agents,
                data_agents=data_agents,
                query_plan=query_plan,
                failed_execution=cast(FailedExecution, execution_result.failed_execution),
            )

        if final_result is None:
            final_result = CollaborativeQueryAnsweringResult(
                result=result_type.failed_result(),
            )

        hooks.emit(HookEvent.EXECUTE_QUERY_PLAN_LOOP_END, result=final_result)
        return final_result

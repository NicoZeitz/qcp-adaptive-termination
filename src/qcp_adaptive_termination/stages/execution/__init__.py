"""Execution stage: run a query plan against the selected data agents."""

from typing import cast

from agents.items import TResponseInputItem

from qcp_adaptive_termination import hooks
from qcp_adaptive_termination.hooks import HookEvent
from qcp_adaptive_termination.models import (
    CacheMode,
    DataAgent,
    ResultType,
    StepBudget,
    result_type,
)
from qcp_adaptive_termination.models.execution import (
    ExecutionResult,
    FailedExecution,
    SuccessfulExecution,
)
from qcp_adaptive_termination.models.query_plan import QueryPlan
from qcp_adaptive_termination.models.stages import CollaborativeQueryAnsweringResult, Stage
from qcp_adaptive_termination.observability import cache, context, tracing
from qcp_adaptive_termination.stages.execution.execute_query_plan_next_nodes import (
    execute_query_plan_next_nodes,
)
from qcp_adaptive_termination.stages.execution.replanning import replan
from qcp_adaptive_termination.stages.execution.uncertainty_budget import (
    compute_step_budgets,
)


@hooks.hook(HookEvent.EXECUTION_EXECUTE_QUERY_PLAN, stage_context=True)
async def execute_query_plan_attempt(
    attempt: int,
    query_plan: QueryPlan,
    current_data_agents: list[DataAgent],
    step_budgets: dict[str, StepBudget],
) -> ExecutionResult:
    """Run one execution attempt over the query plan's nodes.

    Args:
        attempt: the current replan attempt index
        query_plan: the query plan to execute
        current_data_agents: the agents available for this attempt
        step_budgets: per-step completeness budgets

    Returns:
        (ExecutionResult): the outcome of the attempt
    """
    with tracing.own_span(
        f"{attempt} - {Stage.EXECUTION.label}",
        input={
            "query_plan": query_plan,
            "current_data_agents": current_data_agents,
        },
    ):
        while True:
            execution_result = await execute_query_plan_next_nodes(
                current_data_agents,
                query_plan,
                step_budgets,
            )

            if isinstance(execution_result, SuccessfulExecution) and execution_result.execution_finished:
                return ExecutionResult(
                    success=execution_result.success,
                    result=execution_result.result,
                    step_uncertainties=execution_result.step_uncertainties,
                )
            if isinstance(execution_result, FailedExecution):
                return ExecutionResult(success=execution_result.success, failed_execution=execution_result)


@hooks.hook(HookEvent.EXECUTION_STAGE, stage_context=True)
@cache.cached(mode=CacheMode.WRITE_ONLY)
async def execute_query_plan(
    query_plan: QueryPlan,
    relevant_data_agents: list[DataAgent],
    data_agents: list[DataAgent],
) -> CollaborativeQueryAnsweringResult:
    """Execute a query plan, replanning on failure up to the configured limit.

    Args:
        query_plan: the query plan to execute
        relevant_data_agents: the agents selected for this plan
        data_agents: all known data agents

    Returns:
        (CollaborativeQueryAnsweringResult): the execution result
    """
    config = context.get_config()
    max_replan_attempts = config.execution.replanning_max_attempts

    attempt = 0
    # Conversation history threaded through replan attempts, extended with each replan's
    # continuation so the next replan continues from there.
    replan_history: list[TResponseInputItem] = []
    while True:
        step_budgets = compute_step_budgets(
            query_plan,
            config.completeness.target_confidence,
            config.completeness.target_apportionment,
        )

        print(
            f"[EXEC] attempt {attempt}: executing {len(query_plan.steps)} step(s) "
            f"across {len(relevant_data_agents)} relevant agent(s)",
            flush=True,
        )
        execution_result = await execute_query_plan_attempt(
            attempt=attempt,
            query_plan=query_plan,
            current_data_agents=relevant_data_agents,
            step_budgets=step_budgets,
        )

        if execution_result.success:
            print("[EXEC] plan executed successfully", flush=True)
            return CollaborativeQueryAnsweringResult(
                result=cast(ResultType, execution_result.result),
                query_plan=query_plan,
                step_uncertainties=execution_result.step_uncertainties,
            )

        if attempt >= max_replan_attempts:
            print("[EXEC] max replan attempts reached — returning failed result", flush=True)
            return CollaborativeQueryAnsweringResult(
                result=result_type.failed_result(),
                query_plan=query_plan,
            )

        relevant_data_agents, query_plan = await replan(
            attempt=attempt,
            current_data_agents=relevant_data_agents,
            data_agents=data_agents,
            query_plan=query_plan,
            failed_execution=cast(FailedExecution, execution_result.failed_execution),
            history=replan_history,
        )
        attempt += 1

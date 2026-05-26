from typing import cast

from qtypes.collaborative_query_answering_result import (
    CollaborativeQueryAnsweringResult,
)
from qtypes.data_agent import DataAgent
from qtypes.execution_result import ExecutionResult
from qtypes.query_plan import QueryPlan
from qtypes.query_plan_step_execution_result import FailedExecution, SuccessfulExecution
from qtypes.result import ResultType, get_failed_result
from stages.execute_query_plan.add_agents_for_needs import add_agents_for_needs
from stages.execute_query_plan.replan_query_plan import replan_query_plan
from stages.execute_query_plan.execute_query_plan_next_step import (
    execute_query_plan_next_step,
)
from stages.utils.cache import cached
from stages.utils.span import append_to_span_output, own_custom_span


async def execute_query_plan_attempt(
    replan_attempt: int,
    query_plan: QueryPlan,
    current_data_agents: list[DataAgent],
) -> ExecutionResult:
    with own_custom_span(
        f"{replan_attempt} - Execute Query Plan",
        input={
            "query_plan": query_plan,
            "current_data_agents": current_data_agents,
        },
    ):
        while True:
            execution_result = await execute_query_plan_next_step(
                current_data_agents, query_plan
            )

            if (
                isinstance(execution_result, SuccessfulExecution)
                and execution_result.execution_finished
            ):
                return ExecutionResult(
                    success=execution_result.success, result=execution_result.result
                )
            elif isinstance(execution_result, FailedExecution):
                return ExecutionResult(
                    success=execution_result.success, failed_execution=execution_result
                )


async def replan(
    replan_attempt: int,
    current_data_agents: list[DataAgent],
    data_agents: list[DataAgent],
    query_plan: QueryPlan,
    failed_execution: FailedExecution,
) -> tuple[list[DataAgent], QueryPlan]:
    with own_custom_span(
        f"{replan_attempt} - Stage 4: Replan",
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
            updated_data_agents = await add_agents_for_needs(
                data_agents=data_agents,
                current_data_agents=current_data_agents,
                needs=failed_execution.needs,
                query=query_plan.query,
            )

        new_query_plan = await replan_query_plan(
            data_agents=data_agents,
            relevant_data_agents=updated_data_agents,
            query_plan=query_plan,
            reason=failed_execution,
        )

        append_to_span_output(
            {
                "updated_data_agents": updated_data_agents,
                "new_query_plan": new_query_plan,
            }
        )

        return updated_data_agents, new_query_plan


@cached(name_prefix="Stage 3: Execute Query Plan")
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
                result=cast(ResultType, execution_result.result)
            )

        if replan_attempt >= MAX_REPLAN_ATTEMPTS:
            return CollaborativeQueryAnsweringResult(result=get_failed_result())

        relevant_data_agents, query_plan = await replan(
            replan_attempt=replan_attempt,
            current_data_agents=relevant_data_agents,
            data_agents=data_agents,
            query_plan=query_plan,
            failed_execution=cast(FailedExecution, execution_result.failed_execution),
        )

    return CollaborativeQueryAnsweringResult(result=get_failed_result())

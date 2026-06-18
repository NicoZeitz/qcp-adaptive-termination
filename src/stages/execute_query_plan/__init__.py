from typing import cast

from qtypes.collaborative_query_answering_result import (
    CollaborativeQueryAnsweringResult,
)
from qtypes.data_agent import DataAgent
from qtypes.execution_result import ExecutionResult
from qtypes.multi_agent_iteration_record import MultiAgentIterationRecord
from qtypes.need import serialize_need
from qtypes.query_plan import QueryPlan
from qtypes.query_plan_step_execution_result import FailedExecution, SuccessfulExecution
from qtypes.result import ResultType, get_failed_result
from stages.common.data_agent.agent_registry import AgentRegistryType
from stages.execute_query_plan.execute_query_plan_next_step import (
    execute_query_plan_next_step,
)
from stages.filter_relevant_agents import FilterType, filter_relevant_agents_internal
from stages.replan_query_plan import replan_query_plan
from utils.cache import cached
from utils.span import append_to_span_output, own_custom_span

# Stored separately from the cached result to avoid serialization issues with DataFrames.
last_iteration_records: list[MultiAgentIterationRecord] = []


async def execute_query_plan_attempt(
    replan_attempt: int,
    query_plan: QueryPlan,
    current_data_agents: list[DataAgent],
    iteration_records: list[MultiAgentIterationRecord],
    continue_after_final_result: bool = False,
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
                current_data_agents,
                query_plan,
                iteration_records,
                continue_after_final_result,
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
            current_agent_names = {agent.name for agent in current_data_agents}
            remaining_agents = [
                agent for agent in data_agents if agent.name not in current_agent_names
            ]
            needs_context = "\n".join(
                f"- {serialize_need(need)}" for need in failed_execution.needs
            )
            if remaining_agents:
                result = await filter_relevant_agents_internal(
                    query=query_plan.query,
                    data_agents=remaining_agents,
                    agent_registry_type=AgentRegistryType.NATURAL_LANGUAGE,
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

        append_to_span_output(
            {
                "updated_data_agents": updated_data_agents,
                "new_query_plan": new_query_plan,
            }
        )

        return updated_data_agents, new_query_plan


@cached(name_prefix="Stage 3: Execute Query Plan", mode="WRITE_ONLY")
async def execute_query_plan(
    query_plan: QueryPlan,
    relevant_data_agents: list[DataAgent],
    data_agents: list[DataAgent],
    continue_after_final_result: bool = False,
) -> CollaborativeQueryAnsweringResult:
    global last_iteration_records
    MAX_REPLAN_ATTEMPTS = 2
    iteration_records: list[MultiAgentIterationRecord] = []

    for replan_attempt in range(MAX_REPLAN_ATTEMPTS + 1):
        execution_result = await execute_query_plan_attempt(
            replan_attempt=replan_attempt,
            query_plan=query_plan,
            current_data_agents=relevant_data_agents,
            iteration_records=iteration_records,
            continue_after_final_result=continue_after_final_result,
        )

        if execution_result.success:
            last_iteration_records = iteration_records
            return CollaborativeQueryAnsweringResult(
                result=cast(ResultType, execution_result.result),
            )

        if replan_attempt >= MAX_REPLAN_ATTEMPTS:
            last_iteration_records = iteration_records
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

    last_iteration_records = iteration_records
    return CollaborativeQueryAnsweringResult(
        result=get_failed_result(),
    )

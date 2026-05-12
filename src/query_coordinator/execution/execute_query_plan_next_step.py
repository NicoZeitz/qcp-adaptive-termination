from typing import cast

from agents import AgentsException, MaxTurnsExceeded

from data_agent.data_agent import DataAgent
from qtypes.AdditionalInformation import AdditionalInformation
from qtypes.query_plan.need import exception_to_need
from query_coordinator.execution.answer_step_coordinator import answer_step_coordinator
from query_coordinator.execution.answer_step_data_agent import answer_step_data_agent
from query_coordinator.query_coordinator import QueryCoordinator
from qtypes.partial_query_answer import PartialQueryAnswer
from qtypes.query_plan.query_plan import QueryPlan, QueryPlanStep
from qtypes.query_plan.query_plan_step_execution_result import (
    FailedExecution,
    QueryPlanStepExecutionResult,
    SuccessfulExecution,
)
from qtypes.query_plan.result import ResultType
from utils.span import append_to_span_input, append_to_span_output, own_custom_span


async def execute_query_plan_step(
    coordinator: QueryCoordinator,
    data_agents: list[DataAgent],
    query_plan_step: QueryPlanStep,
) -> PartialQueryAnswer:
    additional_information: list[AdditionalInformation] = [
        AdditionalInformation(
            dependency=dependency.dependency, data=dependency.step.answer
        )
        for dependency in query_plan_step.dependencies
        if dependency.step.answer is not None
    ]

    with own_custom_span(f"Execute Step: {query_plan_step.id}") as span:
        append_to_span_input(
            span,
            {
                "step": query_plan_step.to_span_data(),
                "additional_information": [
                    information.to_span_data() for information in additional_information
                ],
            },
        )

        if query_plan_step.to_agent == "Coordinator":
            result = await answer_step_coordinator(
                coordinator, query_plan_step, additional_information
            )
        else:
            result = await answer_step_data_agent(
                data_agents, query_plan_step, additional_information
            )

        append_to_span_output(span, {"partial_answer": result.to_span_data()})
        return result


async def execute_query_plan_next_step(
    coordinator: QueryCoordinator, data_agents: list[DataAgent], query_plan: QueryPlan
) -> QueryPlanStepExecutionResult:
    step_to_execute = None
    for step in query_plan.steps:
        if step.answer is not None:
            continue  # Skip steps that have already been executed

        # Assume all dependencies are satisfied
        step_to_execute = step
        break

    if step_to_execute is None:
        return SuccessfulExecution(
            step=query_plan.steps[-1],
            result=cast(ResultType, query_plan.steps[-1].answer),
            execution_finished=True,
        )

    try:
        result = await execute_query_plan_step(coordinator, data_agents, step_to_execute)

        step_to_execute.answer = (
            result.answer
        )  # Update the query plan with the (partial) answer (if any)

        if result.has_answer and not result.has_needs:
            return SuccessfulExecution(
                step=step_to_execute,
                result=cast(ResultType, result.answer),
                execution_finished=all(
                    step.answer is not None for step in query_plan.steps
                ),
            )

        return FailedExecution(
            step=step_to_execute,
            needs=result.needs,
            partial_answer=result.answer,
            has_partial_answer=result.has_answer,
        )
    except (MaxTurnsExceeded, AgentsException) as e:
        result = PartialQueryAnswer(has_answer=False, needs=[exception_to_need(e)])
        return FailedExecution(
            step=step_to_execute,
            needs=result.needs,
            partial_answer=result.answer,
            has_partial_answer=result.has_answer,
        )

from typing import cast

from agents import AgentsException, MaxTurnsExceeded

from qtypes.additional_information import AdditionalInformation
from qtypes.data_agent import DataAgent
from qtypes.need import exception_to_need
from qtypes.partial_query_answer import PartialQueryAnswer
from qtypes.query_plan import QueryPlan
from qtypes.query_plan_step import QueryPlanStep
from qtypes.query_plan_step_execution_result import (
    FailedExecution,
    QueryPlanStepExecutionResult,
    SuccessfulExecution,
)
from qtypes.result import ResultType
from stages.execute_query_plan.answer_step_data_agent import answer_step_data_agent
from stages.execute_query_plan.answer_step_coordinator import answer_step_coordinator
from utils.span import append_to_span_output, own_custom_span


async def execute_query_plan_next_step(
    data_agents: list[DataAgent], query_plan: QueryPlan
) -> QueryPlanStepExecutionResult:
    step_to_execute = None
    step_id_to_answer: dict[str, ResultType] = {}
    for step in query_plan.steps:
        if step.answer is not None:
            step_id_to_answer[step.id] = step.answer
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
        additional_information = [
            AdditionalInformation(
                dependency=dependency.dependency,
                data=step_id_to_answer[dependency.step_id],
            )
            for dependency in step_to_execute.dependencies
            if dependency.step_id in step_id_to_answer
        ]

        result = await execute_query_plan_step(
            data_agents, step_to_execute, additional_information
        )

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
        result = PartialQueryAnswer()
        result.add_needs({exception_to_need(e)})
        return FailedExecution(
            step=step_to_execute,
            needs=result.needs,
            partial_answer=result.answer,
            has_partial_answer=result.has_answer,
        )


async def execute_query_plan_step(
    data_agents: list[DataAgent],
    query_plan_step: QueryPlanStep,
    additional_information: list[AdditionalInformation],
) -> PartialQueryAnswer:
    with own_custom_span(
        f"Execute Step: {query_plan_step.id}",
        input={
            "step": query_plan_step,
            "additional_information": additional_information,
        },
    ):
        if query_plan_step.to_agent == "Coordinator":
            result = await answer_step_coordinator(
                query_plan_step, additional_information
            )
        else:
            result = await answer_step_data_agent(
                data_agents,
                query_plan_step.query,
                query_plan_step.to_agent,
                additional_information,
            )

        append_to_span_output({"partial_answer": result})
        return result

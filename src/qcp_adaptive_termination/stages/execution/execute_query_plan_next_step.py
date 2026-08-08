from typing import cast

from agents import AgentsException, MaxTurnsExceeded

from qcp_adaptive_termination.models import (
    DataAgent,
    ResultType,
    exception_to_need,
)
from qcp_adaptive_termination.models.execution import (
    AdditionalInformation,
    FailedExecution,
    PartialQueryAnswer,
    QueryPlanStepExecutionResult,
    SuccessfulExecution,
)
from qcp_adaptive_termination.models.query_plan import (
    MultiAgentQueryPlanStep,
    QueryPlan,
    QueryPlanStep,
    SingleAgentQueryPlanStep,
)
from qcp_adaptive_termination.observability import tracing
from qcp_adaptive_termination.stages.execution.coordinator import (
    answer_step_coordinator,
)
from qcp_adaptive_termination.stages.execution.data_agent import answer_step_data_agent
from qcp_adaptive_termination.stages.execution.multi_agent import (
    answer_step_multiple_agents,
)


async def execute_query_plan_next_step(
    data_agents: list[DataAgent],
    query_plan: QueryPlan,
) -> QueryPlanStepExecutionResult:
    step_to_execute = None
    step_id_to_answer: dict[str, ResultType] = {}
    for step in query_plan.steps:
        if step.answer is not None:
            step_id_to_answer[step.id] = step.answer
            continue

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
                step_id=dependency.step_id,
                dependency=dependency.dependency,
                data=step_id_to_answer[dependency.step_id],
            )
            for dependency in step_to_execute.dependencies
            if dependency.step_id in step_id_to_answer
        ]

        result = await execute_query_plan_step(
            data_agents,
            step_to_execute,
            additional_information,
        )

        step_to_execute.answer = result.answer

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
    with tracing.own_span(
        f"Execute Step: {query_plan_step.id}",
        input={
            "step": query_plan_step,
            "additional_information": additional_information,
        },
    ):
        if isinstance(query_plan_step, SingleAgentQueryPlanStep):
            result = await execute_query_plan_step_for_single_agent(
                data_agents,
                query_plan_step,
                query_plan_step.to_agent,
                additional_information,
            )
        else:
            result = await answer_step_multiple_agents(
                data_agents=data_agents,
                query_plan_step=cast(MultiAgentQueryPlanStep, query_plan_step),
                additional_information=additional_information,
            )

        tracing.append_to_span_output({"partial_answer": result})
        return result


async def execute_query_plan_step_for_single_agent(
    data_agents: list[DataAgent],
    query_plan_step: QueryPlanStep,
    to_agent: str,
    additional_information: list[AdditionalInformation],
) -> PartialQueryAnswer:
    if to_agent == "Coordinator":
        return await answer_step_coordinator(query_plan_step, additional_information)

    return await answer_step_data_agent(
        data_agents,
        query_plan_step.query,
        to_agent,
        additional_information,
    )

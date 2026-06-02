import asyncio
from typing import cast

from agents import AgentsException, MaxTurnsExceeded
import pandas as pd

from qtypes.additional_information import AdditionalInformation
from qtypes.data_agent import DataAgent
from qtypes.need import exception_to_need
from qtypes.partial_query_answer import PartialQueryAnswer
from qtypes.query_plan import QueryPlan
from qtypes.query_plan_step import (
    MultiAgentQueryPlanStep,
    QueryPlanStep,
    SingleAgentQueryPlanStep,
)
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
                step_id=dependency.step_id,
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
        if isinstance(query_plan_step, SingleAgentQueryPlanStep):
            result = await execute_query_plan_step_for_single_agent(
                data_agents,
                query_plan_step,
                query_plan_step.to_agent,
                additional_information,
            )
        else:
            result = await execute_query_plan_step_for_multiple_agents(
                data_agents,
                cast(MultiAgentQueryPlanStep, query_plan_step),
                additional_information,
            )

        append_to_span_output({"partial_answer": result})
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


def create_merged_raw_answers(
    agent_answers: list[tuple[str, ResultType]],
) -> ResultType:
    if not agent_answers:
        return pd.DataFrame()

    merged_results: list[pd.DataFrame] = []
    for agent_name, answer in agent_answers:
        answer_with_source = answer.copy()
        answer_with_source["source_agent"] = agent_name
        merged_results.append(answer_with_source)

    return pd.concat(merged_results, ignore_index=True)


def create_multi_agent_additional_information(
    query_plan_step: MultiAgentQueryPlanStep,
    additional_information: list[AdditionalInformation],
    agent_answers: list[tuple[str, ResultType]],
) -> list[AdditionalInformation]:
    return [
        *additional_information,
        *[
            AdditionalInformation(
                step_id=f"{query_plan_step.id}:{agent_name}",
                dependency=f"Result from agent {agent_name}",
                data=answer,
            )
            for agent_name, answer in agent_answers
        ],
    ]


async def execute_query_plan_step_for_multiple_agents(
    data_agents: list[DataAgent],
    query_plan_step: MultiAgentQueryPlanStep,
    additional_information: list[AdditionalInformation],
) -> PartialQueryAnswer:
    local_results = PartialQueryAnswer()
    agent_answers: list[tuple[str, ResultType]] = []

    agent_results = await asyncio.gather(
        *[
            execute_query_plan_step_for_single_agent(
                data_agents,
                query_plan_step,
                target_agent,
                additional_information,
            )
            for target_agent in query_plan_step.to_agent
        ]
    )

    for target_agent, agent_result in zip(query_plan_step.to_agent, agent_results):
        if agent_result.has_answer and agent_result.answer is not None:
            agent_answers.append((target_agent, agent_result.answer))

        if agent_result.has_needs:
            local_results.add_needs(agent_result.needs)

    if local_results.has_needs:
        if agent_answers:
            local_results.set_answer(create_merged_raw_answers(agent_answers))
        return local_results

    merge_step = SingleAgentQueryPlanStep(
        id=f"{query_plan_step.id}:merge",
        query=query_plan_step.merge_task,
        to_agent="Coordinator",
        dependencies=[],
        answer=None,
    )

    merged_result = await answer_step_coordinator(
        merge_step,
        create_multi_agent_additional_information(
            query_plan_step, additional_information, agent_answers
        ),
    )

    return merged_result

"""Dependency-based DAG scheduling of query plan steps."""

import asyncio
from typing import cast

from agents import AgentsException, MaxTurnsExceeded

import qcp_adaptive_termination.hooks as hooks
from qcp_adaptive_termination.hooks import HookEvent
from qcp_adaptive_termination.models import DataAgent, ResultType, StepBudget, StepUncertainty, need_type
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
from qcp_adaptive_termination.observability import context, tracing
from qcp_adaptive_termination.stages.execution.coordinator import (
    answer_step_coordinator,
)
from qcp_adaptive_termination.stages.execution.data_agent import answer_step_data_agent
from qcp_adaptive_termination.stages.execution.multi_agent import (
    answer_step_multiple_agents,
)


def _build_additional_information(
    query_plan_step: QueryPlanStep,
    step_id_to_answer: dict[str, ResultType],
) -> list[AdditionalInformation]:
    """Collect the satisfying dependency answers for a step."""
    return [
        AdditionalInformation(
            step_id=dependency.step_id,
            dependency=dependency.dependency,
            data=step_id_to_answer[dependency.step_id],
        )
        for dependency in query_plan_step.dependencies
        if dependency.step_id in step_id_to_answer
    ]


async def execute_query_plan_next_nodes(
    data_agents: list[DataAgent],
    query_plan: QueryPlan,
    step_budgets: dict[str, StepBudget],
) -> QueryPlanStepExecutionResult:
    """Execute all ready steps in parallel until the plan finishes or fails.

    Args:
        data_agents: the available data agents
        query_plan: the query plan to execute
        step_budgets: per-step completeness budgets

    Returns:
        (QueryPlanStepExecutionResult): the execution outcome
    """
    step_id_to_answer: dict[str, ResultType] = {}
    step_uncertainties: dict[str, StepUncertainty] = {}
    for step in query_plan.steps:
        if step.answer is not None:
            step_id_to_answer[step.id] = step.answer

    pending = [s for s in query_plan.steps if s.id not in step_id_to_answer]
    if not pending:
        if not query_plan.steps:
            raise ValueError("Query plan has no steps")
        return SuccessfulExecution(
            step=query_plan.steps[-1],
            result=cast(ResultType, query_plan.steps[-1].answer),
            execution_finished=True,
        )

    try:
        while pending:
            ready = [s for s in pending if all(d.step_id in step_id_to_answer for d in s.dependencies)]
            if not ready:
                step = pending[0]
                return FailedExecution(
                    step=step,
                    needs=[need_type.from_exception(ValueError("Unresolvable step dependency"))],
                    partial_answer=None,
                    has_partial_answer=False,
                )

            results = await asyncio.gather(
                *[
                    execute_query_plan_step(
                        data_agents, step, _build_additional_information(step, step_id_to_answer), step_budgets
                    )
                    for step in ready
                ],
                return_exceptions=True,
            )

            for step, step_result in zip(ready, results, strict=True):
                if isinstance(step_result, BaseException):
                    if isinstance(step_result, (MaxTurnsExceeded, AgentsException)):
                        partial = PartialQueryAnswer()
                        partial.add_needs({need_type.from_exception(step_result)})
                        print(f"[STEP] {step.id} failed with exception: {step_result!r}", flush=True)
                        return FailedExecution(
                            step=step,
                            needs=partial.needs,
                            partial_answer=partial.answer,
                            has_partial_answer=partial.has_answer,
                        )
                    raise step_result

                result, uncertainty = step_result
                if result.has_answer and not result.has_needs:
                    step_id_to_answer[step.id] = cast(ResultType, result.answer)
                    if uncertainty is not None:
                        step_uncertainties[step.id] = uncertainty
                    print(
                        f"[STEP] {step.id} completed: answer={'yes' if result.has_answer else 'no'} "
                        f"needs={list(result.needs)} uncertainty={'yes' if uncertainty is not None else 'no'}",
                        flush=True,
                    )
                else:
                    print(
                        f"[STEP] {step.id} incomplete: needs={result.needs} has_answer={result.has_answer}",
                        flush=True,
                    )
                    return FailedExecution(
                        step=step,
                        needs=result.needs,
                        partial_answer=result.answer,
                        has_partial_answer=result.has_answer,
                    )

            pending = [s for s in pending if s.id not in step_id_to_answer]

        last_step = query_plan.steps[-1]
        return SuccessfulExecution(
            step=last_step,
            result=step_id_to_answer[last_step.id],
            execution_finished=True,
            step_uncertainties=step_uncertainties,
        )
    except (MaxTurnsExceeded, AgentsException) as e:
        result = PartialQueryAnswer()
        result.add_needs({need_type.from_exception(e)})
        return FailedExecution(
            step=pending[0] if pending else query_plan.steps[-1],
            needs=result.needs,
            partial_answer=result.answer,
            has_partial_answer=result.has_answer,
        )


@hooks.hook(HookEvent.EXECUTION_EXECUTE_QUERY_PLAN_EXECUTE_STEP)
async def execute_query_plan_step(
    data_agents: list[DataAgent],
    query_plan_step: QueryPlanStep,
    additional_information: list[AdditionalInformation],
    step_budgets: dict[str, StepBudget],
) -> tuple[PartialQueryAnswer, StepUncertainty | None]:
    """Execute one query plan step and return its answer and uncertainty.

    Args:
        data_agents: the available data agents
        query_plan_step: the step to execute
        additional_information: the satisfying dependency answers
        step_budgets: per-step completeness budgets

    Returns:
        (tuple[PartialQueryAnswer, StepUncertainty | None]): the step answer and its uncertainty
    """
    with (
        context.step_context(query_plan_step.id),
        tracing.own_span(
            f"Execute Step: {query_plan_step.id}",
            input={
                "step": query_plan_step,
                "additional_information": additional_information,
            },
        ),
    ):
        uncertainty: StepUncertainty | None = None
        if isinstance(query_plan_step, SingleAgentQueryPlanStep) and query_plan_step.to_agent == "Coordinator":
            result = await answer_step_coordinator(query_plan_step, additional_information)
            uncertainty = StepUncertainty(completeness=1.0)
        elif isinstance(query_plan_step, SingleAgentQueryPlanStep) and query_plan_step.to_agent != "Coordinator":
            result = await answer_step_data_agent(
                data_agents,
                query_plan_step.query,
                query_plan_step.to_agent,
                additional_information,
            )
            uncertainty = StepUncertainty(completeness=1.0)
        elif isinstance(query_plan_step, MultiAgentQueryPlanStep):
            result, uncertainty = await answer_step_multiple_agents(
                data_agents=data_agents,
                query_plan_step=query_plan_step,
                additional_information=additional_information,
                step_budgets=step_budgets,
            )
        else:
            raise ValueError(f"Unsupported query plan step type: {type(query_plan_step)}")

        tracing.append_to_span_output({"partial_answer": result})
        return result, uncertainty

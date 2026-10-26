"""Multi-agent step execution: query agents in ranked order until complete."""

import pandas as pd
from agents import TResponseInputItem

from qcp_adaptive_termination import hooks
from qcp_adaptive_termination.estimator import Estimator
from qcp_adaptive_termination.hooks import HookEvent
from qcp_adaptive_termination.models import (
    AlignedResultList,
    DataAgent,
    StepBudget,
)
from qcp_adaptive_termination.models.execution import (
    AdditionalInformation,
    PartialQueryAnswer,
    StopDecision,
)
from qcp_adaptive_termination.models.query_plan import MultiAgentQueryPlanStep
from qcp_adaptive_termination.models.uncertainty import StepUncertainty
from qcp_adaptive_termination.observability import context, tracing
from qcp_adaptive_termination.selection_strategy.agent_selector import AgentSelector
from qcp_adaptive_termination.stages.execution.data_agent import answer_step_data_agent
from qcp_adaptive_termination.stages.execution.multi_agent.align_schema import align_schema
from qcp_adaptive_termination.stages.execution.multi_agent.completeness_evaluator import (
    run_completeness_evaluator,
)
from qcp_adaptive_termination.stages.execution.multi_agent.estimator_utils import (
    update_estimator,
)
from qcp_adaptive_termination.stages.execution.multi_agent.result_producer import (
    produce_result,
)
from qcp_adaptive_termination.stages.execution.multi_agent.uncertainty_recording import (
    build_step_uncertainty,
)


def merge_agent_result_into_result_list(
    result_list: AlignedResultList,
    agent_name: str,
    agent_result: pd.DataFrame,
) -> AlignedResultList:
    """Merge one agent's result into the aligned result list.

    Args:
        result_list: the aligned result list so far
        agent_name: the agent that produced the result
        agent_result: the agent's result dataframe

    Returns:
        (AlignedResultList): the merged aligned result list
    """
    schema = result_list.schema_metadata

    agent_result[schema.source_agent_column] = agent_name
    merged = pd.concat([result_list.data, agent_result], ignore_index=True)

    return AlignedResultList(data=merged, schema_metadata=schema)


@hooks.hook(HookEvent.EXECUTION_EXECUTE_QUERY_PLAN_EXECUTE_STEP_ANSWER_STEP_MULTIPLE_AGENTS)
async def answer_step_multiple_agents(
    data_agents: list[DataAgent],
    query_plan_step: MultiAgentQueryPlanStep,
    additional_information: list[AdditionalInformation],
    step_budgets: dict[str, StepBudget],
) -> tuple[PartialQueryAnswer, StepUncertainty | None]:
    """Execute a multi-agent step, querying agents until the result is complete.

    Args:
        data_agents: the available data agents
        query_plan_step: the multi-agent step to execute
        additional_information: the satisfying dependency answers
        step_budgets: per-step completeness budgets

    Returns:
        (tuple[PartialQueryAnswer, StepUncertainty | None]): the step answer and its uncertainty
    """
    agents_by_name = {a.name: a for a in data_agents}
    agents_in_step = [agents_by_name[name] for name in query_plan_step.to_agent]

    selector = await AgentSelector.create(agents=agents_in_step, query=query_plan_step.query)
    remaining_selections = selector.rank(selector.selections)

    result_list = await align_schema(
        data_agents=data_agents,
        agent_names=[s.agent.name for s in remaining_selections[: context.get_config().align_schema_agent_limit]],
        query=query_plan_step.query,
        merge_task=query_plan_step.merge_task,
    )

    step_budget = step_budgets.get(query_plan_step.id)
    answer = PartialQueryAnswer()
    uncertainty: StepUncertainty | None = None
    estimator: Estimator | None = None
    history: list[TResponseInputItem] = []

    while remaining_selections:
        next_agent, *remaining_selections = remaining_selections
        agent_name = next_agent.agent.name

        # Query the agent and merge its result.
        result = await answer_step_data_agent(
            data_agents=data_agents,
            query=query_plan_step.query,
            to_agent=agent_name,
            additional_information=additional_information,
            schema_metadata=result_list.schema_metadata,
        )

        if result.answer is None or result.has_needs or not result.has_answer:
            answer.add_needs(result.needs)
            if len(result_list.data) > 0:
                answer.set_answer(result_list.data)
            elif result.answer is not None:
                answer.set_answer(result.answer)
            break

        result_list = merge_agent_result_into_result_list(result_list, agent_name, result.answer)
        estimator = update_estimator(estimator, result_list, agent_name, step_budget)

        # Decide whether to stop (all agents queried, or completeness evaluator says stop).
        decision = await run_completeness_evaluator(
            result_list,
            estimator,
            query_plan_step.query,
            query_plan_step.merge_task,
            step_budget,
            agents_queried=len(agents_in_step) - len(remaining_selections),
            force_result=not remaining_selections,
            history=history,
        )

        if isinstance(decision, StopDecision):
            final_result = await produce_result(
                result_list,
                estimator,
                query_plan_step.query,
                query_plan_step.merge_task,
                decision.stop_reason,
            )
            if final_result is not None:
                answer.set_answer(final_result)
            uncertainty = build_step_uncertainty(
                estimator,
                decision.stop_reason,
                result_list.schema_metadata.aggregation_types,
            )
            if not context.get_config().continue_after_final_result or not remaining_selections:
                break

        selector.update(result_list, agent_names=[agent_name])
        remaining_selections = selector.rank(remaining_selections)

    tracing.append_to_span_output(
        {
            "agents_queried": len(agents_in_step) - len(remaining_selections),
            "answer": answer,
        }
    )

    return answer, uncertainty

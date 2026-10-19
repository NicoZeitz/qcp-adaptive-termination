import logging
from dataclasses import dataclass

import pandas as pd

from qcp_adaptive_termination import hooks
from qcp_adaptive_termination.estimator import Estimator
from qcp_adaptive_termination.hooks import HookEvent
from qcp_adaptive_termination.models import (
    AlignedResultList,
    DataAgent,
    EstimatorStatistics,
    StepBudget,
    StopReason,
)
from qcp_adaptive_termination.models.execution import (
    AdditionalInformation,
    FinalResultOutcome,
    PartialQueryAnswer,
    RaiseNeedOutcome,
)
from qcp_adaptive_termination.models.query_plan import MultiAgentQueryPlanStep
from qcp_adaptive_termination.observability import context, tracing
from qcp_adaptive_termination.selection_strategy.agent_selector import AgentSelector
from qcp_adaptive_termination.stages.execution.data_agent import answer_step_data_agent
from qcp_adaptive_termination.stages.execution.multi_agent.align_schema import align_schema
from qcp_adaptive_termination.stages.execution.multi_agent.completeness_evaluator import (
    run_completeness_evaluator,
)
from qcp_adaptive_termination.stages.execution.multi_agent.estimator_utils import (
    compute_estimator_statistics,
)
from qcp_adaptive_termination.stages.execution.multi_agent.uncertainty_recording import (
    build_step_uncertainty,
)


@dataclass(slots=True)
class _AgentRecord:
    agent: str
    estimator: EstimatorStatistics | None


def merge_agent_result(
    result_list: AlignedResultList,
    agent_name: str,
    agent_result: pd.DataFrame,
) -> AlignedResultList:
    schema = result_list.schema_metadata

    agent_result[schema.source_agent_column] = agent_name
    merged = pd.concat([result_list.data, agent_result], ignore_index=True)

    return AlignedResultList(data=merged, schema_metadata=schema)


async def query_single_agent(
    data_agents: list[DataAgent],
    query_plan_step: MultiAgentQueryPlanStep,
    agent_name: str,
    additional_information: list[AdditionalInformation],
    result_list: AlignedResultList,
) -> tuple[str, pd.DataFrame | None, RaiseNeedOutcome | None]:
    """Query one agent and return its raw result without touching shared state."""
    query = query_plan_step.query

    result = await answer_step_data_agent(
        data_agents=data_agents,
        query=query,
        to_agent=agent_name,
        additional_information=additional_information,
        schema_metadata=result_list.schema_metadata,
    )

    if len(result.needs) > 0 or result.answer is None or result.has_needs or not result.has_answer:
        need_outcome = RaiseNeedOutcome(
            needs=result.needs,
            partial_answer=result.answer,
            has_partial_answer=result.has_answer,
        )
        return agent_name, None, need_outcome

    return agent_name, result.answer, None


async def query_and_merge_agent(
    data_agents: list[DataAgent],
    query_plan_step: MultiAgentQueryPlanStep,
    agent_name: str,
    additional_information: list[AdditionalInformation],
    result_list: AlignedResultList,
    agents_queried: int,
    agents_total: int,
    previous_estimator: Estimator | None,
) -> tuple[AlignedResultList, _AgentRecord, Estimator | None, RaiseNeedOutcome | None]:
    """Query one agent, merge its result, and compute estimator statistics."""
    agent_name, agent_df, need_outcome = await query_single_agent(
        data_agents=data_agents,
        query_plan_step=query_plan_step,
        agent_name=agent_name,
        additional_information=additional_information,
        result_list=result_list,
    )

    if need_outcome is not None:
        return result_list, _AgentRecord(agent=agent_name, estimator=None), previous_estimator, need_outcome

    assert agent_df is not None
    merged = merge_agent_result(result_list, agent_name, agent_df)
    agent_stats, current_estimator = await compute_estimator_statistics(
        merged,
        agents_queried,
        agents_total,
        [agent_name],
        previous_estimator=previous_estimator,
    )
    return merged, _AgentRecord(agent=agent_name, estimator=agent_stats), current_estimator, None


@hooks.hook(HookEvent.EXECUTION_EXECUTE_QUERY_PLAN_EXECUTE_STEP_ANSWER_STEP_MULTIPLE_AGENTS)
async def answer_step_multiple_agents(
    data_agents: list[DataAgent],
    query_plan_step: MultiAgentQueryPlanStep,
    additional_information: list[AdditionalInformation],
    step_budgets: dict[str, StepBudget],
) -> PartialQueryAnswer:
    agents_by_name = {a.name: a for a in data_agents}
    missing = [name for name in query_plan_step.to_agent if name not in agents_by_name]
    if missing:
        logging.warning(f"Step {query_plan_step.id}: agents not found in data_agents and will be skipped: {missing}")
    agents_in_step = [agents_by_name[name] for name in query_plan_step.to_agent if name in agents_by_name]

    selector = await AgentSelector.create(
        agents=agents_in_step,
        query=query_plan_step.query,
    )
    agents_queried: list[str] = []

    answer = PartialQueryAnswer()
    agent_records: list[_AgentRecord] = []

    ranked_all = selector.rank(selector.selections)
    align_schema_names = [s.agent.name for s in ranked_all[: context.get_config().align_schema_agent_limit]]

    result_list = await align_schema(
        data_agents=data_agents,
        agent_names=align_schema_names,
        query=query_plan_step.query,
        merge_task=query_plan_step.merge_task,
    )

    remaining_selections = ranked_all
    current_estimator: Estimator | None = None
    step_budget = step_budgets.get(query_plan_step.id)

    while remaining_selections:
        next_agent = remaining_selections[0]
        remaining_selections = remaining_selections[1:]
        agent_name = next_agent.agent.name

        result_list, record, current_estimator, need_outcome = await query_and_merge_agent(
            data_agents=data_agents,
            query_plan_step=query_plan_step,
            agent_name=agent_name,
            additional_information=additional_information,
            result_list=result_list,
            agents_queried=len(agents_queried) + 1,
            agents_total=len(agents_in_step),
            previous_estimator=current_estimator,
        )
        agent_records.append(record)
        agents_queried.append(agent_name)

        if need_outcome is not None:
            answer.add_needs(need_outcome.needs)
            answer.set_uncertainty(build_step_uncertainty(query_plan_step.id, None, StopReason.NEED))
            if len(result_list.data) > 0:
                answer.set_answer(result_list.data)
            elif need_outcome.partial_answer is not None:
                answer.set_answer(need_outcome.partial_answer)
            break

        final_stats = record.estimator
        if final_stats is None:
            # No estimator data (agent returned empty / no overlap). If this is the
            # last agent, force a final result from whatever data exists; otherwise
            # keep querying.
            if len(agents_queried) == len(agents_in_step):
                outcome = await run_completeness_evaluator(
                    result_list,
                    EstimatorStatistics(
                        agents_queried=len(agents_queried),
                        agents_total=len(agents_in_step),
                        aggregation_types=result_list.schema_metadata.aggregation_types,
                        new_unique_count=0,
                        observed_count=len(result_list.data),
                    ),
                    query_plan_step.query,
                    query_plan_step.merge_task,
                    step_budget,
                    len(agents_queried),
                    force_result=True,
                )
                if isinstance(outcome, FinalResultOutcome):
                    answer.set_answer(outcome.result)
                    answer.set_uncertainty(build_step_uncertainty(query_plan_step.id, None, outcome.stop_reason))
                break
            continue

        outcome = await run_completeness_evaluator(
            result_list,
            final_stats,
            query_plan_step.query,
            query_plan_step.merge_task,
            step_budget,
            len(agents_queried),
            force_result=len(agents_queried) == len(agents_in_step),
        )

        if isinstance(outcome, RaiseNeedOutcome):
            answer.add_needs(outcome.needs)
            answer.set_uncertainty(build_step_uncertainty(query_plan_step.id, final_stats, StopReason.NEED))
            if len(result_list.data) > 0:
                answer.set_answer(result_list.data)
            elif outcome.partial_answer is not None:
                answer.set_answer(outcome.partial_answer)
            break

        result_list = outcome.result_list

        if isinstance(outcome, FinalResultOutcome):
            answer.set_answer(outcome.result)
            answer.set_uncertainty(build_step_uncertainty(query_plan_step.id, final_stats, outcome.stop_reason))
            if not context.get_config().continue_after_final_result or not remaining_selections:
                break

        selector.update(result_list, agent_names=[agent_name])
        ranked = selector.rank(remaining_selections)
        remaining_selections = ranked

    tracing.append_to_span_output(
        {
            "agent_records_count": len(agent_records),
            "answer": answer,
        }
    )

    return answer

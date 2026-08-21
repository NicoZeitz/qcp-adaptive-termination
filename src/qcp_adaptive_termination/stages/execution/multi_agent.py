import asyncio
import logging
from dataclasses import dataclass

import pandas as pd

from qcp_adaptive_termination import hooks
from qcp_adaptive_termination.hooks import HookEvent
from qcp_adaptive_termination.models import (
    AlignedResultList,
    DataAgent,
    EstimatorStatistics,
    ScoredDataAgent,
)
from qcp_adaptive_termination.models.execution import (
    AdditionalInformation,
    EvaluatorDecision,
    FinalResultOutcome,
    PartialQueryAnswer,
    QueryMoreAgentsOutcome,
    RaiseNeedOutcome,
)
from qcp_adaptive_termination.models.query_plan import MultiAgentQueryPlanStep
from qcp_adaptive_termination.observability import context, tracing
from qcp_adaptive_termination.selection_strategy.agent_selector import AgentSelector
from qcp_adaptive_termination.stages.execution.align_schema import align_schema
from qcp_adaptive_termination.stages.execution.completeness_evaluator import (
    run_completeness_evaluator,
)
from qcp_adaptive_termination.stages.execution.data_agent import answer_step_data_agent
from qcp_adaptive_termination.stages.execution.estimator_utils import (
    compute_estimator_statistics,
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


async def run_agent_batch(
    data_agents: list[DataAgent],
    query_plan_step: MultiAgentQueryPlanStep,
    batch: list[ScoredDataAgent],
    additional_information: list[AdditionalInformation],
    result_list: AlignedResultList,
    agents_queried: list[str],
    remaining_after_batch: list[ScoredDataAgent],
    batch_records: list[_AgentRecord],
    iteration: int,
) -> tuple[FinalResultOutcome | QueryMoreAgentsOutcome | RaiseNeedOutcome, list[_AgentRecord]]:
    """Query all agents in batch in parallel, merge results, then run evaluator once."""
    batch_names = [s.agent.name for s in batch]
    agents_total = len(agents_queried) + len(batch_names) + len(remaining_after_batch)
    batch_label = ", ".join(batch_names)

    hooks.emit(
        HookEvent.MULTI_AGENT_STEP_BATCH_START,
        query_plan_step=query_plan_step,
        additional_information=additional_information,
        iteration=iteration,
        batch=batch,
        queried=agents_queried,
        remaining=remaining_after_batch,
        result_list=result_list,
    )
    with tracing.own_span(
        f"Batch: [{batch_label}]",
        input={
            "query": query_plan_step.query,
            "batch": batch_names,
            "agents_queried": agents_queried,
            "agents_remaining": [s.agent.name for s in remaining_after_batch],
        },
    ):
        results = await asyncio.gather(
            *[
                query_single_agent(
                    data_agents=data_agents,
                    query_plan_step=query_plan_step,
                    agent_name=agent_name,
                    additional_information=additional_information,
                    result_list=result_list,
                )
                for agent_name in batch_names
            ]
        )

        need_results = [
            (agent_name, need_outcome) for agent_name, _, need_outcome in results if need_outcome is not None
        ]
        if need_results:
            merged_needs: list[str] = []
            partial_dfs: list[pd.DataFrame] = []
            for _agent_name, need_outcome in need_results:
                merged_needs.extend(need_outcome.needs)
                if need_outcome.has_partial_answer and need_outcome.partial_answer is not None:
                    partial_dfs.append(need_outcome.partial_answer)
            merged_partial = pd.concat(partial_dfs, ignore_index=True) if partial_dfs else None
            combined_need_outcome = RaiseNeedOutcome(
                needs=list(set(merged_needs)),
                partial_answer=merged_partial,
                has_partial_answer=merged_partial is not None,
            )
            tracing.append_to_span_output(
                {
                    "outcome": combined_need_outcome,
                    "agents_with_needs": [a for a, _ in need_results],
                }
            )
            hooks.emit(
                HookEvent.MULTI_AGENT_STEP_BATCH_END,
                query_plan_step=query_plan_step,
                additional_information=additional_information,
                iteration=iteration,
                batch=batch,
                remaining=remaining_after_batch,
                result_list=result_list,
                estimators=[None] * len(batch_names),
                decision="need",
            )
            return combined_need_outcome, [_AgentRecord(agent=name, estimator=None) for name in batch_names]

        per_agent_stats: list[tuple[str, AlignedResultList, EstimatorStatistics]] = []
        current_result_list = result_list
        agents_queried_so_far = len(agents_queried)
        for idx, (agent_name, agent_df, _) in enumerate(results):
            assert agent_df is not None
            current_result_list = merge_agent_result(current_result_list, agent_name, agent_df)
            agent_stats = compute_estimator_statistics(
                current_result_list,
                agents_queried_so_far + idx + 1,
                agents_total,
                [agent_name],
            )
            per_agent_stats.append((agent_name, current_result_list, agent_stats))

        merged_result_list = current_result_list
        final_stats = per_agent_stats[-1][2]

        if final_stats.sample_coverage is None and remaining_after_batch:
            new_records = [_AgentRecord(agent=name, estimator=stats) for name, _, stats in per_agent_stats]
            tracing.append_to_span_output({"outcome": "query_more_agents", "batch": batch_names})
            hooks.emit(
                HookEvent.MULTI_AGENT_STEP_BATCH_END,
                query_plan_step=query_plan_step,
                additional_information=additional_information,
                iteration=iteration,
                batch=batch,
                remaining=remaining_after_batch,
                result_list=merged_result_list,
                estimators=[stats for _, _, stats in per_agent_stats],
                decision="query_more_agents",
            )
            return QueryMoreAgentsOutcome(result_list=merged_result_list), new_records

        outcome = await run_completeness_evaluator(
            merged_result_list,
            final_stats,
            query_plan_step.query,
            [r.estimator for r in batch_records if r.estimator is not None],
            query_plan_step.merge_task,
        )
        decision: EvaluatorDecision = outcome.decision

        new_records = [_AgentRecord(agent=name, estimator=stats) for name, _, stats in per_agent_stats]

        tracing.append_to_span_output({"outcome": outcome, "batch": batch_names})
        hooks.emit(
            HookEvent.MULTI_AGENT_STEP_BATCH_END,
            query_plan_step=query_plan_step,
            additional_information=additional_information,
            iteration=iteration,
            batch=batch,
            remaining=remaining_after_batch,
            result_list=merged_result_list,
            estimators=[stats for _, _, stats in per_agent_stats],
            decision=decision,
        )
        return outcome, new_records


async def _answer_step_multiple_agents_via_loop(
    data_agents: list[DataAgent],
    query_plan_step: MultiAgentQueryPlanStep,
    additional_information: list[AdditionalInformation],
    batch_size: int = 1,
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
    batch_records: list[_AgentRecord] = []

    ranked_all = selector.rank(selector.selections)
    align_schema_names = [s.agent.name for s in ranked_all[: context.get_config().align_schema_agent_limit]]

    hooks.emit(
        HookEvent.MULTI_AGENT_STEP_START,
        query_plan_step=query_plan_step,
        additional_information=additional_information,
        initial_ranking=ranked_all,
    )

    result_list = await align_schema(
        data_agents=data_agents,
        agent_names=align_schema_names,
        query=query_plan_step.query,
        merge_task=query_plan_step.merge_task,
    )

    first_batch_size = max(2, batch_size)
    batch_selections = ranked_all[:first_batch_size]
    remaining_selections = ranked_all[first_batch_size:]
    iteration = 0

    while batch_selections:
        outcome, new_records = await run_agent_batch(
            data_agents=data_agents,
            query_plan_step=query_plan_step,
            batch=batch_selections,
            additional_information=additional_information,
            result_list=result_list,
            agents_queried=agents_queried,
            remaining_after_batch=remaining_selections,
            batch_records=batch_records,
            iteration=iteration,
        )
        batch_records.extend(new_records)
        agents_queried.extend(r.agent for r in new_records)
        iteration += 1

        if isinstance(outcome, RaiseNeedOutcome):
            answer.add_needs(outcome.needs)

            if len(result_list.data) > 0:
                answer.set_answer(result_list.data)
            elif outcome.partial_answer is not None:
                answer.set_answer(outcome.partial_answer)

            break

        result_list = outcome.result_list

        if isinstance(outcome, FinalResultOutcome):
            answer.set_answer(outcome.result)

            if not context.get_config().continue_after_final_result or not remaining_selections:
                break

        selector.update(result_list, agent_names=[r.agent for r in new_records])
        ranked = selector.rank(remaining_selections)
        batch_selections = ranked[:batch_size]
        remaining_selections = ranked[batch_size:]

    tracing.append_to_span_output(
        {
            "batch_records_count": len(batch_records),
            "answer": answer,
        }
    )

    hooks.emit(
        HookEvent.MULTI_AGENT_STEP_END,
        query_plan_step=query_plan_step,
        additional_information=additional_information,
        result_list=result_list,
        queried=agents_queried,
        remaining=[s.agent.name for s in remaining_selections],
        estimators=[r.estimator for r in batch_records],
    )

    return answer


async def answer_step_multiple_agents(
    data_agents: list[DataAgent],
    query_plan_step: MultiAgentQueryPlanStep,
    additional_information: list[AdditionalInformation],
    batch_size: int = 1,
) -> PartialQueryAnswer:
    return await hooks.call_override(
        HookEvent.ANSWER_STEP_MULTIPLE_AGENTS_OVERRIDE,
        default=lambda: _answer_step_multiple_agents_via_loop(
            data_agents, query_plan_step, additional_information, batch_size
        ),
        data_agents=data_agents,
        query_plan_step=query_plan_step,
        additional_information=additional_information,
    )

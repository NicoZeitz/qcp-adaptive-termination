import asyncio
from typing import cast

import pandas as pd
from agents import (
    CustomSpanData,
    Span,
)

from qcp_adaptive_termination.model.agent.data_agent import DataAgent
from qcp_adaptive_termination.model.estimation.aligned_result_list import (
    AlignedResultList,
)
from qcp_adaptive_termination.model.estimation.estimator_statistics import (
    EstimatorStatistics,
)
from qcp_adaptive_termination.model.execution.additional_information import (
    AdditionalInformation,
)
from qcp_adaptive_termination.model.execution.outcome import (
    FinalResultOutcome,
    QueryMoreAgentsOutcome,
    RaiseNeedOutcome,
)
from qcp_adaptive_termination.model.execution.partial_query_answer import (
    PartialQueryAnswer,
)
from qcp_adaptive_termination.model.config import config_var
from qcp_adaptive_termination.model.multi_agent_iteration_record import (
    MultiAgentIterationRecord,
)
from qcp_adaptive_termination.model.query_plan.query_plan_step import (
    MultiAgentQueryPlanStep,
)
from qcp_adaptive_termination.selection_strategy import AgentSelector
from qcp_adaptive_termination.tracing.span import (
    append_to_span_output,
    own_custom_span,
    span_context_var,
)
from qcp_adaptive_termination.stages.execution.align_schema import align_schema
from qcp_adaptive_termination.stages.execution.completeness_evaluator import (
    run_completeness_evaluator,
)
from qcp_adaptive_termination.stages.execution.data_agent import answer_step_data_agent
from qcp_adaptive_termination.stages.execution.estimator_utils import (
    compute_estimator_statistics,
)
from qcp_adaptive_termination import hooks
from qcp_adaptive_termination.hooks import HookEvent

MULTI_AGENT_ITERATION_RECORDS: dict[str, list[MultiAgentIterationRecord]] = {}


@hooks.on(HookEvent.ITERATION_RECORDS)
def _store_iteration_records(
    execution_id: str, records: list[MultiAgentIterationRecord]
) -> None:
    MULTI_AGENT_ITERATION_RECORDS[execution_id] = records


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

    if (
        len(result.needs) > 0
        or result.answer is None
        or result.has_needs
        or not result.has_answer
    ):
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
    batch: list[str],
    additional_information: list[AdditionalInformation],
    result_list: AlignedResultList,
    agents_queried: list[str],
    agents_remaining: list[str],
    iteration_records: list[MultiAgentIterationRecord],
    rankings: list[tuple[str, float | None]],
) -> FinalResultOutcome | QueryMoreAgentsOutcome | RaiseNeedOutcome:
    """Query all agents in batch in parallel, merge results, then run evaluator once."""
    agents_total = len(agents_queried) + len(agents_remaining)
    agents_queried_now = agents_queried + batch
    remaining_agents_now = [a for a in agents_remaining if a not in batch]
    batch_label = ", ".join(batch)

    with own_custom_span(
        f"Batch: [{batch_label}]",
        input={
            "query": query_plan_step.query,
            "batch": batch,
            "agents_queried": agents_queried,
            "agents_remaining": agents_remaining,
            "rankings": rankings,
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
                for agent_name in batch
            ]
        )

        need_results = [
            (agent_name, need_outcome)
            for agent_name, _, need_outcome in results
            if need_outcome is not None
        ]
        if need_results:
            merged_needs: list[str] = []
            partial_dfs: list[pd.DataFrame] = []
            for agent_name, need_outcome in need_results:
                merged_needs.extend(need_outcome.needs)
                if (
                    need_outcome.has_partial_answer
                    and need_outcome.partial_answer is not None
                ):
                    partial_dfs.append(need_outcome.partial_answer)
                iteration_records.append(
                    MultiAgentIterationRecord(
                        step_id=query_plan_step.id,
                        agent_name=agent_name,
                        result_list=result_list.data,
                        estimator_result=None,
                        evaluator_decision=None,
                        queried_agents=agents_queried_now,
                        remaining_agents=remaining_agents_now,
                    )
                )
            merged_partial = (
                pd.concat(partial_dfs, ignore_index=True) if partial_dfs else None
            )
            combined_need_outcome = RaiseNeedOutcome(
                needs=list(set(merged_needs)),
                partial_answer=merged_partial,
                has_partial_answer=merged_partial is not None,
            )
            append_to_span_output(
                {
                    "outcome": combined_need_outcome,
                    "agents_with_needs": [a for a, _ in need_results],
                }
            )
            return combined_need_outcome

        per_agent_stats: list[tuple[str, AlignedResultList, EstimatorStatistics]] = []
        current_result_list = result_list
        for idx, (agent_name, agent_df, _) in enumerate(results):
            assert agent_df is not None
            current_result_list = merge_agent_result(
                current_result_list, agent_name, agent_df
            )
            agent_stats = compute_estimator_statistics(
                current_result_list,
                len(agents_queried) + idx + 1,
                agents_total,
                [agent_name],
            )
            per_agent_stats.append((agent_name, current_result_list, agent_stats))

        merged_result_list = current_result_list
        final_stats = per_agent_stats[-1][2]

        if final_stats.sample_coverage is None:
            outcome = QueryMoreAgentsOutcome(result_list=merged_result_list)
        else:
            outcome = await run_completeness_evaluator(
                merged_result_list,
                final_stats,
                query_plan_step.query,
                [
                    r.estimator_result
                    for r in iteration_records
                    if r.estimator_result is not None
                ],
                query_plan_step.merge_task,
            )

        for i, (agent_name, _, agent_stats) in enumerate(per_agent_stats):
            is_last = i == len(per_agent_stats) - 1
            iteration_records.append(
                MultiAgentIterationRecord(
                    step_id=query_plan_step.id,
                    agent_name=agent_name,
                    result_list=merged_result_list.data,
                    estimator_result=agent_stats,
                    evaluator_decision=outcome.decision if is_last else "batch_query",
                    queried_agents=agents_queried + batch[: i + 1],
                    remaining_agents=batch[i + 1 :] + remaining_agents_now,
                )
            )

        append_to_span_output({"outcome": outcome, "batch": batch})
        return outcome


async def answer_step_multiple_agents(
    data_agents: list[DataAgent],
    query_plan_step: MultiAgentQueryPlanStep,
    additional_information: list[AdditionalInformation],
    batch_size: int = 1,
) -> PartialQueryAnswer:
    agents_by_name = {a.name: a for a in data_agents}
    agents_in_step = [
        agents_by_name[name]
        for name in query_plan_step.to_agent
        if name in agents_by_name
    ]

    selector = await AgentSelector.create(
        agents=agents_in_step,
        query=query_plan_step.query,
    )
    agents_queried: list[str] = []

    answer = PartialQueryAnswer()
    iteration_records = []

    ranked_all = selector.rank(selector.selections)
    rankings = [(s.agent.name, s.score) for s in ranked_all]
    align_schema_names = [
        s.agent.name for s in ranked_all[: config_var.get().align_schema_agent_limit]
    ]

    result_list = await align_schema(
        data_agents=data_agents,
        agent_names=align_schema_names,
        query=query_plan_step.query,
        merge_task=query_plan_step.merge_task,
    )

    first_batch_size = max(2, batch_size)
    batch_selections = ranked_all[:first_batch_size]
    remaining_selections = ranked_all[first_batch_size:]

    while batch_selections:
        batch = [s.agent.name for s in batch_selections]
        outcome = await run_agent_batch(
            data_agents=data_agents,
            query_plan_step=query_plan_step,
            batch=batch,
            additional_information=additional_information,
            result_list=result_list,
            agents_queried=agents_queried,
            agents_remaining=[
                s.agent.name for s in batch_selections + remaining_selections
            ],
            iteration_records=iteration_records,
            rankings=rankings,
        )
        agents_queried.extend(batch)

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

            if not config_var.get().continue_after_final_result:
                break

        selector.update(result_list, agent_names=batch)
        ranked = selector.rank(remaining_selections)
        rankings = [(s.agent.name, s.score) for s in ranked]
        batch_selections = ranked[:batch_size]
        remaining_selections = ranked[batch_size:]

    append_to_span_output(
        {
            "iteration_records_count": len(iteration_records),
            "iteration_records": iteration_records,
            "answer": answer,
        }
    )
    hooks.emit(
        HookEvent.ITERATION_RECORDS,
        execution_id=f"{query_plan_step.id} {cast(Span[CustomSpanData], span_context_var.get()).span_id}",
        records=iteration_records,
    )
    return answer

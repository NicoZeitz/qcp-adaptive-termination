from qtypes.additional_information import AdditionalInformation
from qtypes.aligned_result_list import (
    AlignedResultList,
)
from qtypes.data_agent import DataAgent
from qtypes.multi_agent_iteration_record import (
    MultiAgentIterationRecord,
)
from qtypes.outcome import FinalResultOutcome, QueryMoreAgentsOutcome, RaiseNeedOutcome
from qtypes.partial_query_answer import PartialQueryAnswer
from qtypes.query_plan_step import MultiAgentQueryPlanStep
from stages.execution.answer_step_data_agent import answer_step_data_agent
from stages.execution.answer_step_multiple_agents.align_schema import align_schema
from stages.execution.answer_step_multiple_agents.completeness_evaluator import (
    run_completeness_evaluator,
)
from stages.execution.answer_step_multiple_agents.compute_estimator import (
    compute_estimator_statistics,
)
from stages.execution.answer_step_multiple_agents.validation import SOURCE_AGENT_COLUMN
from utils.span import append_to_span_output, own_custom_span

MULTI_AGENT_ITERATION_RECORDS: dict[str, list[MultiAgentIterationRecord]] = {}


async def run_agent_round(
    data_agents: list[DataAgent],
    query_plan_step: MultiAgentQueryPlanStep,
    agent_current: str,
    additional_information: list[AdditionalInformation],
    result_list: AlignedResultList,
    agents_queried: list[str],
    agents_remaining: list[str],
    iteration_records: list[MultiAgentIterationRecord],
) -> FinalResultOutcome | QueryMoreAgentsOutcome | RaiseNeedOutcome:
    query = query_plan_step.query
    agents_total = len(agents_remaining) + len(agents_queried)
    agents_queried_now = agents_queried + [agent_current]
    remaining_agents_now = [a for a in agents_remaining if a != agent_current]

    with own_custom_span(
        f"Multiple Agents - Round {len(agents_queried_now)}: {agent_current}",
        input={
            "query": query,
            "agent_name": agent_current,
            "agents_queried": agents_queried,
            "agents_remaining": agents_remaining,
        },
    ):
        iteration_record = MultiAgentIterationRecord(
            step_id=query_plan_step.id,
            agent_name=agent_current,
            result_list=result_list.data,
            estimator_result=None,
            evaluator_decision=None,
            evaluator_next_agents=remaining_agents_now,
        )

        result = await answer_step_data_agent(
            data_agents=data_agents,
            query=query,
            to_agent=agent_current,
            additional_information=additional_information,
        )

        if (
            len(result.needs) > 0
            or result.answer is None
            or result.has_needs
            or not result.has_answer
        ):
            outcome = RaiseNeedOutcome(
                needs=result.needs,
                partial_answer=result.answer,
                has_partial_answer=result.has_answer,
            )
            append_to_span_output(
                {
                    "outcome": outcome,
                    "iteration_record": iteration_record,
                }
            )
            iteration_records.append(iteration_record)
            return outcome

        result_list = await align_schema(
            result_list, agent_current, result.answer, query_plan_step.merge_task
        )

        estimator_statistics = compute_estimator_statistics(
            result_list, len(agents_queried_now), agents_total
        )

        if estimator_statistics is None:
            outcome = QueryMoreAgentsOutcome(result_list=result_list)
        else:
            iteration_record.estimator_result = estimator_statistics[0]
            new_unique_count = (
                result_list.data[SOURCE_AGENT_COLUMN] == agent_current
            ).sum()
            outcome = await run_completeness_evaluator(
                result_list,
                estimator_statistics[1],
                estimator_statistics[0],
                query,
                new_unique_count,
                iteration_records,
            )
            iteration_record.evaluator_decision = outcome.decision

        append_to_span_output(
            {
                "outcome": outcome,
                "iteration_record": iteration_record,
            }
        )
        iteration_records.append(iteration_record)
        return outcome


async def answer_step_multiple_agents(
    data_agents: list[DataAgent],
    query_plan_step: MultiAgentQueryPlanStep,
    additional_information: list[AdditionalInformation],
    continue_after_final_result: bool = False,
) -> PartialQueryAnswer:
    all_agents = list(query_plan_step.to_agent)
    agents_queried: list[str] = []
    result_list = AlignedResultList()

    answer = PartialQueryAnswer()
    iteration_records = []

    with own_custom_span(
        f"Multiple Agents: {query_plan_step.id}",
        input={"query": query_plan_step.query, "agents": all_agents},
    ) as span:
        for agent_name in all_agents:
            outcome = await run_agent_round(
                data_agents=data_agents,
                query_plan_step=query_plan_step,
                agent_current=agent_name,
                additional_information=additional_information,
                result_list=result_list,
                agents_queried=agents_queried,
                agents_remaining=[a for a in all_agents if a not in agents_queried],
                iteration_records=iteration_records,
            )
            agents_queried.append(agent_name)

            if isinstance(outcome, RaiseNeedOutcome):
                answer.add_needs(outcome.needs)

                if len(result_list.data) > 0:
                    answer.set_answer(result_list.data)
                elif outcome.partial_answer is not None:
                    answer.set_answer(outcome.partial_answer)

                break

            if isinstance(outcome, FinalResultOutcome):
                answer.set_answer(outcome.result)

                if continue_after_final_result:
                    continue

                break

            result_list = outcome.result_list

        append_to_span_output(
            {
                "iteration_records_count": len(iteration_records),
                "iteration_records": iteration_records,
                "answer": answer,
            }
        )
        MULTI_AGENT_ITERATION_RECORDS[f"{query_plan_step.id} {span.span_id}"] = (
            iteration_records
        )
        return answer

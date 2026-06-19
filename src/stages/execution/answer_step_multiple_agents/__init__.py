from qtypes.additional_information import AdditionalInformation
from qtypes.aligned_result_list import (
    AlignedResultList,
)
from qtypes.data_agent import DataAgent
from qtypes.multi_agent_iteration_record import (
    EvaluatorDecision,
    MultiAgentIterationRecord,
)
from qtypes.partial_query_answer import PartialQueryAnswer
from qtypes.query_plan_step import MultiAgentQueryPlanStep
from qtypes.result import ResultType
from stages.execution.answer_step_data_agent import answer_step_data_agent
from stages.execution.answer_step_multiple_agents.align_schema import align_schema
from stages.execution.answer_step_multiple_agents.completeness_evaluator import (
    run_completeness_evaluator,
)
from stages.execution.answer_step_multiple_agents.compute_estimator import (
    compute_estimator_result,
)
from utils.span import append_to_span_output, own_custom_span


async def answer_step_multiple_agents(
    data_agents: list[DataAgent],
    query_plan_step: MultiAgentQueryPlanStep,
    additional_information: list[AdditionalInformation],
    continue_after_final_result: bool = False,
) -> tuple[PartialQueryAnswer, list[MultiAgentIterationRecord]]:
    query = query_plan_step.query
    all_agents = list(sorted(query_plan_step.to_agent))
    agents_total = len(all_agents)

    result_list = AlignedResultList()
    iteration_records: list[MultiAgentIterationRecord] = []
    collected_needs: list = []
    per_agent_results: dict[str, ResultType] = {}

    last_final_result: ResultType | None = None
    agents_to_query = list(all_agents)
    queried: list[str] = []

    with own_custom_span(
        f"Multiple Agents: {query_plan_step.id}",
        input={"query": query, "agents": all_agents},
    ):
        for agent_name in all_agents:
            if agent_name not in agents_to_query:
                continue

            iteration_index = len(iteration_records) + 1
            with own_custom_span(
                f"Multiple Agents - Round {iteration_index}: {agent_name}",
                input={
                    "query": query,
                    "agent_name": agent_name,
                    "iteration": iteration_index,
                    "agents_queried_so_far": list(queried),
                    "agents_remaining": [a for a in all_agents if a not in queried],
                },
            ):
                agent_partial = await answer_step_data_agent(
                    data_agents=data_agents,
                    query=query,
                    to_agent=agent_name,
                    additional_information=additional_information,
                )
                queried.append(agent_name)

                if not agent_partial.has_answer or agent_partial.answer is None:
                    if agent_partial.has_needs:
                        collected_needs.extend(agent_partial.needs)
                    estimator_result = (
                        compute_estimator_result(
                            result_list.schema,
                            per_agent_results,
                            len(queried),
                            agents_total,
                        )
                        if result_list.schema is not None
                        else None
                    )
                    record = MultiAgentIterationRecord(
                        step_id=query_plan_step.id,
                        agent_name=agent_name,
                        result_list=result_list.data.copy(),
                        estimator_result=estimator_result,
                        evaluator_decision="query_more_agents",
                        evaluator_next_agents=[
                            a for a in all_agents if a not in queried
                        ],
                    )
                    iteration_records.append(record)
                    append_to_span_output(
                        {
                            "agent_has_answer": False,
                            "evaluator_decision": "query_more_agents",
                        }
                    )
                    continue

                per_agent_results[agent_name] = agent_partial.answer

                result_list = await align_schema(
                    result_list, agent_name, agent_partial.answer, query
                )

                estimator_result = (
                    compute_estimator_result(
                        result_list.schema,
                        per_agent_results,
                        len(queried),
                        agents_total,
                    )
                    if result_list.schema is not None
                    else None
                )

                remaining = [a for a in all_agents if a not in queried]

                if estimator_result is None:
                    # first agent done, always continue
                    decision: EvaluatorDecision = "query_more_agents"
                    next_agents = remaining
                    final_result_value = None
                else:
                    (
                        decision,
                        next_agents,
                        final_result_value,
                    ) = await run_completeness_evaluator(
                        result_list, estimator_result, query, remaining
                    )

                record = MultiAgentIterationRecord(
                    step_id=query_plan_step.id,
                    agent_name=agent_name,
                    result_list=result_list.data.copy(),
                    estimator_result=estimator_result,
                    evaluator_decision=decision,
                    evaluator_next_agents=next_agents,
                )
                iteration_records.append(record)

                append_to_span_output(
                    {
                        "agent_has_answer": True,
                        "aligned_result_rows": len(result_list.data),
                        "evaluator_decision": decision,
                        "evaluator_next_agents": next_agents,
                        "estimator_sample_coverage": estimator_result.sample_coverage
                        if estimator_result is not None
                        else None,
                        "estimator_agents_queried": estimator_result.agents_queried
                        if estimator_result is not None
                        else None,
                    }
                )

                if decision == "final_result":
                    if final_result_value is not None:
                        last_final_result = final_result_value
                    if not continue_after_final_result:
                        break
                elif decision == "query_more_agents":
                    agents_to_query = [a for a in next_agents if a not in queried]

        append_to_span_output({"iteration_records_count": len(iteration_records)})

        answer = PartialQueryAnswer()
        if last_final_result is not None:
            answer.set_answer(last_final_result)
        elif not result_list.data.empty:
            answer.set_answer(result_list.data)

        if collected_needs:
            answer.add_needs(collected_needs)

        return answer, iteration_records

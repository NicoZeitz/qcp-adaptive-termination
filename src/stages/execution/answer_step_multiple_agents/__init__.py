from dataclasses import dataclass, field

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


@dataclass
class AgentRoundOutcome:
    record: MultiAgentIterationRecord
    result_list: AlignedResultList
    new_agent_result: ResultType | None
    new_needs: list = field(default_factory=list)
    decision: EvaluatorDecision = "query_more_agents"
    next_agents: list[str] = field(default_factory=list)
    final_result_value: ResultType | None = None


async def run_agent_round(
    data_agents: list[DataAgent],
    query: str,
    step_id: str,
    agent_name: str,
    additional_information: list[AdditionalInformation],
    result_list: AlignedResultList,
    per_agent_results: dict[str, ResultType],
    queried: list[str],
    all_agents: list[str],
    agents_total: int,
    iteration_index: int,
) -> AgentRoundOutcome:
    queried_now = queried + [agent_name]

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

        if not agent_partial.has_answer or agent_partial.answer is None:
            estimator_result = (
                compute_estimator_result(
                    result_list.schema,
                    per_agent_results,
                    len(queried_now),
                    agents_total,
                )
                if result_list.schema is not None
                else None
            )
            record = MultiAgentIterationRecord(
                step_id=step_id,
                agent_name=agent_name,
                result_list=result_list.data.copy(),
                estimator_result=estimator_result,
                evaluator_decision="query_more_agents",
                evaluator_next_agents=[a for a in all_agents if a not in queried_now],
            )
            append_to_span_output(
                {"agent_has_answer": False, "evaluator_decision": "query_more_agents"}
            )
            return AgentRoundOutcome(
                record=record,
                result_list=result_list,
                new_agent_result=None,
                new_needs=list(agent_partial.needs) if agent_partial.has_needs else [],
                next_agents=[a for a in all_agents if a not in queried_now],
            )

        updated_result_list = await align_schema(
            result_list, agent_name, agent_partial.answer, query
        )
        updated_per_agent_results = {
            **per_agent_results,
            agent_name: agent_partial.answer,
        }

        estimator_result = (
            compute_estimator_result(
                updated_result_list.schema,
                updated_per_agent_results,
                len(queried_now),
                agents_total,
            )
            if updated_result_list.schema is not None
            else None
        )

        remaining = [a for a in all_agents if a not in queried_now]

        if estimator_result is None:
            decision: EvaluatorDecision = "query_more_agents"
            next_agents = remaining
            final_result_value = None
        else:
            (
                decision,
                next_agents,
                final_result_value,
            ) = await run_completeness_evaluator(
                updated_result_list, estimator_result, query, remaining
            )

        record = MultiAgentIterationRecord(
            step_id=step_id,
            agent_name=agent_name,
            result_list=updated_result_list.data.copy(),
            estimator_result=estimator_result,
            evaluator_decision=decision,
            evaluator_next_agents=next_agents,
        )
        append_to_span_output(
            {
                "agent_has_answer": True,
                "aligned_result_rows": len(updated_result_list.data),
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
        return AgentRoundOutcome(
            record=record,
            result_list=updated_result_list,
            new_agent_result=agent_partial.answer,
            decision=decision,
            next_agents=next_agents,
            final_result_value=final_result_value,
        )


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
            outcome = await run_agent_round(
                data_agents=data_agents,
                query=query,
                step_id=query_plan_step.id,
                agent_name=agent_name,
                additional_information=additional_information,
                result_list=result_list,
                per_agent_results=per_agent_results,
                queried=queried,
                all_agents=all_agents,
                agents_total=agents_total,
                iteration_index=iteration_index,
            )

            queried.append(agent_name)
            collected_needs.extend(outcome.new_needs)
            result_list = outcome.result_list
            if outcome.new_agent_result is not None:
                per_agent_results[agent_name] = outcome.new_agent_result
            iteration_records.append(outcome.record)

            if outcome.new_agent_result is None:
                continue

            if outcome.final_result_value is not None:
                last_final_result = outcome.final_result_value

            if outcome.decision == "final_result":
                if not continue_after_final_result:
                    break
            elif outcome.decision == "query_more_agents":
                agents_to_query = [a for a in outcome.next_agents if a not in queried]

        append_to_span_output({"iteration_records_count": len(iteration_records)})

        answer = PartialQueryAnswer()
        if last_final_result is not None:
            answer.set_answer(last_final_result)
        elif not result_list.data.empty:
            answer.set_answer(result_list.data)

        if collected_needs:
            answer.add_needs(collected_needs)

        return answer, iteration_records

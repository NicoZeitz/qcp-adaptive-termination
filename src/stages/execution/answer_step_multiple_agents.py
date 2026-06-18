from typing import Any, cast

from agents import FunctionToolResult, RunContextWrapper, ToolsToFinalOutputResult
import pandas as pd

from code_execution.code_execution_environment import CodeExecutionEnvironment
from qtypes.additional_information import AdditionalInformation
from qtypes.aligned_result_list import AggregationType, AlignedResultList, AlignedSchemaMetadata
from qtypes.data_agent import DataAgent
from qtypes.estimator_result import EstimatorResult
from qtypes.multi_agent_iteration_record import (
    EvaluatorDecision,
    MultiAgentIterationRecord,
)
from qtypes.partial_query_answer import PartialQueryAnswer, PartialQueryAnswerDummy
from qtypes.query_plan_step import MultiAgentQueryPlanStep
from qtypes.result import ResultType, try_parse_value_to_result_type
from stages.common.agent_configuration import get_coordinator_agent_configuration
from stages.execution.answer_step_data_agent import answer_step_data_agent
from unknown_unknown import DataItem, Estimator
from utils.run_agent import run_agent
from utils.span import append_to_span_output, own_custom_span


def dataframe_to_data_items(
    df: pd.DataFrame, key_cols: list[str], value_col: str | None, source_id: str
) -> list[DataItem]:
    items = []
    for _, row in df.iterrows():
        name = "|".join(str(row[k]) for k in key_cols if k in row)
        value = float(row[value_col]) if value_col and value_col in row else 1.0
        items.append(DataItem(name=name, value=value, source_id=source_id))
    return items


def compute_estimator_result(
    aligned: AlignedResultList,
    agents_queried: int,
    agents_total: int,
) -> EstimatorResult | None:
    schema = aligned.schema
    if schema is None or aligned.data.empty:
        return None

    try:
        source_samples: list[list[DataItem]] = []
        for source_id in aligned.data["source_agent"].unique():
            subset = aligned.data[aligned.data["source_agent"] == source_id]
            items = dataframe_to_data_items(
                subset, schema.keys, schema.value_column, str(source_id)
            )
            if items:
                source_samples.append(items)

        if len(source_samples) < 2:
            return None

        est = Estimator(source_samples)
        agg = schema.aggregation_type
        coverage = est.sample_coverage()

        result = EstimatorResult(
            sample_coverage=coverage,
            agents_queried=agents_queried,
            agents_total=agents_total,
            aggregation_type=agg,
        )

        if agg == "count":
            result.observed_count = est.sample_count()
            result.estimated_count = est.estimated_count()
            lower, upper = est.estimated_count_bounds()
            result.estimated_count_lower = lower
            result.estimated_count_upper = upper
        elif agg == "sum":
            result.observed_sum = est.sample_sum()
            result.estimated_sum = est.estimated_sum()
            result.estimated_sum_lower = est.estimated_sum_lower_bound()
            result.estimated_sum_upper = est.estimated_sum_upper_bound()
        elif agg == "avg":
            result.observed_avg = est.sample_avg()
            result.estimated_avg = est.estimated_avg()
            result.estimated_avg_lower = est.estimated_avg_lower_bound()
            result.estimated_avg_upper = est.estimated_avg_upper_bound()
        elif agg == "min":
            result.estimated_min = est.estimated_min()
        elif agg == "max":
            result.estimated_max = est.estimated_max()

        return result
    except Exception:
        return None


async def align_schema(
    aligned: AlignedResultList,
    agent_name: str,
    agent_result: ResultType,
    query: str,
) -> AlignedResultList:
    is_first = aligned.schema is None

    new_data = agent_result.copy()
    new_data["source_agent"] = agent_name

    partial_answer: PartialQueryAnswer | None = None
    captured_schema: AlignedSchemaMetadata | None = None

    def set_schema(aggregation_type: AggregationType, keys: list, value_column: str | None = None):
        nonlocal partial_answer, captured_schema
        captured_schema = AlignedSchemaMetadata(
            aggregation_type=aggregation_type,
            keys=list(keys),
            value_column=value_column,
        )
        partial_answer = PartialQueryAnswer()
        partial_answer.set_answer(new_data)

    def merge_result(merged_df: Any):
        nonlocal partial_answer
        if partial_answer is None:
            partial_answer = PartialQueryAnswer()
        partial_answer.set_answer(try_parse_value_to_result_type(merged_df))

    if is_first:
        code_environment = CodeExecutionEnvironment(
            variables={
                "query": query,
                "agent_result": new_data,
                "pd": pd,
            },
            tools={"set_schema": set_schema},
        )
        prompt_name = "execution-coordinator-align-schema-first"
        prompt_parameters: dict[str, str | None] = {
            "query": query,
            "agent_name": agent_name,
            "agent_result_preview": str(new_data.head(5).to_string()),
        }
    else:
        code_environment = CodeExecutionEnvironment(
            variables={
                "query": query,
                "existing_result_list": aligned.data,
                "agent_result": new_data,
                "pd": pd,
            },
            tools={"merge_result": merge_result},
        )
        prompt_name = "execution-coordinator-align-schema-continue"
        prompt_parameters: dict[str, str | None] = {
            "query": query,
            "agent_name": agent_name,
            "agent_result_preview": str(new_data.head(5).to_string()),
            "existing_result_preview": str(aligned.data.head(10).to_string()),
            "schema": str(aligned.schema),
        }

    def custom_tool_handler(
        _context: RunContextWrapper[None],
        _tool_results: list[FunctionToolResult],
    ) -> ToolsToFinalOutputResult:
        is_done = partial_answer is not None and partial_answer.is_final
        return ToolsToFinalOutputResult(
            is_final_output=is_done,
            final_output=partial_answer,
        )

    with own_custom_span(
        f"Query Coordinator - Align Schema {'First' if is_first else 'Continue'}",
        input={
            "query": query,
            "agent_name": agent_name,
            "is_first": is_first,
            "agent_result_rows": len(new_data),
            "agent_result_preview": str(new_data.head(5).to_string()),
            "existing_result_rows": len(aligned.data) if not is_first else None,
            "existing_result_preview": str(aligned.data.head(10).to_string()) if not is_first
            else None,
        },
    ):
        await run_agent(
            agent_configuration=get_coordinator_agent_configuration(
                "Align Schema First" if is_first else "Align Schema Continue"
            ),
            prompt_name=prompt_name,
            prompt_parameters=prompt_parameters,
            tools=[code_environment.as_function_tool()],
            tool_use_behavior=custom_tool_handler,
            reset_tool_choice=False,
            parallel_tool_calls=False,
            tool_choice="required",
            output_type=PartialQueryAnswerDummy,
            get_actual_result=lambda: partial_answer,
        )
        append_to_span_output({
            "has_answer": partial_answer is not None and partial_answer.has_answer,
            "result_rows": len(partial_answer.answer)
            if partial_answer is not None
            and partial_answer.has_answer
            and partial_answer.answer is not None
            else 0,
            "captured_schema": str(captured_schema) if captured_schema is not None else None,
        })

    if partial_answer is None or not partial_answer.has_answer:
        # fallback: just append raw
        new_data["source_agent"] = agent_name
        merged = (
            pd.concat([aligned.data, new_data], ignore_index=True)
            if not aligned.data.empty
            else new_data
        )
        return AlignedResultList(data=merged, schema=aligned.schema)

    new_schema = captured_schema if is_first else aligned.schema
    return AlignedResultList(
        data=cast(pd.DataFrame, partial_answer.answer),
        schema=new_schema if new_schema is not None else aligned.schema,
    )


async def _run_completeness_evaluator(
    aligned: AlignedResultList,
    estimator_result: EstimatorResult | None,
    query: str,
    remaining_agents: list[str],
) -> tuple[EvaluatorDecision, list[str], ResultType | None]:
    decision: EvaluatorDecision = "query_more_agents"
    next_agents: list[str] = list(remaining_agents)
    final_result_value: ResultType | None = None

    def final_result(result: Any):
        nonlocal decision, final_result_value
        decision = "final_result"
        final_result_value = try_parse_value_to_result_type(result)

    def query_more_agents(agents: list):
        nonlocal decision, next_agents
        decision = "query_more_agents"
        next_agents = [a for a in agents if a in remaining_agents]

    est_summary = ""
    if estimator_result is not None:
        est_summary = (
            f"aggregation_type: {estimator_result.aggregation_type}\n"
            f"sample_coverage: {estimator_result.sample_coverage:.3f}\n"
            f"agents_queried: {estimator_result.agents_queried} / {estimator_result.agents_total}\n"
        )
        if estimator_result.estimated_count is not None:
            est_summary += f"estimated_count: {estimator_result.estimated_count:.1f} [{estimator_result.estimated_count_lower:.1f}, {estimator_result.estimated_count_upper:.1f}]\n"
        if estimator_result.estimated_sum is not None:
            est_summary += f"estimated_sum: {estimator_result.estimated_sum:.3f} [{estimator_result.estimated_sum_lower:.3f}, {estimator_result.estimated_sum_upper:.3f}]\n"
        if estimator_result.estimated_avg is not None:
            est_summary += f"estimated_avg: {estimator_result.estimated_avg:.3f} [{estimator_result.estimated_avg_lower:.3f}, {estimator_result.estimated_avg_upper:.3f}]\n"
        if estimator_result.estimated_min is not None:
            est_summary += f"estimated_min: {estimator_result.estimated_min}\n"
        if estimator_result.estimated_max is not None:
            est_summary += f"estimated_max: {estimator_result.estimated_max}\n"

    code_environment = CodeExecutionEnvironment(
        variables={
            "query": query,
            "result_list": aligned.data,
            "remaining_agents": remaining_agents,
            "pd": pd,
        },
        tools={
            "final_result": final_result,
            "query_more_agents": query_more_agents,
        },
    )

    def custom_tool_handler(
        _context: RunContextWrapper[None],
        _tool_results: list[FunctionToolResult],
    ) -> ToolsToFinalOutputResult:
        is_done = decision == "final_result" or decision == "query_more_agents"
        return ToolsToFinalOutputResult(is_final_output=is_done, final_output=None)

    with own_custom_span(
        "Query Coordinator - Completeness Evaluator",
        input={
            "query": query,
            "result_list_rows": len(aligned.data),
            "result_list_preview": str(aligned.data.head(10).to_string()),
            "estimator_summary": est_summary
            or "Not yet available (fewer than 2 agents queried).",
            "remaining_agents": remaining_agents,
            "remaining_agents_count": len(remaining_agents),
        },
    ):
        await run_agent(
            agent_configuration=get_coordinator_agent_configuration(
                "Completeness Evaluator"
            ),
            prompt_name="execution-coordinator-completeness-evaluator",
            prompt_parameters={
                "query": query,
                "result_list_preview": str(aligned.data.head(10).to_string()),
                "estimator_summary": est_summary
                or "Not yet available (fewer than 2 agents queried).",
                "remaining_agents": ", ".join(remaining_agents)
                if remaining_agents
                else "none",
            },
            tools=[code_environment.as_function_tool()],
            tool_use_behavior=custom_tool_handler,
            reset_tool_choice=False,
            parallel_tool_calls=False,
            tool_choice="required",
            output_type=PartialQueryAnswerDummy,
            get_actual_result=lambda: None,
        )
        append_to_span_output({
            "decision": decision,
            "next_agents": next_agents,
            "next_agents_count": len(next_agents),
            "has_final_result": final_result_value is not None,
        })

    return decision, next_agents, final_result_value


async def answer_step_multiple_agents(
    data_agents: list[DataAgent],
    query_plan_step: MultiAgentQueryPlanStep,
    additional_information: list[AdditionalInformation],
    continue_after_final_result: bool = False,
) -> tuple[PartialQueryAnswer, list[MultiAgentIterationRecord]]:
    query = query_plan_step.query
    all_agents = list(query_plan_step.to_agent)
    agents_total = len(all_agents)

    aligned = AlignedResultList()
    iteration_records: list[MultiAgentIterationRecord] = []
    collected_needs: list = []

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
                    estimator_result = compute_estimator_result(
                        aligned, len(queried), agents_total
                    )
                    record = MultiAgentIterationRecord(
                        step_id=query_plan_step.id,
                        agent_name=agent_name,
                        result_list=aligned.data.copy(),
                        estimator_result=estimator_result,
                        evaluator_decision="query_more_agents",
                        evaluator_next_agents=[a for a in all_agents if a not in queried],
                    )
                    iteration_records.append(record)
                    append_to_span_output({
                        "agent_has_answer": False,
                        "evaluator_decision": "query_more_agents",
                    })
                    continue

                aligned = await align_schema(
                    aligned, agent_name, agent_partial.answer, query
                )

                estimator_result = compute_estimator_result(
                    aligned, len(queried), agents_total
                )

                remaining = [a for a in all_agents if a not in queried]
                (
                    decision,
                    next_agents,
                    final_result_value,
                ) = await _run_completeness_evaluator(
                    aligned, estimator_result, query, remaining
                )

                record = MultiAgentIterationRecord(
                    step_id=query_plan_step.id,
                    agent_name=agent_name,
                    result_list=aligned.data.copy(),
                    estimator_result=estimator_result,
                    evaluator_decision=decision,
                    evaluator_next_agents=next_agents,
                )
                iteration_records.append(record)

                append_to_span_output({
                    "agent_has_answer": True,
                    "aligned_result_rows": len(aligned.data),
                    "evaluator_decision": decision,
                    "evaluator_next_agents": next_agents,
                    "estimator_sample_coverage": estimator_result.sample_coverage
                    if estimator_result is not None
                    else None,
                    "estimator_agents_queried": estimator_result.agents_queried
                    if estimator_result is not None
                    else None,
                })

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
        elif not aligned.data.empty:
            answer.set_answer(aligned.data)

        if collected_needs:
            answer.add_needs(collected_needs)

        return answer, iteration_records

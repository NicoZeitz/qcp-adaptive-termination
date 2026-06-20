from code_execution.code_execution_environment import CodeExecutionEnvironment
from qtypes.aligned_result_list import AlignedResultList
from qtypes.estimator_statistics import EstimatorStatistics
from qtypes.multi_agent_iteration_record import MultiAgentIterationRecord
from qtypes.outcome import EvaluatorDecision
from qtypes.outcome import FinalResultOutcome
from qtypes.outcome import QueryMoreAgentsOutcome
from qtypes.result import ResultType, try_parse_value_to_result_type
from stages.common.agent_configuration import get_coordinator_agent_configuration
from unknown_unknown.estimator import Estimator
from utils.preview import preview_dataframe, result_to_json_preview
from utils.run_agent import run_agent
from utils.span import append_to_span_output, own_custom_span


import pandas as pd
from agents import FunctionToolResult, RunContextWrapper, ToolsToFinalOutputResult


from typing import Any, Callable


def build_estimator_summary(
    estimator_result: EstimatorStatistics, new_unique_count: int
) -> str:
    summary = (
        f"Aggregation type: {estimator_result.aggregation_type}\n"
        f"Agents queried: {estimator_result.agents_queried} /  {estimator_result.agents_total} ({estimator_result.agents_queried / estimator_result.agents_total:.2%}%)\n"
        f"New unique entries added by this agent: {new_unique_count}\n"
        f"Sample Coverage: {estimator_result.sample_coverage:.3f}\n"
    )
    summary += f"Observed {estimator_result.aggregation_type}: {estimator_result.observed_value}\n"
    summary += f"Estimated {estimator_result.aggregation_type}: {estimator_result.estimated_value}\n"
    if (
        estimator_result.estimated_value_lower is not None
        and estimator_result.estimated_value_upper is not None
    ):
        summary += f"Estimated {estimator_result.aggregation_type} confidence interval: [{estimator_result.estimated_value_lower}, {estimator_result.estimated_value_upper}]\n"

    return summary


def build_previous_statistics_summary(
    previous_records: list[MultiAgentIterationRecord],
) -> str:
    if not previous_records:
        return "(none)"
    lines = []
    for record in previous_records:
        decision = record.evaluator_decision or "n/a"
        stats = record.estimator_result
        if stats is None:
            lines.append(
                f"- {record.agent_name}: decision={decision}, no estimator statistics available"
            )
        else:
            line = (
                f"- {record.agent_name}: decision={decision}, "
                f"coverage={stats.sample_coverage:.3f}, "
                f"observed={stats.observed_value}, "
                f"estimated={stats.estimated_value}"
            )
            if (
                stats.estimated_value_lower is not None
                and stats.estimated_value_upper is not None
            ):
                line += f", CI=[{stats.estimated_value_lower}, {stats.estimated_value_upper}]"
            lines.append(line)
    return "\n".join(lines)


async def run_completeness_evaluator(
    aligned_result_list: AlignedResultList,
    estimator: Estimator,
    estimator_statistics: EstimatorStatistics,
    query: str,
    new_unique_count: int,
    previous_records: list[MultiAgentIterationRecord],
) -> QueryMoreAgentsOutcome | FinalResultOutcome:
    decision: EvaluatorDecision | None = None
    final_result_value: ResultType | None = None
    force_result = (
        estimator_statistics.agents_queried == estimator_statistics.agents_total
    )

    def final_result(result: Any):
        nonlocal decision, final_result_value
        final_result_value = try_parse_value_to_result_type(result)
        decision = "final_result"

    def query_more_agents():
        nonlocal decision
        decision = "query_more_agents"

    tools: dict[str, Callable] = {
        "final_result": final_result,
    }
    if not force_result:
        tools["query_more_agents"] = query_more_agents

    code_environment = CodeExecutionEnvironment(
        variables={
            "query": query,
            "result_list": aligned_result_list.data,
            "estimator": estimator,
            "pd": pd,
        },
        tools=tools,
    )

    def custom_tool_handler(
        _context: RunContextWrapper[None],
        _tool_results: list[FunctionToolResult],
    ) -> ToolsToFinalOutputResult:
        return ToolsToFinalOutputResult(
            is_final_output=decision is not None,
            final_output=decision == "final_result",
        )

    estimator_summary = build_estimator_summary(estimator_statistics, new_unique_count)
    previous_statistics_summary = build_previous_statistics_summary(previous_records)
    result_list_preview = preview_dataframe(aligned_result_list.data)

    with own_custom_span(
        "Query Coordinator - Completeness Evaluator",
        input={
            "query": query,
            "result_list_preview": result_list_preview,
            "estimator_statistics": estimator_statistics,
            "previous_records_count": len(previous_records),
        },
    ):
        await run_agent(
            agent_configuration=get_coordinator_agent_configuration(
                "Completeness Evaluator"
            ),
            prompt_name="execution-coordinator-completeness-evaluator",
            prompt_parameters={
                "query": query,
                "result_list_preview": result_list_preview,
                "estimator_summary": estimator_summary,
                "previous_statistics_summary": previous_statistics_summary,
                "force_result": force_result,
            },
            tools=[code_environment.as_function_tool()],
            tool_use_behavior=custom_tool_handler,
            reset_tool_choice=False,
            parallel_tool_calls=False,
            tool_choice="required",
            output_type=bool,
        )
        append_to_span_output(
            {
                "decision": decision,
                "final_result": result_to_json_preview(final_result_value)
                if final_result_value is not None
                else None,
            }
        )

    if decision == "final_result":
        return FinalResultOutcome(result=final_result_value)
    else:
        return QueryMoreAgentsOutcome(result_list=aligned_result_list)

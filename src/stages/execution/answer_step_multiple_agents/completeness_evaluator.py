from code_execution.code_execution_environment import CodeExecutionEnvironment
from qtypes.aligned_result_list import AlignedResultList
from qtypes.estimator_statistics import EstimatorStatistics
from qtypes.execution_config import EstimatorMode, ExecutionConfig
from qtypes.outcome import EvaluatorDecision
from qtypes.outcome import FinalResultOutcome
from qtypes.outcome import QueryMoreAgentsOutcome
from qtypes.result import ResultType, try_parse_value_to_result_type
from stages.common.agent_configuration import get_coordinator_agent_configuration
from utils.preview import preview_dataframe, result_to_json_preview
from utils.run_agent import run_agent
from utils.span import append_to_span_output, own_custom_span


import pandas as pd
from agents import FunctionToolResult, RunContextWrapper, ToolsToFinalOutputResult


from typing import Any, Callable


def build_estimator_summary(
    estimator_statistics: EstimatorStatistics,
    *,
    include_aggregation_type: bool = True,
) -> str | None:
    config = ExecutionConfig.get()
    mode = config.estimator_mode
    if mode == EstimatorMode.NONE:
        return None

    include_ci = config.include_confidence_intervals
    include_est_count = mode in (EstimatorMode.FULL, EstimatorMode.COUNT_ONLY)
    include_obs_count = include_est_count or mode == EstimatorMode.OBSERVED_ONLY
    include_est_value = mode in (EstimatorMode.FULL, EstimatorMode.VALUE_ONLY)
    include_obs_value = include_est_value or mode == EstimatorMode.OBSERVED_ONLY

    agg = estimator_statistics.aggregation_type
    lines = []
    if include_aggregation_type:
        lines.append(f"Aggregation type: {agg}")
    lines.append(
        f"Agents queried: {estimator_statistics.agents_queried} /  {estimator_statistics.agents_total} ({estimator_statistics.agents_queried / estimator_statistics.agents_total:.2%})"
    )
    lines.append(
        f"New unique entries added by this agent: {estimator_statistics.new_unique_count}"
    )
    if estimator_statistics.sample_coverage is not None:
        lines.append(f"Sample Coverage: {estimator_statistics.sample_coverage:.3f}")

    if include_obs_count:
        lines.append(f"Observed count: {estimator_statistics.observed_count}")
    if include_est_count and estimator_statistics.estimated_count is not None:
        lines.append(f"Estimated count: {estimator_statistics.estimated_count:.3f}")
        if (
            include_ci
            and estimator_statistics.estimated_count_lower is not None
            and estimator_statistics.estimated_count_upper is not None
        ):
            lines.append(
                f"Estimated count confidence interval: [{estimator_statistics.estimated_count_lower:.3f}, {estimator_statistics.estimated_count_upper:.3f}]"
            )

    if include_obs_value and estimator_statistics.observed_value is not None:
        lines.append(f"Observed {agg}: {estimator_statistics.observed_value}")
    if include_est_value and estimator_statistics.estimated_value is not None:
        lines.append(f"Estimated {agg}: {estimator_statistics.estimated_value:.3f}")
        if (
            include_ci
        ):
            estimated_value_lower = estimator_statistics.estimated_value_lower or float('-inf')
            estimated_value_upper = estimator_statistics.estimated_value_upper or float('inf')
            lines.append(
                f"Estimated {agg} confidence interval: [{estimated_value_lower:.3f}, {estimated_value_upper:.3f}]"
            )
    return "\n".join(lines)


def build_previous_statistics_summary(
    previous_estimator_statistics: list[EstimatorStatistics],
) -> str | None:
    if (
        not previous_estimator_statistics
        or ExecutionConfig.get().estimator_mode == EstimatorMode.NONE
    ):
        return None

    lines = []
    for estimator_statistic in reversed(previous_estimator_statistics):
        summary = build_estimator_summary(
            estimator_statistic, include_aggregation_type=False
        )
        lines.append(summary)
    return "\n\n".join(lines)


async def run_completeness_evaluator(
    aligned_result_list: AlignedResultList,
    estimator_statistics: EstimatorStatistics,
    query: str,
    previous_estimator_statistics: list[EstimatorStatistics],
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

    estimator_summary = build_estimator_summary(estimator_statistics)
    previous_statistics_summary = build_previous_statistics_summary(
        previous_estimator_statistics
    )
    result_list_preview = preview_dataframe(aligned_result_list.data)

    with own_custom_span(
        "Query Coordinator - Completeness Evaluator",
        input={
            "query": query,
            "result_list_preview": result_list_preview,
            "estimator_statistics": estimator_statistics,
            "previous_estimator_statistics_count": len(previous_estimator_statistics),
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
        return FinalResultOutcome(
            result=final_result_value, result_list=aligned_result_list
        )
    else:
        return QueryMoreAgentsOutcome(result_list=aligned_result_list)

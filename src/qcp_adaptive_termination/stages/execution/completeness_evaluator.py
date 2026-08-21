from collections.abc import Callable
from typing import Any

import pandas as pd
from agents import (
    FunctionToolResult,
    RunContextWrapper,
    ToolsToFinalOutputResult,
)
from pydantic import BaseModel, ConfigDict

import qcp_adaptive_termination.hooks as hooks
from qcp_adaptive_termination.code_execution.code_execution_environment import (
    CodeExecutionEnvironment,
)
from qcp_adaptive_termination.hooks import HookEvent
from qcp_adaptive_termination.llm import runner
from qcp_adaptive_termination.models import (
    AgentRole,
    AggregationType,
    AlignedResultList,
    EstimatorMode,
    EstimatorStatistics,
    SerializedOptionalResultType,
    result_type,
)
from qcp_adaptive_termination.models.execution import (
    EvaluatorDecision,
    FinalResultOutcome,
    QueryMoreAgentsOutcome,
)
from qcp_adaptive_termination.observability import context, tracing


class CompletenessEvaluatorResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    decision: EvaluatorDecision | None = None
    result: SerializedOptionalResultType = None


def build_estimator_summary(estimator_statistics: EstimatorStatistics) -> str | None:
    config = context.get_config()
    mode = config.estimator_mode
    if mode == EstimatorMode.NONE:
        return None

    include_ci = config.include_confidence_intervals
    include_est_count = mode in (EstimatorMode.FULL, EstimatorMode.COUNT_ONLY)
    include_obs_count = include_est_count or mode == EstimatorMode.OBSERVED_ONLY
    include_est_value = mode in (EstimatorMode.FULL, EstimatorMode.VALUE_ONLY)
    include_obs_value = include_est_value or mode == EstimatorMode.OBSERVED_ONLY

    lines = []
    lines.append(
        f"Agents queried: {estimator_statistics.agents_queried} /  {estimator_statistics.agents_total}"
        f" ({estimator_statistics.agents_queried / estimator_statistics.agents_total:.2%})"
    )
    lines.append(f"New unique entries added by this agent: {estimator_statistics.new_unique_count}")

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
                f"Estimated count confidence interval: "
                f"[{estimator_statistics.estimated_count_lower:.3f}, {estimator_statistics.estimated_count_upper:.3f}]"
            )

    for agg in estimator_statistics.aggregation_types:
        if agg == AggregationType.COUNT:
            continue

        lines.append(f"Aggregation type: {agg}")

        observed_value = None
        estimated_value = None
        estimated_value_lower = None
        estimated_value_upper = None

        if agg == AggregationType.SUM:
            observed_value = estimator_statistics.observed_value_sum
            estimated_value = estimator_statistics.estimated_value_sum
            estimated_value_lower = estimator_statistics.estimated_value_sum_lower
            estimated_value_upper = estimator_statistics.estimated_value_sum_upper
        elif agg == AggregationType.AVG:
            observed_value = estimator_statistics.observed_value_avg
            estimated_value = estimator_statistics.estimated_value_avg
            estimated_value_lower = estimator_statistics.estimated_value_avg_lower
            estimated_value_upper = estimator_statistics.estimated_value_avg_upper
        elif agg == AggregationType.MIN:
            observed_value = estimator_statistics.observed_value_min
            estimated_value = estimator_statistics.estimated_value_min
        elif agg == AggregationType.MAX:
            observed_value = estimator_statistics.observed_value_max
            estimated_value = estimator_statistics.estimated_value_max

        if include_obs_value and observed_value is not None:
            lines.append(f"Observed {agg}: {observed_value}")

        if include_est_value and estimated_value is not None:
            lines.append(f"Estimated {agg}: {estimated_value:.3f}")
            if include_ci and (estimated_value_lower is not None or estimated_value_upper is not None):
                estimated_value_lower = estimated_value_lower or float("-inf")
                estimated_value_upper = estimated_value_upper or float("inf")
                lines.append(
                    f"Estimated {agg} confidence interval: [{estimated_value_lower:.3f}, {estimated_value_upper:.3f}]"
                )

    return "\n".join(lines)


def build_previous_statistics_summary(
    previous_estimator_statistics: list[EstimatorStatistics],
) -> str | None:
    if not previous_estimator_statistics or context.get_config().estimator_mode == EstimatorMode.NONE:
        return None

    lines = []
    for estimator_statistic in reversed(previous_estimator_statistics):
        summary = build_estimator_summary(estimator_statistic)
        lines.append(summary)
    return "\n\n".join(line for line in lines if line is not None)


async def run_completeness_evaluator(
    aligned_result_list: AlignedResultList,
    estimator_statistics: EstimatorStatistics,
    query: str,
    previous_estimator_statistics: list[EstimatorStatistics],
    merge_task: str,
) -> QueryMoreAgentsOutcome | FinalResultOutcome:
    force_result = estimator_statistics.agents_queried == estimator_statistics.agents_total
    return await hooks.call_override(
        HookEvent.COMPLETENESS_EVALUATOR_OVERRIDE,
        default=lambda: _evaluate_via_llm(
            aligned_result_list, estimator_statistics, query, previous_estimator_statistics, merge_task, force_result
        ),
        aligned_result_list=aligned_result_list,
        estimator_statistics=estimator_statistics,
        query=query,
        previous_estimator_statistics=previous_estimator_statistics,
        merge_task=merge_task,
        force_result=force_result,
    )


async def _evaluate_via_llm(
    aligned_result_list: AlignedResultList,
    estimator_statistics: EstimatorStatistics,
    query: str,
    previous_estimator_statistics: list[EstimatorStatistics],
    merge_task: str,
    force_result: bool,
) -> QueryMoreAgentsOutcome | FinalResultOutcome:
    evaluator_result = CompletenessEvaluatorResult()

    def final_result(result: Any):
        evaluator_result.decision = "final_result"
        evaluator_result.result = result_type.try_parse_value(result)

    def query_more_agents():
        evaluator_result.decision = "query_more_agents"

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
            is_final_output=evaluator_result.decision is not None,
            final_output=evaluator_result,
        )

    estimator_summary = build_estimator_summary(estimator_statistics)
    previous_statistics_summary = build_previous_statistics_summary(previous_estimator_statistics)
    result_list_preview = result_type.serialize(aligned_result_list.data, context="preview")

    with tracing.own_span(
        "Query Coordinator - Completeness Evaluator",
        input={
            "query": query,
            "result_list_preview": result_list_preview,
            "estimator_statistics": estimator_statistics,
            "previous_estimator_statistics_count": len(previous_estimator_statistics),
        },
    ):
        evaluator_result = await runner.run_agent(
            role=AgentRole.COORDINATOR_COMPLETENESS_EVALUATOR,
            prompt_parameters={
                "query": query,
                "result_list_preview": result_list_preview,
                "estimator_summary": estimator_summary,
                "previous_statistics_summary": previous_statistics_summary,
                "force_result": force_result,
                "merge_task": merge_task,
            },
            tools=[code_environment.as_function_tool()],
            tool_use_behavior=custom_tool_handler,
            reset_tool_choice=False,
            parallel_tool_calls=False,
            tool_choice="required",
            output_type=CompletenessEvaluatorResult,
        )
        tracing.append_to_span_output(
            {
                "decision": evaluator_result.decision,
                "final_result": result_type.serialize(evaluator_result.result, context="preview")
                if evaluator_result.result is not None
                else None,
            }
        )

    if evaluator_result.decision == "final_result":
        if evaluator_result.result is None:
            raise RuntimeError("Completeness evaluator returned final_result decision but no result")
        return FinalResultOutcome(
            result=evaluator_result.result,
            result_list=aligned_result_list,
        )
    else:
        return QueryMoreAgentsOutcome(result_list=aligned_result_list)

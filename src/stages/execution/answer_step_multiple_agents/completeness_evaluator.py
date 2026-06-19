from code_execution.code_execution_environment import CodeExecutionEnvironment
from qtypes.aligned_result_list import AlignedResultList
from qtypes.estimator_result import EstimatorResult
from qtypes.multi_agent_iteration_record import EvaluatorDecision
from qtypes.partial_query_answer import PartialQueryAnswerDummy
from qtypes.result import ResultType, try_parse_value_to_result_type
from stages.common.agent_configuration import get_coordinator_agent_configuration
from utils.preview import preview_dataframe
from utils.run_agent import run_agent
from utils.span import append_to_span_output, own_custom_span


import pandas as pd
from agents import FunctionToolResult, RunContextWrapper, ToolsToFinalOutputResult


from typing import Any


def build_estimator_summary(estimator_result: EstimatorResult | None) -> str:
    if estimator_result is None:
        return ""
    summary = (
        f"aggregation_type: {estimator_result.aggregation_type}\n"
        f"sample_coverage: {estimator_result.sample_coverage:.3f}\n"
        f"agents_queried: {estimator_result.agents_queried} / {estimator_result.agents_total}\n"
    )
    if estimator_result.estimated_count is not None:
        summary += f"estimated_count: {estimator_result.estimated_count:.1f} [{estimator_result.estimated_count_lower:.1f}, {estimator_result.estimated_count_upper:.1f}]\n"
    if estimator_result.estimated_sum is not None:
        summary += f"estimated_sum: {estimator_result.estimated_sum:.3f} [{estimator_result.estimated_sum_lower:.3f}, {estimator_result.estimated_sum_upper:.3f}]\n"
    if estimator_result.estimated_avg is not None:
        summary += f"estimated_avg: {estimator_result.estimated_avg:.3f} [{estimator_result.estimated_avg_lower:.3f}, {estimator_result.estimated_avg_upper:.3f}]\n"
    if estimator_result.estimated_min is not None:
        summary += f"estimated_min: {estimator_result.estimated_min}\n"
    if estimator_result.estimated_max is not None:
        summary += f"estimated_max: {estimator_result.estimated_max}\n"
    return summary


async def run_completeness_evaluator(
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

    est_summary = build_estimator_summary(estimator_result)

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
            "result_list_preview": preview_dataframe(aligned.data),
            "estimator_summary": est_summary
            or "Not yet available (fewer than 2 agents queried).",
            "estimator_result": estimator_result.__dict__
            if estimator_result is not None
            else None,
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
                "result_list_preview": preview_dataframe(aligned.data),
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
        append_to_span_output(
            {
                "decision": decision,
                "next_agents": next_agents,
                "next_agents_count": len(next_agents),
                "has_final_result": final_result_value is not None,
            }
        )

    return decision, next_agents, final_result_value

"""
Split completeness decision: deterministic estimator + LLM world-knowledge stop.

Implements PLAN.md step 2/3: the completeness evaluator is split into
- a deterministic estimator stop (world-knowledge-free, based on worst-case completeness)
- an LLM world-knowledge stop agent (sees only query + result list)
- a result-producing agent (produces the final answer once a stop is decided)
"""

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
    AlignedResultList,
    EstimatorStatistics,
    SerializedOptionalResultType,
    StepBudget,
    StopReason,
    result_type,
)
from qcp_adaptive_termination.models.execution import (
    EstimatorDecision,
    FinalResultOutcome,
    QueryMoreAgentsOutcome,
)
from qcp_adaptive_termination.observability import tracing
from qcp_adaptive_termination.stages.execution.multi_agent.deterministic_estimator import (
    run_deterministic_estimator,
)


class WorldKnowledgeStopResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    stop: bool = False


class ResultProducerResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    result: SerializedOptionalResultType = None


@hooks.hook(HookEvent.EXECUTION_EXECUTE_QUERY_PLAN_EXECUTE_STEP_ANSWER_STEP_MULTIPLE_AGENTS_COMPLETENESS_ESTIMATOR)
async def run_estimator_decision(
    aligned_result_list: AlignedResultList,
    estimator_statistics: EstimatorStatistics,
    step_budget: StepBudget | None,
    agents_queried: int,
) -> EstimatorDecision:
    """Deterministic stop decision from the estimator (overridable)."""
    return run_deterministic_estimator(estimator_statistics, step_budget, agents_queried)


@hooks.hook(HookEvent.EXECUTION_EXECUTE_QUERY_PLAN_EXECUTE_STEP_ANSWER_STEP_MULTIPLE_AGENTS_COMPLETENESS_LLM)
async def run_world_knowledge_stop(
    aligned_result_list: AlignedResultList,
    query: str,
) -> bool:
    """LLM world-knowledge stop: sees only query + result list, no estimator stats."""
    result = WorldKnowledgeStopResult()

    def stop():
        result.stop = True

    def continue_querying():
        result.stop = False

    code_environment = CodeExecutionEnvironment(
        variables={
            "query": query,
            "result_list": aligned_result_list.data,
            "pd": pd,
        },
        tools={
            "stop": stop,
            "continue_querying": continue_querying,
        },
    )

    def custom_tool_handler(
        _context: RunContextWrapper[None],
        _tool_results: list[FunctionToolResult],
    ) -> ToolsToFinalOutputResult:
        return ToolsToFinalOutputResult(
            is_final_output=result.stop is not None,
            final_output=result,
        )

    result_list_preview = result_type.serialize(aligned_result_list.data, context="preview")

    with tracing.own_span(
        "Coordinator - World Knowledge Stop",
        input={
            "query": query,
            "result_list_preview": result_list_preview,
        },
    ):
        result = await runner.run_agent(
            role=AgentRole.COORDINATOR_WORLD_KNOWLEDGE_STOP,
            prompt_parameters={
                "query": query,
                "result_list_preview": result_list_preview,
            },
            tools=[code_environment.as_function_tool()],
            tool_use_behavior=custom_tool_handler,
            reset_tool_choice=False,
            parallel_tool_calls=False,
            tool_choice="required",
            output_type=WorldKnowledgeStopResult,
        )
        tracing.append_to_span_output({"stop": result.stop})

    return result.stop


@hooks.hook(HookEvent.EXECUTION_EXECUTE_QUERY_PLAN_EXECUTE_STEP_ANSWER_STEP_MULTIPLE_AGENTS_COMPLETENESS)
async def run_completeness_evaluator(
    aligned_result_list: AlignedResultList,
    estimator_statistics: EstimatorStatistics,
    query: str,
    merge_task: str,
    step_budget: StepBudget | None,
    agents_queried: int,
    force_result: bool,
) -> QueryMoreAgentsOutcome | FinalResultOutcome:
    """
    Combined completeness decision: deterministic estimator, then LLM world knowledge.

    If the estimator says stop, or the LLM world-knowledge agent says the result is
    complete, produce the final result via the result-producing agent. Otherwise
    continue querying.
    """
    if force_result:
        return await _produce_result(
            aligned_result_list, estimator_statistics, query, merge_task, StopReason.ALL_AGENTS
        )

    estimator_decision = await run_estimator_decision(
        aligned_result_list, estimator_statistics, step_budget, agents_queried
    )

    if estimator_decision.stop:
        return await _produce_result(aligned_result_list, estimator_statistics, query, merge_task, StopReason.ESTIMATOR)

    llm_stop = await run_world_knowledge_stop(aligned_result_list, query)
    if llm_stop:
        return await _produce_result(aligned_result_list, estimator_statistics, query, merge_task, StopReason.LLM)

    return QueryMoreAgentsOutcome(result_list=aligned_result_list)


async def _produce_result(
    aligned_result_list: AlignedResultList,
    estimator_statistics: EstimatorStatistics,
    query: str,
    merge_task: str,
    stop_reason: StopReason,
) -> FinalResultOutcome:
    """Produce the final answer via the result-producing agent."""
    result = ResultProducerResult()

    def final_result(result_value: Any):
        result.result = result_type.try_parse_value(result_value)

    code_environment = CodeExecutionEnvironment(
        variables={
            "query": query,
            "result_list": aligned_result_list.data,
            "pd": pd,
        },
        tools={
            "final_result": final_result,
        },
    )

    def custom_tool_handler(
        _context: RunContextWrapper[None],
        _tool_results: list[FunctionToolResult],
    ) -> ToolsToFinalOutputResult:
        return ToolsToFinalOutputResult(
            is_final_output=result.result is not None,
            final_output=result,
        )

    result_list_preview = result_type.serialize(aligned_result_list.data, context="preview")

    with tracing.own_span(
        "Coordinator - Result Producer",
        input={
            "query": query,
            "result_list_preview": result_list_preview,
            "estimator_statistics": estimator_statistics,
        },
    ):
        result = await runner.run_agent(
            role=AgentRole.COORDINATOR_RESULT_PRODUCER,
            prompt_parameters={
                "query": query,
                "result_list_preview": result_list_preview,
                "merge_task": merge_task,
            },
            tools=[code_environment.as_function_tool()],
            tool_use_behavior=custom_tool_handler,
            reset_tool_choice=False,
            parallel_tool_calls=False,
            tool_choice="required",
            output_type=ResultProducerResult,
        )
        tracing.append_to_span_output({"result": result.result})

    if result.result is None:
        raise RuntimeError("Result producer returned no result")
    return FinalResultOutcome(
        result=result.result,
        result_list=aligned_result_list,
        stop_reason=stop_reason,
    )

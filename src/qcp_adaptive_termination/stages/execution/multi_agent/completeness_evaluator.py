"""
Split completeness decision: deterministic estimator + LLM world-knowledge stop.

Implements PLAN.md step 2/3: the completeness evaluator is split into
- a deterministic estimator stop (world-knowledge-free, based on worst-case completeness)
- an LLM world-knowledge stop agent (sees only query + result list)
- a result-producing agent (produces the final answer once a stop is decided)
"""

import math

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
from qcp_adaptive_termination.estimator import Estimator
from qcp_adaptive_termination.hooks import HookEvent
from qcp_adaptive_termination.llm import runner
from qcp_adaptive_termination.models import (
    AgentRole,
    AlignedResultList,
    StepBudget,
    StopReason,
    result_type,
)
from qcp_adaptive_termination.models.execution import StopDecision
from qcp_adaptive_termination.observability import context, tracing


class WorldKnowledgeStopResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    stop: bool = False


@hooks.hook(HookEvent.EXECUTION_EXECUTE_QUERY_PLAN_EXECUTE_STEP_ANSWER_STEP_MULTIPLE_AGENTS_COMPLETENESS_ESTIMATOR)
def run_deterministic_estimator(
    estimator: Estimator,
    step_budget: StepBudget | None,
    agents_queried: int,
) -> bool:
    """
    Decide whether to stop based on worst-case completeness.

    Returns True to stop, False to keep querying. Computes only what the stop
    rule needs: observed count and the upper bound of the estimated count.
    """
    config = context.get_config()

    # Warm-up: never stop deterministically before enough agents have been queried.
    if agents_queried <= config.warmup_agents:
        tracing.indented_logging(f"Estimator: warm-up ({agents_queried} <= {config.warmup_agents} agents) — continue")
        return False

    # No budget (e.g. find queries or no multi-agent budget) → no deterministic stop.
    if step_budget is None:
        return False

    observed = estimator.sample_count()
    if observed == 0:
        return False

    # Worst-case completeness: observed / N_hat_upper(epsilon_step).
    upper = estimator.estimated_count_upper_bound()
    if not math.isfinite(upper) or upper <= 0:
        return False

    c_worst = observed / upper
    stop = c_worst >= step_budget.t_step

    tracing.indented_logging(
        f"Estimator: C_worst={c_worst:.4f} (observed={observed}, N_hat_upper={upper:.2f}) "
        f"threshold={step_budget.t_step:.4f} → {'STOP' if stop else 'continue'}"
    )
    return stop


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
    estimator: Estimator | None,
    query: str,
    merge_task: str,
    step_budget: StepBudget | None,
    agents_queried: int,
    force_result: bool,
) -> StopDecision | None:
    """
    Combined completeness decision: deterministic estimator, then LLM world knowledge.

    Returns a `StopDecision` when the result is complete (the caller produces the
    final answer), or ``None`` to keep querying.
    """
    if force_result:
        return StopDecision(stop_reason=StopReason.ALL_AGENTS)

    if estimator is not None and run_deterministic_estimator(estimator, step_budget, agents_queried):
        return StopDecision(stop_reason=StopReason.ESTIMATOR)

    llm_stop = await run_world_knowledge_stop(aligned_result_list, query)
    if llm_stop:
        return StopDecision(stop_reason=StopReason.LLM)

    return None

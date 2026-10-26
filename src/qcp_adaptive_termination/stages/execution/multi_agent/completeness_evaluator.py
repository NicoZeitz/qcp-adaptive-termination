"""Completeness decision: deterministic estimator, then LLM world-knowledge stop."""

import logging
import math
from typing import Any

import pandas as pd
from agents import (
    FunctionToolResult,
    RunContextWrapper,
    ToolsToFinalOutputResult,
    TResponseInputItem,
)
from agents.exceptions import MaxTurnsExceeded

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


def _result_list_for_llm(aligned_result_list: AlignedResultList) -> pd.DataFrame:
    """Deduplicated result list without the source-agent column, for the LLM."""
    schema = aligned_result_list.schema_metadata
    data = aligned_result_list.data
    if schema.source_agent_column in data.columns:
        data = data.drop(columns=[schema.source_agent_column])
    keys = [c for c in schema.keys if c in data.columns]
    return data.drop_duplicates(subset=keys) if keys else data


def _estimator_stats(estimator: Estimator | None) -> str | None:
    """Render the estimator's unknown-unknowns for the LLM stopper prompt.

    Returns:
        (str | None): a compact stats block, or None when there is no estimator
    """
    if estimator is None:
        return None
    observed = estimator.sample_count()
    estimated = estimator.estimated_count()
    lower, upper = estimator.estimated_count_bounds()

    def ratio(num: float, den):
        """Ratio as a fixed-width string, or n/a when the denominator is zero."""
        return f"{num / den:.3f}" if den and den > 0 else "n/a"

    return (
        f"\n\n## Estimator (unknown-unknowns) stats\n"
        f"Observed unique: {observed}\n"
        f"Estimated total: {estimated:.1f}\n"
        f"Estimated bound: [{lower:.1f}, {upper:.1f}]\n"
        f"Sample coverage: {estimator.sample_coverage():.3f}\n"
        f"Observed / lower (best-case completeness): {ratio(observed, lower)}\n"
        f"Observed / middle (estimated completeness): {ratio(observed, estimated)}\n"
        f"Observed / upper (worst-case completeness): {ratio(observed, upper)}\n"
    )


@hooks.hook(HookEvent.EXECUTION_EXECUTE_QUERY_PLAN_EXECUTE_STEP_ANSWER_STEP_MULTIPLE_AGENTS_COMPLETENESS_ESTIMATOR)
def run_deterministic_estimator(
    estimator: Estimator,
    step_budget: StepBudget | None,
    agents_queried: int,
) -> bool:
    """Decide whether to stop based on worst-case completeness.

    Args:
        estimator: the current estimator
        step_budget: the step's completeness budget
        agents_queried: how many agents have been queried

    Returns:
        (bool): True to stop, False to keep querying
    """
    config = context.get_config()

    # Warm-up: never stop deterministically before enough agents have been queried.
    if agents_queried <= config.warmup_agents:
        logging.info("Estimator: warm-up (%s <= %s agents) — continue", agents_queried, config.warmup_agents)
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

    logging.info(
        "Estimator: C_worst=%.4f (observed=%s, N_hat_upper=%.2f) threshold=%.4f → %s",
        c_worst,
        observed,
        upper,
        step_budget.t_step,
        "STOP" if stop else "continue",
    )
    print(
        f"[COMPLETENESS] step={step_budget.step_id if step_budget else '?'} agents_queried={agents_queried} "
        f"observed={observed} upper={upper} c_worst={c_worst:.4f} stop={'STOP' if stop else 'continue'}",
        flush=True,
    )
    return stop


@hooks.hook(HookEvent.EXECUTION_EXECUTE_QUERY_PLAN_EXECUTE_STEP_ANSWER_STEP_MULTIPLE_AGENTS_COMPLETENESS_LLM)
async def run_world_knowledge_stop(
    aligned_result_list: AlignedResultList,
    query: str,
    label: str | None = None,
    history: list[TResponseInputItem] | None = None,
    estimator: Estimator | None = None,
) -> bool:
    """Ask the LLM whether the result list is complete using world knowledge.

    ``label`` identifies this evaluation in traces; ``history`` carries prior
    turns forward; ``estimator`` supplies unknown-unknown stats when configured.

    Args:
        aligned_result_list: the aligned result list
        query: the step's query
        label: identifier for trace and print association
        history: prior conversation items to continue from
        estimator: the current estimator for unknown-unknown stats

    Returns:
        (bool): True to stop, False to keep querying
    """
    config = context.get_config()
    result: bool | None = None

    def stop():
        """Signal that the result list is complete."""
        nonlocal result
        result = True

    def continue_querying():
        """Signal that more agents should be queried."""
        nonlocal result
        result = False

    llm_result_list = _result_list_for_llm(aligned_result_list)
    result_list_preview = result_type.serialize(llm_result_list, context="preview")
    prompt_parameters: dict[str, Any] = {
        "query": query,
        "result_list_preview": result_list_preview,
    }
    if config.completeness_show_estimator_stats or config.long_list_behavior == "estimator_reasoning":
        stats = _estimator_stats(estimator)
        if stats:
            prompt_parameters["estimator_stats"] = stats
    if config.completeness_summarize_history and history:
        prompt_parameters["is_continuation"] = True

    code_environment = CodeExecutionEnvironment(
        variables={
            "query": query,
            "result_list": llm_result_list,
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
        """Expose the stop decision as final output once decided."""
        return ToolsToFinalOutputResult(
            is_final_output=result is not None,
            final_output=result,
        )

    with tracing.own_span(
        "Coordinator - World Knowledge Stop",
        input={
            "query": query,
            "result_list_preview": result_list_preview,
        },
    ):
        max_turns_exceeded = False
        run_result = None
        try:
            run_result = await runner.run_agent(
                role=AgentRole.EXECUTION_COORDINATOR_WORLD_KNOWLEDGE_STOP,
                prompt_parameters=prompt_parameters,
                tools=[code_environment.as_function_tool()],
                tool_use_behavior=custom_tool_handler,
                reset_tool_choice=False,
                parallel_tool_calls=False,
                tool_choice="required",
                output_type=bool,
                previous_input=history,
                label=label,
            )
            result = run_result.output
        except MaxTurnsExceeded as e:
            max_turns_exceeded = True
            result = None
            logging.warning("[LLM] completeness reached max_turns without deciding — forcing continue (%s)", e)
        if history is not None and run_result is not None:
            history[:] = run_result.continuation_input
        tracing.append_to_span_output({"stop": result, "max_turns_exceeded": max_turns_exceeded})

    return bool(result)


@hooks.hook(HookEvent.EXECUTION_EXECUTE_QUERY_PLAN_EXECUTE_STEP_ANSWER_STEP_MULTIPLE_AGENTS_COMPLETENESS)
async def run_completeness_evaluator(
    aligned_result_list: AlignedResultList,
    estimator: Estimator | None,
    query: str,
    merge_task: str,
    step_budget: StepBudget | None,
    agents_queried: int,
    force_result: bool,
    history: list[TResponseInputItem] | None = None,
) -> StopDecision | None:
    """Return the combined completeness decision for a step.

    Args:
        aligned_result_list: the aligned result list
        estimator: the current estimator
        query: the step's query
        merge_task: how agent results are merged
        step_budget: the step's completeness budget
        agents_queried: how many agents have been queried
        force_result: whether to stop because all agents were queried
        history: prior conversation items to continue from

    Returns:
        (StopDecision | None): the stop decision, or None to keep querying
    """
    observed = estimator.sample_count() if estimator is not None else None

    if force_result:
        print(
            f"[COMPLETENESS] step={step_budget.step_id if step_budget else '?'} agents_queried={agents_queried} "
            f"observed={observed} decision=all_agents",
            flush=True,
        )
        return StopDecision(stop_reason=StopReason.ALL_AGENTS)

    if aligned_result_list.data.empty:
        # No samples yet: neither the deterministic estimator nor the LLM
        # world-knowledge stop can judge completeness on an empty result list.
        # Keep querying more agents.
        print(
            f"[COMPLETENESS] step={step_budget.step_id if step_budget else '?'} agents_queried={agents_queried} "
            f"observed={observed} decision=continue (no samples)",
            flush=True,
        )
        return None

    if estimator is not None and run_deterministic_estimator(estimator, step_budget, agents_queried):
        print(
            f"[COMPLETENESS] step={step_budget.step_id if step_budget else '?'} agents_queried={agents_queried} "
            f"observed={observed} decision=estimator",
            flush=True,
        )
        return StopDecision(stop_reason=StopReason.ESTIMATOR)

    label = f"COMPLETENESS step={step_budget.step_id if step_budget else '?'} agents_queried={agents_queried}"
    config = context.get_config()

    # Long-list behavior: when the deduplicated result list is long, the LLM
    # should either be skipped entirely (prevent) or given the estimator's
    # unknown-unknown stats to reason with (estimator_reasoning).
    list_len = len(_result_list_for_llm(aligned_result_list))
    if list_len > config.stop_llm_max_list_items and config.long_list_behavior == "prevent":
        print(
            f"[COMPLETENESS] step={step_budget.step_id if step_budget else '?'} agents_queried={agents_queried} "
            f"observed={observed} decision=continue (long list {list_len} > {config.stop_llm_max_list_items}, "
            f"LLM prevented)",
            flush=True,
        )
        return None
    # estimator_reasoning: fall through with estimator stats fed to the LLM.

    llm_stop = await run_world_knowledge_stop(
        aligned_result_list, query, label=label, history=history, estimator=estimator
    )
    print(
        f"[COMPLETENESS] step={step_budget.step_id if step_budget else '?'} agents_queried={agents_queried} "
        f"observed={observed} decision={'llm' if llm_stop else 'continue'}",
        flush=True,
    )
    if llm_stop:
        return StopDecision(stop_reason=StopReason.LLM)

    return None

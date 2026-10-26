"""Synthesis stage: turn recorded uncertainty into a user-facing answer."""

from functools import reduce
from operator import mul

import qcp_adaptive_termination.hooks as hooks
from qcp_adaptive_termination.hooks import HookEvent
from qcp_adaptive_termination.models import StepUncertainty, result_type
from qcp_adaptive_termination.models.query_plan import QueryPlan, QueryPlanStep, SingleAgentQueryPlanStep
from qcp_adaptive_termination.models.result_type import SerializedResultType
from qcp_adaptive_termination.models.stages import Stage, SynthesizedResult
from qcp_adaptive_termination.observability import tracing


def _fmt(value: float | int | None) -> str:
    """Format a numeric value to two decimals (None when the value is None)."""
    if value is None:
        return "None"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return f"{value:.2f}"


def _format_value(key: str, observed: float | None, point: float | None, lo: float | None, hi: float | None) -> str:
    """Format one aggregation's observed and estimated values as a bullet item."""
    if key in ("min", "max"):
        if point is not None:
            return f"  * Estimated {key}: {_fmt(point)} (confirmed)"
        return f"  * Estimated {key}: {_fmt(observed)} (observed only)"
    if point is None:
        return f"  * Estimated {key}: {_fmt(observed)} (observed only)"
    if lo is not None and hi is not None:
        return f"  * Estimated {key}: {_fmt(point)} ({_fmt(lo)}–{_fmt(hi)}), observed: {_fmt(observed)}"
    if observed is not None:
        return f"  * Estimated {key}: {_fmt(point)}, observed: {_fmt(observed)}"
    return f"  * Estimated {key}: {_fmt(point)}"


def _format_step(summary: str, step: QueryPlanStep, su: StepUncertainty | None) -> str:
    """Append the human-readable uncertainty block for one step to the summary."""
    summary += f"* Step {step.id}: {step.query}\n"

    completeness = su.completeness if su is not None else 1.0
    completeness_line = f"  * Completeness: {completeness:.1%}"
    if isinstance(step, SingleAgentQueryPlanStep):
        completeness_line += " (SingleAgentStep)"
    elif su is not None and su.completeness_lower is not None and su.completeness_upper is not None:
        completeness_line += f" ({su.completeness_lower:.1%}–{su.completeness_upper:.1%})"
    summary += completeness_line + "\n"

    if su is None:
        return summary

    if su.stop_reason is not None:
        summary += f"  * Stop reason: {su.stop_reason.value}\n"

    if su.completeness < 1.0 and su.estimated_count is not None:
        lo = su.estimated_count_lower
        hi = su.estimated_count_upper
        summary += f"  * Estimated count: {_fmt(su.estimated_count)}"
        if lo is not None and hi is not None:
            summary += f" ({_fmt(lo)}–{_fmt(hi)})"
        summary += f", observed: {_fmt(su.observed_count)}\n"

    for key in {**su.observed_value, **su.estimated_value}:
        summary += (
            _format_value(
                key,
                su.observed_value.get(key),
                su.estimated_value.get(key),
                su.estimated_value_lower.get(key),
                su.estimated_value_upper.get(key),
            )
            + "\n"
        )

    return summary


@hooks.hook(HookEvent.SYNTHESIS_STAGE, stage_context=True)
async def synthesize_result(
    query_plan: QueryPlan,
    result: SerializedResultType,
    step_uncertainties: dict[str, StepUncertainty],
) -> SynthesizedResult:
    """Combine recorded uncertainty into a final user-facing answer.

    Args:
        query_plan: the executed query plan
        result: the base result to summarize
        step_uncertainties: the per-step uncertainty metadata, keyed by step id

    Returns:
        (SynthesizedResult): the synthesized answer with combined completeness
    """
    with tracing.own_span(
        Stage.SYNTHESIS.label,
        input={
            "query_plan": query_plan,
            "step_uncertainties": step_uncertainties,
        },
    ):
        combined_completeness = reduce(mul, (u.completeness for u in step_uncertainties.values()), 1.0)
        summary = f"""Query:
{query_plan.query}

Result:
```
{result_type.serialize(result, context="full_preview")}
```

Estimated completeness: {combined_completeness:.1%}
"""

        for step in query_plan.steps:
            summary = _format_step(summary, step, step_uncertainties.get(step.id))

        synthesized = SynthesizedResult(
            result=result,
            summary=summary,
            combined_completeness=combined_completeness,
            step_uncertainties=step_uncertainties,
        )

        tracing.append_to_span_output(
            {
                "combined_completeness": combined_completeness,
                "summary": summary,
            }
        )
        return synthesized

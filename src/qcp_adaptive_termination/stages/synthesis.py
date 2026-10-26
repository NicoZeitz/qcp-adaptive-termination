"""Synthesis stage: turn recorded uncertainty into a user-facing answer."""

import qcp_adaptive_termination.hooks as hooks
from qcp_adaptive_termination.hooks import HookEvent
from qcp_adaptive_termination.models import StepUncertainty
from qcp_adaptive_termination.models.query_plan import QueryPlan
from qcp_adaptive_termination.models.result_type import SerializedResultType
from qcp_adaptive_termination.models.stages import Stage, SynthesizedResult
from qcp_adaptive_termination.observability import tracing


def _combine_completeness(uncertainties: list[StepUncertainty]) -> float | None:
    """Product of per-step completeness along the dependency chain (Lusser)."""
    product = 1.0
    any_value = False
    for u in uncertainties:
        if u.completeness is not None:
            product *= u.completeness
            any_value = True
    return product if any_value else None


@hooks.hook(HookEvent.SYNTHESIS_STAGE, stage_context=True)
async def synthesize_result(
    query: str,
    query_plan: QueryPlan | None,
    result: SerializedResultType,
    step_uncertainties: list[StepUncertainty],
) -> SynthesizedResult:
    """Combine recorded uncertainty into a final user-facing answer.

    Args:
        query: the user query
        query_plan: the executed query plan
        result: the base result to summarize
        step_uncertainties: the per-step uncertainty metadata

    Returns:
        (SynthesizedResult): the synthesized answer with combined completeness
    """
    with tracing.own_span(
        Stage.SYNTHESIS.label,
        input={
            "query": query,
            "query_plan": query_plan,
            "step_uncertainties": step_uncertainties,
        },
    ):
        combined = _combine_completeness(step_uncertainties)

        value_lines = []
        for u in step_uncertainties:
            for agg, point in u.value_point.items():
                lo = u.value_lower.get(agg)
                hi = u.value_upper.get(agg)
                if lo is not None and hi is not None:
                    value_lines.append(f"  step {u.step_id} {agg}: {point:.2f} (CI {lo:.2f}–{hi:.2f})")
                else:
                    value_lines.append(f"  step {u.step_id} {agg}: {point:.2f}")

        lines = [f"Query: {query}"]
        if combined is not None:
            lines.append(f"Estimated completeness: {combined:.1%}")
        if value_lines:
            lines.append("Estimated values:")
            lines.extend(value_lines)
        summary = "\n".join(lines)

        synthesized = SynthesizedResult(
            result=result,
            summary=summary,
            combined_completeness=combined,
            step_uncertainties=step_uncertainties,
        )

        tracing.append_to_span_output(
            {
                "combined_completeness": combined,
                "summary": summary,
            }
        )
        return synthesized

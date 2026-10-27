"""Completeness evaluation worker: run one query and capture per-step results."""

import asyncio
import json
import logging
import math
import re
from argparse import Namespace
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
from agents.items import TResponseInputItem
from agents.usage import Usage
from pydantic import BaseModel

import qcp_adaptive_termination.hooks as hooks
from qcp_adaptive_termination.estimator import Estimator
from qcp_adaptive_termination.hooks import HookEvent
from qcp_adaptive_termination.llm import transcript as transcript_mod
from qcp_adaptive_termination.models import (
    AgentRole,
    AggregationType,
    AlignedResultList,
    AlignedSchemaMetadata,
    Config,
    DataAgent,
    StepBudget,
    StopReason,
    result_type,
)
from qcp_adaptive_termination.models.execution import (
    AdditionalInformation,
    FailedExecution,
    PartialQueryAnswer,
    StopDecision,
)
from qcp_adaptive_termination.models.query_plan import (
    MultiAgentQueryPlanStep,
    QueryPlan,
)
from qcp_adaptive_termination.models.stages import (
    CollaborativeQueryAnswering,
    CollaborativeQueryAnsweringResult,
    DraftedQueryPlan,
    Stage,
)
from qcp_adaptive_termination.observability import context
from scripts.completeness_evaluation.ground_truth import compute_ground_truth
from scripts.completeness_evaluation.models import BatchRecord, GroundTruth, StepResult
from scripts.completeness_evaluation.results import save_step_results
from scripts.utils.canonical import (
    load_meta_mapping,
    make_align_schema_override,
    make_data_agent_override,
    parse_canonical_table,
    query_agent_canonical,
)
from scripts.utils.setup import ScriptContext, run_script
from scripts.utils.sql import parse_select_columns

# Per-step dispatch maps: step execution_id -> step-specific handler.
_per_step_align: dict[str, Callable[..., AlignedResultList]] = {}
_per_step_data: dict[str, Callable[..., PartialQueryAnswer]] = {}


def _json_safe(value: Any) -> Any:
    """Convert an LLM run output into a JSON-serializable value."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    return value


def _replan_error(
    execution_id: str,
    data_agents: list[DataAgent],
    relevant_data_agents: list[DataAgent],
    query_plan: QueryPlan,
    reason: FailedExecution,
) -> QueryPlan:
    """Raise since replanning is unexpected in the evaluator benchmark."""
    del execution_id, data_agents, relevant_data_agents, query_plan
    raise RuntimeError(f"Unexpected replan triggered in evaluator benchmark. Step: {reason.step.id!r}")


def _make_align_dispatch(
    handlers: dict[str, Callable[..., AlignedResultList]],
) -> Callable[..., AlignedResultList]:
    """Dispatch ALIGN_SCHEMA by step query (each step has a distinct query)."""

    def _dispatch(
        execution_id: str,
        data_agents: list[DataAgent],
        agent_names: list[str],
        query: str,
        merge_task: str,
    ) -> AlignedResultList:
        """Route an align_schema call to the handler for its step query."""
        handler = handlers.get(query)
        if handler is None:
            raise ValueError(f"No align_schema handler for query={query!r}")
        return handler(
            execution_id,
            data_agents=data_agents,
            agent_names=agent_names,
            query=query,
            merge_task=merge_task,
        )

    return _dispatch


def _make_data_dispatch(
    handlers: dict[str, Callable[..., PartialQueryAnswer]],
) -> Callable[..., Awaitable[PartialQueryAnswer]]:
    """Dispatch DATA_AGENT by step query (each step has a distinct query).

    Runs the canonical stub and the real data-agent LLM call in parallel so the
    actual query cost is recorded (via the AGENT_COMPLETED usage hook) while the
    stub's deterministic result is kept. Errors from the real call are ignored.
    """

    async def _dispatch(
        execution_id: str,
        data_agent: DataAgent,
        query: str,
        additional_information: list[AdditionalInformation],
        schema_metadata: AlignedSchemaMetadata | None,
        default_fn: Callable[..., Awaitable[PartialQueryAnswer]] | None = None,
    ) -> PartialQueryAnswer:
        """Route a data_agent call to the stub handler, running the real call in parallel."""
        handler = handlers.get(query)
        if handler is None:
            raise ValueError(f"No data_agent handler for query={query!r}")

        async def _run_stub() -> PartialQueryAnswer:
            """Return the canonical stub result for the queried agent."""
            return handler(
                execution_id,
                data_agent=data_agent,
                query=query,
                additional_information=additional_information,
                schema_metadata=schema_metadata,
            )

        async def _run_real() -> None:
            """Run the real data-agent call so its usage is recorded; result discarded."""
            if default_fn is not None:
                await default_fn(
                    data_agent=data_agent,
                    query=query,
                    additional_information=additional_information,
                    schema_metadata=schema_metadata,
                )

        stub_result, _real_result = await asyncio.gather(
            _run_stub(),
            _run_real(),
            return_exceptions=True,
        )
        if isinstance(stub_result, BaseException):
            raise stub_result
        return stub_result

    return _dispatch


def _value_metrics(
    estimator: Estimator,
    aggregation: AggregationType | None,
) -> tuple[float | None, float | None, float | None, float | None]:
    """(observed, estimated, est_lower, est_upper) for the primary aggregation."""
    if aggregation == AggregationType.SUM:
        return (
            estimator.sample_sum(),
            estimator.estimated_sum(),
            estimator.estimated_sum_lower_bound(),
            estimator.estimated_sum_upper_bound(),
        )
    if aggregation == AggregationType.AVG:
        return (
            estimator.sample_avg(),
            estimator.estimated_avg(),
            estimator.estimated_avg_lower_bound(),
            estimator.estimated_avg_upper_bound(),
        )
    if aggregation == AggregationType.MIN:
        return (estimator.sample_min(), None, None, None)
    if aggregation == AggregationType.MAX:
        return (estimator.sample_max(), None, None, None)
    if aggregation == AggregationType.COUNT:
        return (
            float(estimator.sample_count()),
            estimator.estimated_count(),
            estimator.estimated_count_lower_bound(),
            estimator.estimated_count_upper_bound(),
        )
    if aggregation == AggregationType.Q25:
        return (estimator.sample_quantile(0.25), None, None, None)
    if aggregation == AggregationType.MEDIAN:
        return (estimator.sample_quantile(0.5), None, None, None)
    if aggregation == AggregationType.Q75:
        return (estimator.sample_quantile(0.75), None, None, None)
    return (None, None, None, None)


def _estimator_snapshot(
    estimator: Estimator | None,
    aggregation: AggregationType | None,
) -> dict[str, Any]:
    """Extract the estimator's observable state for one completeness check."""
    if estimator is None:
        return {
            "sample_coverage": None,
            "observed_count": None,
            "estimated_count": None,
            "estimated_count_lower": None,
            "estimated_count_upper": None,
            "c_worst": None,
            "c_middle": None,
            "c_best": None,
            "observed_value": None,
            "estimated_value": None,
            "estimated_value_lower": None,
            "estimated_value_upper": None,
        }
    observed_count = estimator.sample_count()
    estimated_count = estimator.estimated_count()
    lower, upper = estimator.estimated_count_bounds()

    def ratio(numerator, denominator) -> float | None:
        """Relative ratio of two counts, None when the denominator is unusable."""
        if denominator is None or not math.isfinite(denominator) or denominator <= 0:
            return None
        return numerator / denominator

    observed_value, estimated_value, est_lower, est_upper = _value_metrics(estimator, aggregation)
    return {
        "sample_coverage": estimator.sample_coverage(),
        "observed_count": observed_count,
        "estimated_count": estimated_count,
        "estimated_count_lower": lower,
        "estimated_count_upper": upper,
        "c_worst": ratio(observed_count, upper),
        "c_middle": ratio(observed_count, estimated_count),
        "c_best": ratio(observed_count, lower),
        "observed_value": observed_value,
        "estimated_value": estimated_value,
        "estimated_value_lower": est_lower,
        "estimated_value_upper": est_upper,
    }


def _agents_from_label(label: str | None) -> int | None:
    """Parse agents_queried out of a completeness label (``... agents_queried=N``)."""
    if label is None:
        return None
    match = re.search(r"agents_queried=(\d+)", label)
    return int(match.group(1)) if match else None


def _true_value_for(aggregation: AggregationType | None, gt: GroundTruth) -> float | None:
    """Ground-truth value matching the step's primary aggregation."""
    if aggregation in (AggregationType.SUM,):
        return gt.value_sum
    if aggregation in (AggregationType.AVG,):
        return gt.value_avg
    if aggregation in (AggregationType.MIN,):
        return gt.value_min
    if aggregation in (AggregationType.MAX,):
        return gt.value_max
    if aggregation in (AggregationType.COUNT,):
        return float(gt.count)
    return None


def _uses_value(aggregation: AggregationType | None) -> bool:
    """True when the step's primary aggregation is a value aggregate (not count/quantile)."""
    return aggregation in (AggregationType.SUM, AggregationType.AVG, AggregationType.MIN, AggregationType.MAX)


def _rel_err(truth: float | None, observed: float | None) -> float | None:
    """Relative error |truth - observed| / |truth| (None when not computable)."""
    if truth is None or truth == 0 or observed is None:
        return None
    return abs(truth - observed) / abs(truth)


def _produced_scalar_and_errors(
    produced: Any,
    gt: GroundTruth,
) -> tuple[float | None, float | None, float | None]:
    """Extract a scalar from the produced answer and relative errors vs truth.

    A single numeric cell is treated as the produced scalar; errors are computed
    against the ground-truth average and sum.
    """
    if isinstance(produced, dict) and produced.get("data"):
        cells = [c for row in produced["data"] for c in row]
        if len(cells) == 1 and isinstance(cells[0], (int, float)) and not isinstance(cells[0], bool):
            scalar = float(cells[0])
        else:
            return None, None, None
    elif isinstance(produced, (int, float)) and not isinstance(produced, bool):
        scalar = float(produced)
    else:
        return None, None, None

    def rel_err(truth: float | None) -> float | None:
        """Relative error of the scalar vs a truth value, None when uncomputable."""
        if truth is None or truth == 0:
            return None
        return abs(scalar - truth) / abs(truth)

    return scalar, rel_err(gt.value_avg), rel_err(gt.value_sum)


class _StepTracker:
    """Collects one completeness-check record per queried agent, per step."""

    def __init__(self, output_dir: Path):
        """Initialize per-step accumulation state."""
        self._output_dir = output_dir
        # step query -> step id
        self._query_to_step: dict[str, str] = {}
        # step_id -> ordered agent names
        self._ordered: dict[str, list[str]] = {}
        # step_id -> list of BatchRecord (one per agent query / completeness check)
        self._batches: dict[str, list[BatchRecord]] = {}
        self._seen_keys: dict[str, set[tuple]] = {}
        self._total_unique: dict[str, int] = {}
        self._key_cols: dict[str, list[str]] = {}
        self._agents_total: dict[str, int] = {}
        self._aggregation: dict[str, AggregationType | None] = {}
        self._step_budget_t: dict[str, float] = {}
        self._produced: dict[str, Any] = {}

    def register_step(
        self,
        step_id: str,
        step_query: str,
        key_cols: list[str],
        total_unique: int,
        agents_total: int,
        aggregation: AggregationType | None,
    ):
        """Declare a step's query, key columns, ground truth and aggregation."""
        self._query_to_step[step_query] = step_id
        self._key_cols[step_id] = key_cols
        self._total_unique[step_id] = total_unique
        self._agents_total[step_id] = agents_total
        self._aggregation[step_id] = aggregation

    def _coverage(self, step_id: str) -> float:
        """Fraction of unique keys seen for a step."""
        seen = self._seen_keys.get(step_id, set())
        total = self._total_unique.get(step_id, 0)
        return min(len(seen) / total, 1.0) if total > 0 else 0.0

    def on_agent_query(
        self,
        execution_id: str,
        data_agent: DataAgent,
        query: str,
        result: PartialQueryAnswer,
    ):
        """Record one agent query for the step it belongs to (via step query)."""
        step_id = self._query_to_step.get(query)
        if step_id is None:
            return
        ordered = self._ordered.setdefault(step_id, [])
        ordered.append(data_agent.name)

        df = result.answer
        if df is not None and not df.empty:
            cols = [c for c in self._key_cols.get(step_id, []) if c in df.columns]
            if cols:
                self._seen_keys.setdefault(step_id, set()).update(
                    map(tuple, df[cols].drop_duplicates().values.tolist())
                )

        observed_count = len(self._seen_keys.get(step_id, set()))
        agents_total = self._agents_total.get(step_id, 0)
        agg = self._aggregation.get(step_id)
        self._batches.setdefault(step_id, []).append(
            BatchRecord(
                batch_num=len(self._batches[step_id]) + 1,
                cumulative_agents=len(ordered),
                decision="query_more_agents",
                stop=False,
                agents_queried_fraction=len(ordered) / agents_total if agents_total else 0.0,
                results_count=None,
                unique_results_count=observed_count,
                sample_coverage=None,
                observed_count=observed_count,
                estimated_count=None,
                estimated_count_lower=None,
                estimated_count_upper=None,
                predicted_remaining=None,
                realized_unseen=None,
                c_worst=None,
                c_middle=None,
                c_best=None,
                estimator_active=False,
                estimator_decision=None,
                llm_decision=None,
                aggregation=agg.value if agg is not None else None,
                observed_value=None,
                estimated_value=None,
                estimated_value_lower=None,
                estimated_value_upper=None,
                coverage_achieved=self._coverage(step_id),
                true_count=self._total_unique.get(step_id, 0),
                true_value=None,
                observed_error=None,
                estimated_error=None,
            )
        )

    def on_completeness(
        self,
        execution_id: str,
        aligned_result_list: AlignedResultList,
        estimator: Estimator | None,
        step_budget: StepBudget | None,
        agents_queried: int,
        force_result: bool,
        result: StopDecision | None,
    ):
        """Record the estimator state + stop decision at each completeness check."""
        step_id = step_budget.step_id if step_budget is not None else None
        if step_id is None:
            return
        agg = self._aggregation.get(step_id)
        snapshot = _estimator_snapshot(estimator, agg)

        if result is not None:
            decision = result.stop_reason.value
        elif force_result:
            decision = StopReason.ALL_AGENTS.value
        else:
            decision = "query_more_agents"
        is_stop = decision in (StopReason.ESTIMATOR.value, StopReason.LLM.value, StopReason.ALL_AGENTS.value)
        if step_budget is not None:
            self._step_budget_t[step_id] = step_budget.t_step

        batches = self._batches.setdefault(step_id, [])
        if not batches:
            return
        last = batches[-1]
        last.decision = decision
        last.stop = is_stop
        last.unique_results_count = snapshot["observed_count"]
        last.results_count = len(aligned_result_list.data) if aligned_result_list is not None else None
        last.sample_coverage = snapshot["sample_coverage"]
        last.observed_count = snapshot["observed_count"]
        last.estimated_count = snapshot["estimated_count"]
        last.estimated_count_lower = snapshot["estimated_count_lower"]
        last.estimated_count_upper = snapshot["estimated_count_upper"]
        last.c_worst = snapshot["c_worst"]
        last.c_middle = snapshot["c_middle"]
        last.c_best = snapshot["c_best"]
        last.observed_value = snapshot["observed_value"]
        last.estimated_value = snapshot["estimated_value"]
        last.estimated_value_lower = snapshot["estimated_value_lower"]
        last.estimated_value_upper = snapshot["estimated_value_upper"]
        last.estimator_active = agents_queried >= context.get_config().completeness.estimator.start_agents

    def on_estimator(
        self,
        execution_id: str,
        estimator: Estimator | None,
        step_budget: StepBudget | None,
        agents_queried: int,
        result: bool,
    ):
        """Record the deterministic-estimator sub-decision + metrics for a check."""
        del execution_id
        step_id = step_budget.step_id if step_budget is not None else None
        if step_id is None or agents_queried < 1:
            return
        batches = self._batches.get(step_id)
        if not batches or agents_queried - 1 >= len(batches):
            return
        last = batches[agents_queried - 1]
        start_agents = context.get_config().completeness.estimator.start_agents
        last.estimator_active = agents_queried >= start_agents
        if agents_queried < start_agents:
            last.estimator_decision = "warmup"
        else:
            last.estimator_decision = "stop" if result else "continue"
        # Fill c_* from the estimator even when it did not decide.
        agg = self._aggregation.get(step_id)
        snapshot = _estimator_snapshot(estimator, agg)
        last.c_worst = snapshot["c_worst"]
        last.c_middle = snapshot["c_middle"]
        last.c_best = snapshot["c_best"]
        last.estimated_count_lower = snapshot["estimated_count_lower"]
        last.estimated_count_upper = snapshot["estimated_count_upper"]

    def on_llm(
        self,
        execution_id: str,
        aligned_result_list: AlignedResultList,
        query: str,
        label: str | None,
        result: bool,
    ):
        """Record the LLM world-knowledge stop decision for a check."""
        del execution_id, aligned_result_list
        step_id = self._query_to_step.get(query)
        if step_id is None or label is None:
            return
        agents_queried = _agents_from_label(label)
        batches = self._batches.get(step_id)
        if not batches or agents_queried is None or agents_queried < 1 or agents_queried - 1 >= len(batches):
            return
        last = batches[agents_queried - 1]
        last.llm_decision = "stop" if result else "continue"

    def on_llm_completed(
        self,
        execution_id: str,
        role: AgentRole,
        agent_name: str | None,
        label: str | None,
        continuation_input: list[TResponseInputItem],
        output: Any,
        usage: Usage,
    ):
        """Write the run's Markdown transcript and raw input JSON."""
        del execution_id, usage
        meta = transcript_mod.RunMeta(role=role, agent_name=agent_name, label=label)
        filename = transcript_mod.transcript_filename(meta)
        markdown = transcript_mod.render_run_markdown(meta, continuation_input, output)
        (self._output_dir / filename).write_text(markdown, encoding="utf-8")
        (self._output_dir / (filename[:-3] + ".json")).write_text(
            json.dumps(
                {"continuation_input": continuation_input, "output": _json_safe(output)},
                indent=2,
            ),
            encoding="utf-8",
        )

    def on_produce_result(
        self,
        execution_id: str,
        aligned_result_list: AlignedResultList,
        estimator: Estimator | None,
        query: str,
        merge_task: str,
        stop_reason: StopReason,
        result: Any,
    ):
        """Record the result-producer's final answer for its step (via step query)."""
        del aligned_result_list, estimator, merge_task, stop_reason, execution_id
        step_id = self._query_to_step.get(query)
        if step_id is None:
            return
        try:
            self._produced[step_id] = result_type.serialize(result, context="json")
        except Exception:
            self._produced[step_id] = result

    def produced_for(self, step_id: str) -> Any | None:
        """Return the recorded produced answer (serialized) for a step."""
        return self._produced.get(step_id)

    def step_budget_t(self, step_id: str) -> float | None:
        """Return the recorded step budget time for a step."""
        return self._step_budget_t.get(step_id)

    def aggregation_for(self, step_id: str) -> AggregationType | None:
        """Return the step's primary aggregation type, if any."""
        return self._aggregation.get(step_id)

    def fill_batch_derived(self):
        """Fill per-check predicted-remaining and realized-unseen from observed counts."""
        for _step_id, batches in self._batches.items():
            prev_observed = 0
            for batch in batches:
                obs = batch.observed_count or 0
                upper = batch.estimated_count_upper
                batch.predicted_remaining = (upper - obs) if upper is not None else None
                batch.realized_unseen = obs - prev_observed
                prev_observed = obs

    def batches_for(self, step_id: str) -> list[BatchRecord]:
        """Return the trajectory (one record per agent) for a step."""
        return self._batches.get(step_id, [])


def _build_step_result(
    step: MultiAgentQueryPlanStep,
    step_meta: dict[str, Any],
    relevant_data_agents: list[DataAgent],
    meta_mapping: dict[str, Any],
    data_dir: Path,
    query_slug: str,
    output_dir: Path,
    tracker: _StepTracker,
) -> StepResult | None:
    """Build one step's result from the captured trajectory + ground truth."""
    canonical_sql: str = step_meta["sql_query"]
    schema_meta = step_meta["aligned_schema_metadata"]
    key_cols: list[str] = schema_meta["keys"]
    value_column: str | None = schema_meta.get("value_column")

    canonical_table = parse_canonical_table(canonical_sql)
    if canonical_table is None:
        logging.warning("%s/%s: could not parse table — skipping", query_slug, step.id)
        return None
    canonical_cols = parse_select_columns(canonical_sql)
    if not canonical_cols:
        logging.warning("%s/%s: could not parse columns — skipping", query_slug, step.id)
        return None

    step_agent_names = set(step.to_agent)
    step_agents = [a for a in relevant_data_agents if a.name in step_agent_names]

    data_cache: dict[str, pd.DataFrame | None] = {}
    with ThreadPoolExecutor() as pool:
        futures = {
            agent.name: pool.submit(
                query_agent_canonical, agent, canonical_sql, canonical_table, meta_mapping, data_dir
            )
            for agent in step_agents
        }
        for name, future in futures.items():
            data_cache[name] = future.result()

    ground_truth = compute_ground_truth(data_cache, key_cols, value_column)
    logging.info(
        "%s/%s: true_count=%d true_sum=%s agents=%d",
        query_slug,
        step.id,
        ground_truth.count,
        f"{ground_truth.value_sum:.2f}" if ground_truth.value_sum is not None else "n/a",
        len(step_agents),
    )

    batches = tracker.batches_for(step.id)
    if not batches:
        logging.warning("%s/%s: no batch records", query_slug, step.id)
        return None

    agents_total = len(step.to_agent)
    aggregation = tracker.aggregation_for(step.id)
    step_budget_t = tracker.step_budget_t(step.id)
    true_value = _true_value_for(aggregation, ground_truth)
    uses_value = _uses_value(aggregation)

    # Fill per-check step-level calibration fields (true value + errors).
    for batch in batches:
        batch.true_value = true_value
        if uses_value:
            observed_metric, estimated_metric = batch.observed_value, batch.estimated_value
        else:
            observed_metric, estimated_metric = batch.observed_count, batch.estimated_count
        batch.observed_error = _rel_err(true_value if uses_value else float(ground_truth.count), observed_metric)
        batch.estimated_error = _rel_err(true_value if uses_value else float(ground_truth.count), estimated_metric)

    stop_batch = batches[-1]
    agents_queried_at_stop = stop_batch.cumulative_agents
    stop_fraction = agents_queried_at_stop / agents_total if agents_total > 0 else 0.0
    early_stop = agents_queried_at_stop < agents_total

    produced_raw = tracker.produced_for(step.id)
    produced_text = json.dumps(produced_raw, default=str) if produced_raw is not None else None
    produced_scalar, rel_err_avg, rel_err_sum = _produced_scalar_and_errors(produced_raw, ground_truth)

    logging.info(
        "%s/%s: stop=%d/%d (%.0f%%) coverage=%.3f count_est=%s true=%d early=%s produced=%s",
        query_slug,
        step.id,
        agents_queried_at_stop,
        agents_total,
        stop_fraction * 100,
        stop_batch.coverage_achieved,
        f"{stop_batch.estimated_count:.1f}" if stop_batch.estimated_count is not None else "n/a",
        ground_truth.count,
        early_stop,
        f"{produced_scalar:.4g}" if produced_scalar is not None else "n/a",
    )

    result = StepResult(
        step_id=step.id,
        agents_total=agents_total,
        agents_queried_at_stop=agents_queried_at_stop,
        stop_fraction=stop_fraction,
        early_stop=early_stop,
        stop_reason=stop_batch.decision,
        ground_truth=ground_truth,
        checks=batches,
        aggregation=aggregation,
        step_budget_t=step_budget_t,
        produced_result=produced_text,
        produced_scalar=produced_scalar,
        result_rel_err_avg=rel_err_avg,
        result_rel_err_sum=rel_err_sum,
    )
    save_step_results(result, query_slug, output_dir)
    return result


class _CompletenessRunner:
    """Hook-only completeness evaluation for one query."""

    def __init__(self):
        """Initialize the hook-only evaluation runner."""
        self.tracker: _StepTracker | None = None
        self.step_infos: list[tuple[MultiAgentQueryPlanStep, dict[str, Any]]] = []
        self.relevant_data_agents: list[DataAgent] = []
        self.all_agents: list[DataAgent] = []
        self.meta_mapping: dict[str, Any] = {}
        self.data_dir: Path | None = None
        self.query: str = ""
        self.query_slug: str = ""
        self.output_dir: Path | None = None
        self._align_dispatch: Callable[..., AlignedResultList] | None = None
        self._data_dispatch: Callable[..., Awaitable[PartialQueryAnswer]] | None = None

    async def before(self, ctx: ScriptContext):
        """Pre-build data caches/ground truth and register all hooks."""
        extra = ctx.extra or {}
        assert isinstance(ctx.data, CollaborativeQueryAnswering)
        assert ctx.output_dir is not None
        assert ctx.dataset_dir is not None
        assert ctx.data_dir is not None

        steps_metadata: dict[str, Any] = extra.get("steps_metadata", {})

        raw_query_plan = extra.get("query_plan")
        if raw_query_plan is None:
            raise RuntimeError(f"{self.query_slug}: no query_plan in extra — cannot evaluate")
        query_plan = QueryPlan.model_validate(raw_query_plan)

        multi_steps = [s for s in query_plan.steps if isinstance(s, MultiAgentQueryPlanStep)]
        if not multi_steps:
            raise RuntimeError(f"{self.query_slug}: no multi-agent steps — nothing to evaluate")

        self.output_dir = ctx.output_dir
        self.data_dir = ctx.data_dir
        self.all_agents = ctx.data.data_agents
        relevant_names: set[str] = set(extra.get("relevant_data_agents", []))
        self.relevant_data_agents = [a for a in self.all_agents if a.name in relevant_names]
        self.query = query_plan.query
        self.query_slug = ctx.input_file.stem
        self.meta_mapping = load_meta_mapping(ctx.dataset_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tracker = _StepTracker(self.output_dir)

        # Pre-build per-step data caches + ground truth.
        for step in multi_steps:
            step_meta = steps_metadata.get(step.id)
            if step_meta is None:
                logging.warning("%s/%s: no steps_metadata — skipping", self.query_slug, step.id)
                continue
            canonical_sql: str = step_meta["sql_query"]
            schema_meta = step_meta["aligned_schema_metadata"]
            canonical_table = parse_canonical_table(canonical_sql)
            if canonical_table is None:
                continue
            step_agent_names = set(step.to_agent)
            step_agents = [a for a in self.relevant_data_agents if a.name in step_agent_names]
            data_cache: dict[str, pd.DataFrame | None] = {}
            with ThreadPoolExecutor() as pool:
                futures = {
                    agent.name: pool.submit(
                        query_agent_canonical, agent, canonical_sql, canonical_table, self.meta_mapping, self.data_dir
                    )
                    for agent in step_agents
                }
                for name, future in futures.items():
                    data_cache[name] = future.result()
            align_handler = make_align_schema_override(step_meta, self.query)
            data_handler = make_data_agent_override(
                canonical_sql, canonical_table, self.meta_mapping, data_cache, self.data_dir
            )
            _per_step_align[step.query] = align_handler
            _per_step_data[step.query] = data_handler

            key_cols: list[str] = schema_meta["keys"]
            value_column: str | None = schema_meta.get("value_column")
            aggregation = (
                AggregationType(schema_meta["aggregation_types"][0]) if schema_meta["aggregation_types"] else None
            )
            gt = compute_ground_truth(data_cache, key_cols, value_column)
            self.tracker.register_step(
                step_id=step.id,
                step_query=step.query,
                key_cols=key_cols,
                total_unique=gt.count,
                agents_total=len(step.to_agent),
                aggregation=aggregation,
            )
            self.step_infos.append((step, step_meta))

        # Inject the precomputed plan so run_stages at EXECUTION sees a DraftedQueryPlan.
        drafted = DraftedQueryPlan(
            query_plan=query_plan,
            relevant_data_agents=self.relevant_data_agents,
            data_agents=self.all_agents,
        )

        @hooks.overwrite(HookEvent.SETUP_LOAD_INPUT)
        def _load_input_override(execution_id, path) -> tuple[Any, dict[str, Any] | None]:
            """Return the precomputed drafted plan and extra context as input."""
            del execution_id, path
            return drafted, extra

        self._align_dispatch = _make_align_dispatch(_per_step_align)
        self._data_dispatch = _make_data_dispatch(_per_step_data)
        hooks.register_overwrite(
            HookEvent.EXECUTION_EXECUTE_QUERY_PLAN_EXECUTE_STEP_ANSWER_STEP_MULTIPLE_AGENTS_ALIGN_SCHEMA,
            self._align_dispatch,
        )
        hooks.register_overwrite(HookEvent.EXECUTION_QUERY_DATA_AGENT, self._data_dispatch)
        hooks.register_overwrite(HookEvent.EXECUTION_REPLAN_DRAFT_QUERY_PLAN, _replan_error)
        hooks.register_post(HookEvent.EXECUTION_QUERY_DATA_AGENT, self.tracker.on_agent_query)
        hooks.register_post(
            HookEvent.EXECUTION_EXECUTE_QUERY_PLAN_EXECUTE_STEP_ANSWER_STEP_MULTIPLE_AGENTS_COMPLETENESS,
            self.tracker.on_completeness,
        )
        hooks.register_post(
            HookEvent.EXECUTION_EXECUTE_QUERY_PLAN_EXECUTE_STEP_ANSWER_STEP_MULTIPLE_AGENTS_COMPLETENESS_ESTIMATOR,
            self.tracker.on_estimator,
        )
        hooks.register_post(
            HookEvent.EXECUTION_EXECUTE_QUERY_PLAN_EXECUTE_STEP_ANSWER_STEP_MULTIPLE_AGENTS_COMPLETENESS_LLM,
            self.tracker.on_llm,
        )
        hooks.register_post(
            HookEvent.EXECUTION_EXECUTE_QUERY_PLAN_EXECUTE_STEP_ANSWER_STEP_MULTIPLE_AGENTS_RESULT_PRODUCER,
            self.tracker.on_produce_result,
        )
        hooks.register_post(HookEvent.AGENT_COMPLETED, self.tracker.on_llm_completed)

    async def after(self, ctx: ScriptContext, result: CollaborativeQueryAnsweringResult):
        """Unregister hooks and build per-step results from the trajectory."""

        assert self.tracker is not None
        if self._align_dispatch is not None:
            hooks.unregister_overwrite(
                HookEvent.EXECUTION_EXECUTE_QUERY_PLAN_EXECUTE_STEP_ANSWER_STEP_MULTIPLE_AGENTS_ALIGN_SCHEMA,
                self._align_dispatch,
            )
        if self._data_dispatch is not None:
            hooks.unregister_overwrite(HookEvent.EXECUTION_QUERY_DATA_AGENT, self._data_dispatch)
        hooks.unregister_overwrite(HookEvent.EXECUTION_REPLAN_DRAFT_QUERY_PLAN, _replan_error)
        hooks.unregister_post(HookEvent.EXECUTION_QUERY_DATA_AGENT, self.tracker.on_agent_query)
        hooks.unregister_post(
            HookEvent.EXECUTION_EXECUTE_QUERY_PLAN_EXECUTE_STEP_ANSWER_STEP_MULTIPLE_AGENTS_COMPLETENESS,
            self.tracker.on_completeness,
        )
        hooks.unregister_post(
            HookEvent.EXECUTION_EXECUTE_QUERY_PLAN_EXECUTE_STEP_ANSWER_STEP_MULTIPLE_AGENTS_COMPLETENESS_ESTIMATOR,
            self.tracker.on_estimator,
        )
        hooks.unregister_post(
            HookEvent.EXECUTION_EXECUTE_QUERY_PLAN_EXECUTE_STEP_ANSWER_STEP_MULTIPLE_AGENTS_COMPLETENESS_LLM,
            self.tracker.on_llm,
        )
        hooks.unregister_post(
            HookEvent.EXECUTION_EXECUTE_QUERY_PLAN_EXECUTE_STEP_ANSWER_STEP_MULTIPLE_AGENTS_RESULT_PRODUCER,
            self.tracker.on_produce_result,
        )
        hooks.unregister_post(HookEvent.AGENT_COMPLETED, self.tracker.on_llm_completed)
        for step, _ in self.step_infos:
            _per_step_align.pop(step.query, None)
            _per_step_data.pop(step.query, None)

        self.tracker.fill_batch_derived()

        assert self.output_dir is not None and self.data_dir is not None

        # Build per-step results from the captured trajectory.
        for step, step_meta in self.step_infos:
            _build_step_result(
                step=step,
                step_meta=step_meta,
                relevant_data_agents=self.relevant_data_agents,
                meta_mapping=self.meta_mapping,
                data_dir=self.data_dir,
                query_slug=self.query_slug,
                output_dir=self.output_dir,
                tracker=self.tracker,
            )


async def run_query(
    *,
    input_file: Path,
    config: Config,
    dataset_dir: Path,
    output_dir: Path,
    cache_dir: Path | None,
    data: CollaborativeQueryAnswering,
    extra: dict[str, Any] | None,
):
    """Run completeness evaluation for one query.

    Args:
        input_file: input query JSON path
        config: config to use for the run
        dataset_dir: dataset folder with _meta/input and meta mapping
        output_dir: directory to write per-query outputs
        cache_dir: directory for cache files
        data: parsed collaborative query answering input
        extra: extra context (query plan and step metadata)
    """
    query_slug = input_file.stem
    run_output_dir = output_dir / query_slug
    args = Namespace(
        name="",
        input=str(input_file),
        config=None,
        output_dir=str(run_output_dir),
        cache_dir=str(cache_dir) if cache_dir else None,
        dataset_dir=str(dataset_dir),
        start_stage=Stage.EXECUTION,
        end_stage=Stage.SYNTHESIS,
    )
    runner = _CompletenessRunner()
    await run_script(
        args=args,
        before_fn=runner.before,
        after_fn=runner.after,
        config=config,
    )

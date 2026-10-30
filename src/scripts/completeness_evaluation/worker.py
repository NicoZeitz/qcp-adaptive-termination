"""Completeness evaluation worker: run one query and capture per-step results."""

import asyncio
import json
import logging
import math
import re
import traceback
from argparse import Namespace
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
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
    DraftedQueryPlan,
    Stage,
    SynthesizedResult,
)
from qcp_adaptive_termination.observability import context
from scripts.completeness_evaluation.ground_truth import compute_ground_truth, truth_frame, truth_keys
from scripts.completeness_evaluation.models import (
    COVERAGE_TARGETS,
    BatchRecord,
    GroundTruth,
    StepResult,
)
from scripts.completeness_evaluation.results import save_step_results
from scripts.utils.canonical import (
    load_meta_mapping,
    make_align_schema_override,
    parse_canonical_table,
    query_agent_canonical,
)
from scripts.utils.setup import ScriptContext, run_script
from scripts.utils.sql import literal_key, literal_predicate_sets, replace_literal_predicates

# Per-step dispatch maps: step query -> step-specific handler.
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
        return (estimator.sample_quantile(0.5), estimator.estimated_median(), None, None)
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
    if aggregation in (AggregationType.MEDIAN,):
        return gt.value_median
    return None


def _uses_value(aggregation: AggregationType | None) -> bool:
    """True when the step's primary aggregation is a value aggregate (not count)."""
    return aggregation in (
        AggregationType.SUM,
        AggregationType.AVG,
        AggregationType.MIN,
        AggregationType.MAX,
        AggregationType.MEDIAN,
    )


def _rel_err(truth: float | None, observed: float | None) -> float | None:
    """Relative error |truth - observed| / |truth| (None when not computable)."""
    if truth is None or truth == 0 or observed is None:
        return None
    return abs(truth - observed) / abs(truth)


def _produced_scalar(produced: pd.DataFrame | None) -> float | None:
    """Return the produced answer's value when it is a single numeric cell."""
    if produced is None or produced.shape != (1, 1):
        return None
    cell = produced.iat[0, 0]
    if isinstance(cell, bool):
        return None
    try:
        value = float(cell)
    except TypeError, ValueError:
        return None
    return value if math.isfinite(value) else None


def _produced_value(
    produced: pd.DataFrame | None,
    source: _StepSource,
) -> tuple[float | None, str | None]:
    """The produced answer's value for the step's primary aggregation, with how it was obtained.

    A single numeric cell is taken as is; a list answer is aggregated like the ground truth.
    """
    scalar = _produced_scalar(produced)
    if scalar is not None:
        return scalar, "scalar"
    if produced is None or produced.empty:
        return None, None
    aggregation = source.aggregation
    if aggregation == AggregationType.COUNT:
        keys = [c for c in source.key_cols if c in produced.columns]
        return (float(len(produced.drop_duplicates(subset=keys))), "aggregated") if keys else (None, None)
    if source.value_column is None:
        return None, None
    # The producer may rename the value column (e.g. area_km2 -> area_km2_avg).
    value_column = next((c for c in produced.columns if source.value_column in str(c)), None)
    if value_column is None:
        return None, None
    values = np.asarray(pd.to_numeric(produced[value_column], errors="coerce"), dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return None, None
    reducers: dict[AggregationType, Callable[[np.ndarray], float]] = {
        AggregationType.SUM: lambda v: float(v.sum()),
        AggregationType.AVG: lambda v: float(v.mean()),
        AggregationType.MIN: lambda v: float(v.min()),
        AggregationType.MAX: lambda v: float(v.max()),
        AggregationType.MEDIAN: lambda v: float(np.median(v)),
    }
    reducer = reducers.get(aggregation) if aggregation is not None else None
    return (reducer(values), "aggregated") if reducer is not None else (None, None)


def _produced_key_metrics(
    produced: pd.DataFrame | None,
    key_cols: list[str],
    keys: set[tuple],
) -> tuple[float | None, float | None]:
    """Precision and recall of the produced answer's keys against the truth keys (list results)."""
    if produced is None or not keys or not all(c in produced.columns for c in key_cols):
        return None, None
    produced_keys = {tuple(literal_key(v) for v in row) for row in produced[key_cols].dropna().values.tolist()}
    truth = {tuple(literal_key(v) for v in row) for row in keys}
    if not produced_keys:
        return None, 0.0
    hit = len(produced_keys & truth)
    return hit / len(produced_keys), hit / len(truth)


def _target_eval(checks: list[BatchRecord], budget: float | None) -> dict[str, int | None]:
    """Map each coverage target (and the step budget as ``budget``) to the first agent index reaching it."""
    targets = {f"{int(t * 100)}": t for t in COVERAGE_TARGETS}
    if budget is not None:
        targets["budget"] = budget
    reached: dict[str, int | None] = dict.fromkeys(targets)
    for check in checks:
        for label, target in targets.items():
            if reached[label] is None and check.coverage_achieved + 1e-9 >= target:
                reached[label] = check.cumulative_agents
    return reached


def _fill_early_stop(checks: list[BatchRecord], stop_agents: int, budget: float | None):
    """Record earliest-stop and agents-saved for each coverage target on every check."""
    reached = _target_eval(checks, budget)
    for check in checks:
        check.earliest_stop = reached
        check.agents_saved = {label: (stop_agents - idx) if idx is not None else None for label, idx in reached.items()}


def _scalar_equality_stop(checks: list[BatchRecord]) -> int | None:
    """Return the first agent index where the observed scalar equals the true scalar."""
    for check in checks:
        if (
            check.observed_value is not None
            and check.true_value is not None
            and check.observed_value == check.true_value
        ):
            return check.cumulative_agents
    return None


@dataclass(slots=True)
class _CompoundRule:
    """A canonical filter predicate whose literals come from a dependency step's answer."""

    dependency_step_id: str
    """Step whose answer provides the filter values"""
    column: str
    """Column of the dependency's answer holding the filter values"""
    literals: frozenset[str]
    """Literal values of the predicate in the canonical SQL"""
    exact: bool
    """Whether the literals are the dependency's full truth (list) rather than a derived subset"""


@dataclass(slots=True)
class _StepSource:
    """Data, SQL and ground truth of one multi-agent step."""

    step: MultiAgentQueryPlanStep
    """The evaluated step"""
    canonical_sql: str
    """Step SQL assuming all previous steps were complete"""
    table: str
    """Canonical table queried by the step"""
    key_cols: list[str]
    """Key columns of the step's aligned schema"""
    value_column: str | None
    """Aggregated value column, if any"""
    aggregations: list[AggregationType]
    """Aggregation types of the step"""
    agents: list[DataAgent]
    """Agents of the step"""
    canonical_cache: dict[str, pd.DataFrame | None]
    """Per-agent results of the canonical SQL"""
    e2e_frame: pd.DataFrame
    """Deduplicated truth rows of the canonical SQL"""
    rules: list[_CompoundRule] = field(default_factory=list)
    """Filter predicates rewritten from dependency answers"""
    sql: str | None = None
    """SQL actually used for the agents, set when the step starts"""
    data_cache: dict[str, pd.DataFrame | None] | None = None
    """Per-agent results of the used SQL"""
    frame: pd.DataFrame | None = None
    """Deduplicated truth rows of the used SQL"""
    compound_source: str | None = None
    """Where the rewritten filter values came from"""

    @property
    def aggregation(self) -> AggregationType | None:
        """Primary aggregation type."""
        return self.aggregations[0] if self.aggregations else None


def _query_agents(source: _StepSource, sql: str, meta_mapping: dict[str, Any], data_dir: Path):
    """Run sql on every agent of the step in parallel."""
    with ThreadPoolExecutor() as pool:
        futures = {
            agent.name: pool.submit(query_agent_canonical, agent, sql, source.table, meta_mapping, data_dir)
            for agent in source.agents
        }
        return {name: future.result() for name, future in futures.items()}


def _compound_rules(
    sql: str,
    dependency_ids: list[str],
    sources: dict[str, _StepSource],
) -> list[_CompoundRule]:
    """Find canonical filter predicates whose literals are (a subset of) a dependency's truth column."""
    rules: list[_CompoundRule] = []
    predicates = set(literal_predicate_sets(sql))
    for literals in predicates:
        for dep_id in dependency_ids:
            dep = sources.get(dep_id)
            if dep is None or dep.e2e_frame.empty:
                continue
            match = next(
                (
                    (col, literals == values)
                    for col in dep.e2e_frame.columns
                    if literals <= (values := frozenset(literal_key(v) for v in dep.e2e_frame[col].dropna().unique()))
                ),
                None,
            )
            if match is not None:
                rules.append(_CompoundRule(dep_id, match[0], literals, match[1]))
                break
    return rules


def _dependency_values(
    rule: _CompoundRule,
    produced: pd.DataFrame | None,
    observed: pd.DataFrame | None,
    truth_values: frozenset[str],
) -> tuple[list[Any], str] | None:
    """Values of a dependency's answer for a compound rule, with their source."""
    if produced is not None and not produced.empty:
        column = rule.column if rule.column in produced.columns else None
        if column is None:
            overlaps = {c: len({literal_key(v) for v in produced[c].dropna()} & truth_values) for c in produced.columns}
            best = max(overlaps, key=lambda c: overlaps[c], default=None)
            column = best if best is not None and overlaps[best] > 0 else None
        if column is not None:
            return produced[column].dropna().drop_duplicates().tolist(), "produced"
    if rule.exact and observed is not None and rule.column in observed.columns:
        return observed[rule.column].dropna().drop_duplicates().tolist(), "observed"
    return None


def _as_literal_values(values: list[Any], literals: frozenset[str]) -> list[Any]:
    """Coerce values to numbers when the canonical literals were numeric."""

    def is_number(text: str) -> bool:
        """Whether a literal string is numeric."""
        try:
            float(text)
        except ValueError:
            return False
        return True

    if not all(is_number(lit) for lit in literals):
        return [str(v) for v in values]
    numbers: list[Any] = []
    for v in values:
        try:
            number = float(v)
        except TypeError, ValueError:
            continue
        numbers.append(int(number) if number.is_integer() else number)
    return numbers


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
        self._e2e_keys: dict[str, set[tuple]] = {}
        self._key_cols: dict[str, list[str]] = {}
        self._agents_total: dict[str, int] = {}
        self._aggregations: dict[str, list[AggregationType]] = {}
        self._step_budget_t: dict[str, float] = {}
        self._produced: dict[str, pd.DataFrame] = {}
        self._used_estimator: dict[str, bool] = {}
        self._last_result_list: dict[str, pd.DataFrame] = {}

    def register_step(
        self,
        step_id: str,
        step_query: str,
        key_cols: list[str],
        total_unique: int,
        e2e_keys: set[tuple],
        agents_total: int,
        aggregations: list[AggregationType],
    ):
        """Declare a step's query, key columns, ground truth and aggregations."""
        self._query_to_step[step_query] = step_id
        self._key_cols[step_id] = key_cols
        self._total_unique[step_id] = total_unique
        self._e2e_keys[step_id] = e2e_keys
        self._agents_total[step_id] = agents_total
        self._aggregations[step_id] = aggregations

    def _aggregation(self, step_id: str) -> AggregationType | None:
        """Primary aggregation of a step."""
        aggregations = self._aggregations.get(step_id) or []
        return aggregations[0] if aggregations else None

    def _coverage(self, step_id: str) -> float:
        """Fraction of unique keys seen for a step."""
        seen = self._seen_keys.get(step_id, set())
        total = self._total_unique.get(step_id, 0)
        return min(len(seen) / total, 1.0) if total > 0 else 0.0

    def _e2e_coverage(self, step_id: str) -> float | None:
        """Fraction of end-to-end truth keys seen for a step."""
        e2e = self._e2e_keys.get(step_id)
        if not e2e:
            return None
        seen = self._seen_keys.get(step_id, set())
        return len(seen & e2e) / len(e2e)

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
        agg = self._aggregation(step_id)
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
                e2e_coverage_achieved=self._e2e_coverage(step_id),
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
        snapshot = _estimator_snapshot(estimator, self._aggregation(step_id))

        if result is not None:
            decision = result.stop_reason.value
        elif force_result:
            decision = StopReason.ALL_AGENTS.value
        else:
            decision = "query_more_agents"
        is_stop = decision in {r.value for r in StopReason}
        if step_budget is not None:
            self._step_budget_t[step_id] = step_budget.t_step
        if aligned_result_list is not None:
            self._last_result_list[step_id] = aligned_result_list.data

        batches = self._batches.setdefault(step_id, [])
        if not batches:
            return
        config = context.get_config().completeness
        lwk = config.llm_world_knowledge
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
        last.estimator_active = config.estimator.enabled and agents_queried >= config.estimator.start_agents
        last.long_list = len(self._seen_keys.get(step_id, set())) > lwk.long_list_start
        last.observed_by_aggregation = {
            agg.value: _value_metrics(estimator, agg)[0] if estimator is not None else None
            for agg in self._aggregations.get(step_id, [])
        }
        if last.llm_decision is not None:
            last.llm_status = last.llm_decision
        elif force_result:
            last.llm_status = "all_agents"
        elif aligned_result_list is None or aligned_result_list.data.empty:
            last.llm_status = "no_samples"
        elif decision == StopReason.ESTIMATOR.value:
            last.llm_status = "estimator_stopped"
        elif not lwk.enabled:
            last.llm_status = "disabled"
        elif last.long_list and lwk.long_list_behavior == "prevent":
            last.llm_status = "long_list_prevented"
        else:
            last.llm_status = "not_called"

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
        snapshot = _estimator_snapshot(estimator, self._aggregation(step_id))
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
        result: bool | None,
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
        last.llm_decision = "undecided" if result is None else "stop" if result else "continue"

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
        """Write the run's Markdown transcript and raw input JSON (data-agent runs are skipped)."""
        del execution_id, usage
        if role == AgentRole.EXECUTION_DATA_AGENT_QUERY:
            return
        meta = transcript_mod.RunMeta(role=role, agent_name=agent_name, label=label)
        filename = transcript_mod.transcript_filename(meta)
        step_id = context.get_step_id()
        if step_id is not None and label is None:
            filename = f"{step_id}_{filename}"
        markdown = transcript_mod.render_run_markdown(meta, continuation_input, output)
        (self._output_dir / filename).write_text(markdown, encoding="utf-8")
        (self._output_dir / (filename[:-3] + ".json")).write_text(
            json.dumps(
                {"continuation_input": continuation_input, "output": _json_safe(output)},
                indent=2,
                default=str,
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
        result: pd.DataFrame | None,
    ):
        """Record the result-producer's final answer for its step (via step query)."""
        del aligned_result_list, merge_task, stop_reason, execution_id
        step_id = self._query_to_step.get(query)
        if step_id is None:
            return
        if isinstance(result, pd.DataFrame):
            self._produced[step_id] = result
        self._used_estimator[step_id] = estimator is not None

    def produced_for(self, step_id: str) -> pd.DataFrame | None:
        """Return the recorded produced answer for a step."""
        return self._produced.get(step_id)

    def used_estimator(self, step_id: str) -> bool:
        """Return whether the result producer was given the estimator for a step."""
        return self._used_estimator.get(step_id, False)

    def last_result_list(self, step_id: str) -> pd.DataFrame | None:
        """Return the aligned result list at a step's last completeness check."""
        return self._last_result_list.get(step_id)

    def step_budget_t(self, step_id: str) -> float | None:
        """Return the recorded step completeness threshold for a step."""
        return self._step_budget_t.get(step_id)

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
    source: _StepSource,
    tracker: _StepTracker,
    query_slug: str,
    output_dir: Path,
) -> StepResult | None:
    """Build one step's result from the captured trajectory + ground truth."""
    step = source.step
    batches = tracker.batches_for(step.id)
    if source.frame is None or not batches:
        logging.warning("%s/%s: no batch records", query_slug, step.id)
        return None

    ground_truth = compute_ground_truth(source.frame, source.value_column)
    e2e_ground_truth = compute_ground_truth(source.e2e_frame, source.value_column)
    keys = truth_keys(source.frame, source.key_cols)
    agents_total = len(step.to_agent)
    aggregation = source.aggregation
    step_budget_t = tracker.step_budget_t(step.id)
    true_value = _true_value_for(aggregation, ground_truth)
    e2e_true_value = _true_value_for(aggregation, e2e_ground_truth)
    uses_value = _uses_value(aggregation)
    truth_metric = true_value if uses_value else float(ground_truth.count)
    e2e_truth_metric = e2e_true_value if uses_value else float(e2e_ground_truth.count)

    for batch in batches:
        batch.true_value = true_value
        batch.e2e_true_count = e2e_ground_truth.count
        batch.e2e_true_value = e2e_true_value
        if uses_value:
            observed_metric, estimated_metric = batch.observed_value, batch.estimated_value
        else:
            observed_metric, estimated_metric = batch.observed_count, batch.estimated_count
        batch.observed_error = _rel_err(truth_metric, observed_metric)
        batch.estimated_error = _rel_err(truth_metric, estimated_metric)
        batch.e2e_observed_error = _rel_err(e2e_truth_metric, observed_metric)

    stop_batch = batches[-1]
    agents_queried_at_stop = stop_batch.cumulative_agents
    stop_fraction = agents_queried_at_stop / agents_total if agents_total > 0 else 0.0
    early_stop = agents_queried_at_stop < agents_total

    _fill_early_stop(batches, agents_queried_at_stop, step_budget_t)
    if uses_value:
        earliest_scalar = _scalar_equality_stop(batches)
        for batch in batches:
            batch.earliest_scalar_stop_agents = earliest_scalar

    produced = tracker.produced_for(step.id)
    produced_text = (
        json.dumps(result_type.serialize(produced, context="json"), default=str) if produced is not None else None
    )
    produced_scalar, produced_value_source = _produced_value(produced, source)
    precision, recall = _produced_key_metrics(produced, source.key_cols, keys)

    logging.info(
        "%s/%s: stop=%d/%d (%.0f%%) coverage=%.3f count_est=%s true=%d e2e_true=%d compounded=%s produced=%s",
        query_slug,
        step.id,
        agents_queried_at_stop,
        agents_total,
        stop_fraction * 100,
        stop_batch.coverage_achieved,
        f"{stop_batch.estimated_count:.1f}" if stop_batch.estimated_count is not None else "n/a",
        ground_truth.count,
        e2e_ground_truth.count,
        source.sql != source.canonical_sql,
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
        e2e_ground_truth=e2e_ground_truth,
        step_sql=source.sql or source.canonical_sql,
        compounded=source.sql is not None and source.sql != source.canonical_sql,
        compound_source=source.compound_source,
        checks=batches,
        aggregation=aggregation,
        step_budget_t=step_budget_t,
        produced_result=produced_text,
        produced_scalar=produced_scalar,
        produced_value_source=produced_value_source,
        produced_rel_err=_rel_err(true_value, produced_scalar),
        e2e_produced_rel_err=_rel_err(e2e_true_value, produced_scalar),
        produced_precision=precision,
        produced_recall=recall,
        produced_used_estimator=tracker.used_estimator(step.id),
    )
    save_step_results(result, query_slug, output_dir)
    return result


class _CompletenessRunner:
    """Hook-only completeness evaluation for one query."""

    def __init__(self):
        """Initialize the hook-only evaluation runner."""
        self.tracker: _StepTracker | None = None
        self.sources: dict[str, _StepSource] = {}
        self.relevant_data_agents: list[DataAgent] = []
        self.all_agents: list[DataAgent] = []
        self.meta_mapping: dict[str, Any] = {}
        self.data_dir: Path | None = None
        self.query: str = ""
        self.query_slug: str = ""
        self.output_dir: Path | None = None
        self.extra: dict[str, Any] = {}
        self.final_step_id: str = ""
        self._align_dispatch: Callable[..., AlignedResultList] | None = None
        self._data_dispatch: Callable[..., Awaitable[PartialQueryAnswer]] | None = None
        self._finalized = False

    def _prepare(self, source: _StepSource):
        """Fix the step's SQL from its dependencies' answers, load its data and register its truth."""
        if source.sql is not None:
            return
        assert self.tracker is not None and self.data_dir is not None
        sql = source.canonical_sql
        origins: list[str] = []
        for rule in source.rules:
            dep = self.sources[rule.dependency_step_id]
            dep_frame = dep.frame if dep.frame is not None else dep.e2e_frame
            truth_values = (
                frozenset(literal_key(v) for v in dep_frame[rule.column].dropna().unique())
                if rule.column in dep_frame.columns
                else frozenset()
            )
            found = _dependency_values(
                rule,
                self.tracker.produced_for(rule.dependency_step_id),
                self.tracker.last_result_list(rule.dependency_step_id),
                truth_values,
            )
            if found is None:
                origins.append("none")
                continue
            values, origin = found
            origins.append(origin)
            new_values = _as_literal_values(values, rule.literals)
            if frozenset(literal_key(v) for v in new_values) != rule.literals:
                sql = replace_literal_predicates(sql, rule.literals, new_values)
        source.sql = sql
        source.compound_source = ",".join(origins) if origins else None
        if sql == source.canonical_sql:
            source.data_cache, source.frame = source.canonical_cache, source.e2e_frame
        else:
            logging.info("%s/%s: compounded SQL %s", self.query_slug, source.step.id, sql)
            source.data_cache = _query_agents(source, sql, self.meta_mapping, self.data_dir)
            source.frame = truth_frame(source.data_cache, source.key_cols, source.value_column)
        self.tracker.register_step(
            step_id=source.step.id,
            step_query=source.step.query,
            key_cols=source.key_cols,
            total_unique=len(source.frame),
            e2e_keys=truth_keys(source.e2e_frame, source.key_cols),
            agents_total=len(source.step.to_agent),
            aggregations=source.aggregations,
        )

    def _data_handler(self, source: _StepSource) -> Callable[..., PartialQueryAnswer]:
        """Build the DATA_AGENT handler serving a step's (possibly compounded) agent results."""

        def _handler(
            execution_id: str,
            data_agent: DataAgent,
            query: str,
            additional_information: list[AdditionalInformation],
            schema_metadata: AlignedSchemaMetadata | None,
        ) -> PartialQueryAnswer:
            """Serve the step's cached result for the queried agent."""
            del execution_id, query, additional_information
            self._prepare(source)
            assert source.data_cache is not None
            df = source.data_cache.get(data_agent.name)
            if df is None or df.empty:
                columns = schema_metadata.columns if schema_metadata is not None else source.key_cols
                return PartialQueryAnswer().set_answer(pd.DataFrame(columns=columns))
            return PartialQueryAnswer().set_answer(df.copy())

        return _handler

    async def before(self, ctx: ScriptContext):
        """Pre-build canonical data caches/ground truth and register all hooks."""
        extra = ctx.extra or {}
        assert isinstance(ctx.data, CollaborativeQueryAnswering)
        assert ctx.output_dir is not None
        assert ctx.dataset_dir is not None
        assert ctx.data_dir is not None

        self.extra = extra
        self.query_slug = ctx.input_file.stem
        steps_metadata: dict[str, Any] = extra.get("steps_metadata", {})

        raw_query_plan = extra.get("query_plan")
        if raw_query_plan is None:
            raise RuntimeError(f"{self.query_slug}: no query_plan in extra — cannot evaluate")
        query_plan = QueryPlan.model_validate(raw_query_plan)
        self.final_step_id = query_plan.steps[-1].id

        multi_steps = [s for s in query_plan.steps if isinstance(s, MultiAgentQueryPlanStep)]
        if not multi_steps:
            raise RuntimeError(f"{self.query_slug}: no multi-agent steps — nothing to evaluate")

        self.output_dir = ctx.output_dir
        self.data_dir = ctx.data_dir
        self.all_agents = ctx.data.data_agents
        relevant_names: set[str] = set(extra.get("relevant_data_agents", []))
        self.relevant_data_agents = [a for a in self.all_agents if a.name in relevant_names]
        self.query = query_plan.query
        self.meta_mapping = load_meta_mapping(ctx.dataset_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tracker = _StepTracker(self.output_dir)

        for step in multi_steps:
            step_meta = steps_metadata.get(step.id)
            if step_meta is None:
                logging.warning("%s/%s: no steps_metadata — skipping", self.query_slug, step.id)
                continue
            canonical_sql: str = step_meta["sql_query"]
            schema_meta = step_meta["aligned_schema_metadata"]
            canonical_table = parse_canonical_table(canonical_sql)
            if canonical_table is None:
                logging.warning("%s/%s: could not parse table — skipping", self.query_slug, step.id)
                continue
            step_agent_names = set(step.to_agent)
            source = _StepSource(
                step=step,
                canonical_sql=canonical_sql,
                table=canonical_table,
                key_cols=schema_meta["keys"],
                value_column=schema_meta.get("value_column"),
                aggregations=[AggregationType(a) for a in schema_meta["aggregation_types"]],
                agents=[a for a in self.relevant_data_agents if a.name in step_agent_names],
                canonical_cache={},
                e2e_frame=pd.DataFrame(),
            )
            source.canonical_cache = _query_agents(source, canonical_sql, self.meta_mapping, self.data_dir)
            source.e2e_frame = truth_frame(source.canonical_cache, source.key_cols, source.value_column)
            self.sources[step.id] = source
            _per_step_align[step.query] = make_align_schema_override(step_meta, self.query)
            _per_step_data[step.query] = self._data_handler(source)

        for source in self.sources.values():
            source.rules = _compound_rules(
                source.canonical_sql, [d.step_id for d in source.step.dependencies], self.sources
            )

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
        for event, handler in self._overwrites():
            hooks.register_overwrite(event, handler)
        for event, handler in self._posts():
            hooks.register_post(event, handler)

    def _overwrites(self) -> list[tuple[HookEvent, Callable[..., Any]]]:
        """Overwrite hooks installed for the evaluation."""
        assert self._align_dispatch is not None and self._data_dispatch is not None
        return [
            (
                HookEvent.EXECUTION_EXECUTE_QUERY_PLAN_EXECUTE_STEP_ANSWER_STEP_MULTIPLE_AGENTS_ALIGN_SCHEMA,
                self._align_dispatch,
            ),
            (HookEvent.EXECUTION_QUERY_DATA_AGENT, self._data_dispatch),
            (HookEvent.EXECUTION_REPLAN_DRAFT_QUERY_PLAN, _replan_error),
        ]

    def _posts(self) -> list[tuple[HookEvent, Callable[..., Any]]]:
        """Post hooks installed for the evaluation."""
        assert self.tracker is not None
        return [
            (HookEvent.EXECUTION_QUERY_DATA_AGENT, self.tracker.on_agent_query),
            (
                HookEvent.EXECUTION_EXECUTE_QUERY_PLAN_EXECUTE_STEP_ANSWER_STEP_MULTIPLE_AGENTS_COMPLETENESS,
                self.tracker.on_completeness,
            ),
            (
                HookEvent.EXECUTION_EXECUTE_QUERY_PLAN_EXECUTE_STEP_ANSWER_STEP_MULTIPLE_AGENTS_COMPLETENESS_ESTIMATOR,
                self.tracker.on_estimator,
            ),
            (
                HookEvent.EXECUTION_EXECUTE_QUERY_PLAN_EXECUTE_STEP_ANSWER_STEP_MULTIPLE_AGENTS_COMPLETENESS_LLM,
                self.tracker.on_llm,
            ),
            (
                HookEvent.EXECUTION_EXECUTE_QUERY_PLAN_EXECUTE_STEP_ANSWER_STEP_MULTIPLE_AGENTS_RESULT_PRODUCER,
                self.tracker.on_produce_result,
            ),
            (HookEvent.AGENT_COMPLETED, self.tracker.on_llm_completed),
        ]

    async def after(self, ctx: ScriptContext, result: SynthesizedResult):
        """Build per-step results and the query summary from the trajectory."""
        del ctx
        self.finalize(result, None)

    def finalize(self, result: SynthesizedResult | None, error: BaseException | None):
        """Unregister hooks, write per-step results and the query summary (also after a failure)."""
        if self._finalized or self.tracker is None or self.output_dir is None:
            return
        self._finalized = True
        if self._align_dispatch is not None:
            for event, handler in self._overwrites():
                hooks.unregister_overwrite(event, handler)
            for event, handler in self._posts():
                hooks.unregister_post(event, handler)
        for source in self.sources.values():
            _per_step_align.pop(source.step.query, None)
            _per_step_data.pop(source.step.query, None)

        self.tracker.fill_batch_derived()
        steps_built: list[str] = []
        for source in self.sources.values():
            try:
                if _build_step_result(source, self.tracker, self.query_slug, self.output_dir) is not None:
                    steps_built.append(source.step.id)
            except Exception:
                logging.exception("%s/%s: failed to build step result", self.query_slug, source.step.id)
        self._write_query_summary(result, error, steps_built)

    def _write_query_summary(
        self,
        result: SynthesizedResult | None,
        error: BaseException | None,
        steps_built: list[str],
    ):
        """Write the query's status and final answer next to the full (ground-truth) answer."""
        assert self.output_dir is not None and self.tracker is not None
        final_scalar = _produced_scalar(result.result) if result is not None else None
        final_source = self.sources.get(self.final_step_id)
        if final_scalar is None and final_source is not None:
            final_scalar, _ = _produced_value(self.tracker.produced_for(self.final_step_id), final_source)
        full_answer = self.extra.get("full_answer")
        full_scalar = (
            float(full_answer) if isinstance(full_answer, (int, float)) and not isinstance(full_answer, bool) else None
        )
        summary = {
            "query": self.query,
            "status": "failed" if error is not None else "complete",
            "error": "".join(traceback.format_exception(error)) if error is not None else None,
            "steps_total": len(self.sources),
            "steps_built": steps_built,
            "compounded_steps": [s.step.id for s in self.sources.values() if s.sql not in (None, s.canonical_sql)],
            "full_answer": full_answer,
            "final_result": result_type.serialize(result.result, context="json") if result is not None else None,
            "final_summary": result.summary if result is not None else None,
            "combined_completeness": result.combined_completeness if result is not None else None,
            "final_scalar": final_scalar,
            "final_rel_err": _rel_err(full_scalar, final_scalar),
        }
        (self.output_dir / "query_summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
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
    del data, extra
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
    try:
        await run_script(
            args=args,
            before_fn=runner.before,
            after_fn=runner.after,
            config=config,
            open_context=False,
        )
    except Exception as error:
        runner.finalize(None, error)
        raise

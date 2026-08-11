from collections import defaultdict
from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal, Protocol, overload
from uuid import uuid4

if TYPE_CHECKING:
    from qcp_adaptive_termination.models import DataAgent, EstimatorStatistics, ScoredDataAgent
    from qcp_adaptive_termination.models.estimation import AlignedResultList
    from qcp_adaptive_termination.models.execution import (
        AdditionalInformation,
        EvaluatorDecision,
        ExecutionResult,
        FailedExecution,
        PartialQueryAnswer,
    )
    from qcp_adaptive_termination.models.query_plan import (
        MultiAgentQueryPlanStep,
        QueryPlan,
        QueryPlanStep,
    )
    from qcp_adaptive_termination.models.stages import (
        CollaborativeQueryAnsweringResult,
        DraftedQueryPlan,
        FilteredRelevantAgents,
    )

_handlers: defaultdict[str, list[Callable[..., Any]]] = defaultdict(list)
_execution_id: ContextVar[str] = ContextVar("execution_id", default="")


@contextmanager
def stage_context(execution_id: str | None = None) -> Generator[str]:
    id_ = execution_id or uuid4().hex[:12]
    token = _execution_id.set(id_)
    try:
        yield id_
    finally:
        _execution_id.reset(token)


def get_execution_id() -> str:
    return _execution_id.get()


class HookEvent(StrEnum):
    FILTER_RELEVANT_AGENTS_START = "filter_relevant_agents_start"
    FILTER_RELEVANT_AGENTS_END = "filter_relevant_agents_end"
    DRAFT_QUERY_PLAN_START = "draft_query_plan_start"
    DRAFT_QUERY_PLAN_END = "draft_query_plan_end"
    EXECUTE_QUERY_PLAN_LOOP_START = "execute_query_plan_loop_start"
    EXECUTE_QUERY_PLAN_LOOP_END = "execute_query_plan_loop_end"
    EXECUTE_QUERY_PLAN_START = "execute_query_plan_start"
    EXECUTE_QUERY_PLAN_END = "execute_query_plan_end"
    REPLAN_START = "replan_start"
    REPLAN_END = "replan_end"
    EXECUTE_STEP_START = "execute_step_start"
    EXECUTE_STEP_END = "execute_step_end"
    DATA_AGENT_QUERY_START = "data_agent_query_start"
    DATA_AGENT_QUERY_END = "data_agent_query_end"
    MULTI_AGENT_STEP_START = "multi_agent_step_start"
    MULTI_AGENT_STEP_BATCH_START = "multi_agent_step_batch_start"
    MULTI_AGENT_STEP_BATCH_END = "multi_agent_step_batch_end"
    MULTI_AGENT_STEP_END = "multi_agent_step_end"


class FILTER_RELEVANT_AGENTS_START_HANDLER(Protocol):
    def __call__(self, execution_id: str, query: str, data_agents: list[DataAgent]) -> None: ...


class FILTER_RELEVANT_AGENTS_END_HANDLER(Protocol):
    def __call__(self, execution_id: str, filtered_relevant_agents: FilteredRelevantAgents) -> None: ...


class DRAFT_QUERY_PLAN_START_HANDLER(Protocol):
    def __call__(
        self, execution_id: str, query: str, data_agents: list[DataAgent], relevant_data_agents: list[DataAgent]
    ) -> None: ...


class DRAFT_QUERY_PLAN_END_HANDLER(Protocol):
    def __call__(self, execution_id: str, drafted_query_plan: DraftedQueryPlan) -> None: ...


class EXECUTE_QUERY_PLAN_LOOP_START_HANDLER(Protocol):
    def __call__(
        self,
        execution_id: str,
        query_plan: QueryPlan,
        current_data_agents: list[DataAgent],
    ) -> None: ...


class EXECUTE_QUERY_PLAN_LOOP_END_HANDLER(Protocol):
    def __call__(self, execution_id: str, result: CollaborativeQueryAnsweringResult) -> None: ...


class EXECUTE_QUERY_PLAN_START_HANDLER(Protocol):
    def __call__(
        self,
        execution_id: str,
        attempt: int,
        query_plan: QueryPlan,
        current_data_agents: list[DataAgent],
    ) -> None: ...


class EXECUTE_QUERY_PLAN_END_HANDLER(Protocol):
    def __call__(self, execution_id: str, attempt: int, result: ExecutionResult) -> None: ...


class REPLAN_START_HANDLER(Protocol):
    def __call__(
        self,
        execution_id: str,
        attempt: int,
        current_data_agents: list[DataAgent],
        data_agents: list[DataAgent],
        query_plan: QueryPlan,
        failed_execution: FailedExecution,
    ) -> None: ...


class REPLAN_END_HANDLER(Protocol):
    def __call__(
        self,
        execution_id: str,
        attempt: int,
        updated_data_agents: list[DataAgent],
        new_query_plan: QueryPlan,
    ) -> None: ...


class EXECUTE_STEP_START_HANDLER(Protocol):
    def __call__(self, execution_id: str, query_plan_step: QueryPlanStep) -> None: ...


class EXECUTE_STEP_END_HANDLER(Protocol):
    def __call__(
        self, execution_id: str, query_plan_step: QueryPlanStep, partial_answer: PartialQueryAnswer
    ) -> None: ...


class DATA_AGENT_QUERY_START_HANDLER(Protocol):
    def __call__(self, execution_id: str, data_agent: DataAgent, query: str) -> None: ...


class DATA_AGENT_QUERY_END_HANDLER(Protocol):
    def __call__(
        self, execution_id: str, data_agent: DataAgent, query: str, partial_answer: PartialQueryAnswer
    ) -> None: ...


class MULTI_AGENT_STEP_START_HANDLER(Protocol):
    def __call__(
        self,
        execution_id: str,
        query_plan_step: MultiAgentQueryPlanStep,
        additional_information: list[AdditionalInformation],
        initial_ranking: list[ScoredDataAgent],
    ) -> None: ...


class MULTI_AGENT_STEP_BATCH_START_HANDLER(Protocol):
    def __call__(
        self,
        execution_id: str,
        query_plan_step: MultiAgentQueryPlanStep,
        additional_information: list[AdditionalInformation],
        iteration: int,
        batch: list[ScoredDataAgent],
        queried: list[str],
        remaining: list[ScoredDataAgent],
        result_list: AlignedResultList,
    ) -> None: ...


class MULTI_AGENT_STEP_BATCH_END_HANDLER(Protocol):
    def __call__(
        self,
        execution_id: str,
        query_plan_step: MultiAgentQueryPlanStep,
        additional_information: list[AdditionalInformation],
        iteration: int,
        batch: list[ScoredDataAgent],
        remaining: list[ScoredDataAgent],
        result_list: AlignedResultList,
        estimators: list[EstimatorStatistics | None],
        decision: EvaluatorDecision | Literal["need"],
    ) -> None: ...


class MULTI_AGENT_STEP_END_HANDLER(Protocol):
    def __call__(
        self,
        execution_id: str,
        query_plan_step: MultiAgentQueryPlanStep,
        additional_information: list[AdditionalInformation],
        result_list: AlignedResultList,
        queried: list[str],
        remaining: list[str],
        estimators: list[EstimatorStatistics | None],
    ) -> None: ...


@overload
def on(
    event: Literal[HookEvent.FILTER_RELEVANT_AGENTS_START],
) -> Callable[[FILTER_RELEVANT_AGENTS_START_HANDLER], FILTER_RELEVANT_AGENTS_START_HANDLER]: ...
@overload
def on(
    event: Literal[HookEvent.FILTER_RELEVANT_AGENTS_END],
) -> Callable[[FILTER_RELEVANT_AGENTS_END_HANDLER], FILTER_RELEVANT_AGENTS_END_HANDLER]: ...
@overload
def on(
    event: Literal[HookEvent.DRAFT_QUERY_PLAN_START],
) -> Callable[[DRAFT_QUERY_PLAN_START_HANDLER], DRAFT_QUERY_PLAN_START_HANDLER]: ...
@overload
def on(
    event: Literal[HookEvent.DRAFT_QUERY_PLAN_END],
) -> Callable[[DRAFT_QUERY_PLAN_END_HANDLER], DRAFT_QUERY_PLAN_END_HANDLER]: ...
@overload
def on(
    event: Literal[HookEvent.EXECUTE_QUERY_PLAN_LOOP_START],
) -> Callable[[EXECUTE_QUERY_PLAN_LOOP_START_HANDLER], EXECUTE_QUERY_PLAN_LOOP_START_HANDLER]: ...
@overload
def on(
    event: Literal[HookEvent.EXECUTE_QUERY_PLAN_LOOP_END],
) -> Callable[[EXECUTE_QUERY_PLAN_LOOP_END_HANDLER], EXECUTE_QUERY_PLAN_LOOP_END_HANDLER]: ...
@overload
def on(
    event: Literal[HookEvent.EXECUTE_QUERY_PLAN_START],
) -> Callable[[EXECUTE_QUERY_PLAN_START_HANDLER], EXECUTE_QUERY_PLAN_START_HANDLER]: ...
@overload
def on(
    event: Literal[HookEvent.EXECUTE_QUERY_PLAN_END],
) -> Callable[[EXECUTE_QUERY_PLAN_END_HANDLER], EXECUTE_QUERY_PLAN_END_HANDLER]: ...
@overload
def on(
    event: Literal[HookEvent.REPLAN_START],
) -> Callable[[REPLAN_START_HANDLER], REPLAN_START_HANDLER]: ...
@overload
def on(
    event: Literal[HookEvent.REPLAN_END],
) -> Callable[[REPLAN_END_HANDLER], REPLAN_END_HANDLER]: ...
@overload
def on(
    event: Literal[HookEvent.EXECUTE_STEP_START],
) -> Callable[[EXECUTE_STEP_START_HANDLER], EXECUTE_STEP_START_HANDLER]: ...
@overload
def on(
    event: Literal[HookEvent.EXECUTE_STEP_END],
) -> Callable[[EXECUTE_STEP_END_HANDLER], EXECUTE_STEP_END_HANDLER]: ...
@overload
def on(
    event: Literal[HookEvent.DATA_AGENT_QUERY_START],
) -> Callable[[DATA_AGENT_QUERY_START_HANDLER], DATA_AGENT_QUERY_START_HANDLER]: ...
@overload
def on(
    event: Literal[HookEvent.DATA_AGENT_QUERY_END],
) -> Callable[[DATA_AGENT_QUERY_END_HANDLER], DATA_AGENT_QUERY_END_HANDLER]: ...
@overload
def on(
    event: Literal[HookEvent.MULTI_AGENT_STEP_START],
) -> Callable[[MULTI_AGENT_STEP_START_HANDLER], MULTI_AGENT_STEP_START_HANDLER]: ...
@overload
def on(
    event: Literal[HookEvent.MULTI_AGENT_STEP_BATCH_START],
) -> Callable[[MULTI_AGENT_STEP_BATCH_START_HANDLER], MULTI_AGENT_STEP_BATCH_START_HANDLER]: ...
@overload
def on(
    event: Literal[HookEvent.MULTI_AGENT_STEP_BATCH_END],
) -> Callable[[MULTI_AGENT_STEP_BATCH_END_HANDLER], MULTI_AGENT_STEP_BATCH_END_HANDLER]: ...
@overload
def on(
    event: Literal[HookEvent.MULTI_AGENT_STEP_END],
) -> Callable[[MULTI_AGENT_STEP_END_HANDLER], MULTI_AGENT_STEP_END_HANDLER]: ...
def on(event: HookEvent) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        _handlers[event].append(fn)
        return fn

    return decorator


@overload
def register(
    event: Literal[HookEvent.FILTER_RELEVANT_AGENTS_START], handler: FILTER_RELEVANT_AGENTS_START_HANDLER
) -> None: ...
@overload
def register(
    event: Literal[HookEvent.FILTER_RELEVANT_AGENTS_END], handler: FILTER_RELEVANT_AGENTS_END_HANDLER
) -> None: ...
@overload
def register(event: Literal[HookEvent.DRAFT_QUERY_PLAN_START], handler: DRAFT_QUERY_PLAN_START_HANDLER) -> None: ...
@overload
def register(event: Literal[HookEvent.DRAFT_QUERY_PLAN_END], handler: DRAFT_QUERY_PLAN_END_HANDLER) -> None: ...
@overload
def register(
    event: Literal[HookEvent.EXECUTE_QUERY_PLAN_LOOP_START], handler: EXECUTE_QUERY_PLAN_LOOP_START_HANDLER
) -> None: ...
@overload
def register(
    event: Literal[HookEvent.EXECUTE_QUERY_PLAN_LOOP_END], handler: EXECUTE_QUERY_PLAN_LOOP_END_HANDLER
) -> None: ...
@overload
def register(event: Literal[HookEvent.EXECUTE_QUERY_PLAN_START], handler: EXECUTE_QUERY_PLAN_START_HANDLER) -> None: ...
@overload
def register(event: Literal[HookEvent.EXECUTE_QUERY_PLAN_END], handler: EXECUTE_QUERY_PLAN_END_HANDLER) -> None: ...
@overload
def register(event: Literal[HookEvent.REPLAN_START], handler: REPLAN_START_HANDLER) -> None: ...
@overload
def register(event: Literal[HookEvent.REPLAN_END], handler: REPLAN_END_HANDLER) -> None: ...
@overload
def register(event: Literal[HookEvent.EXECUTE_STEP_START], handler: EXECUTE_STEP_START_HANDLER) -> None: ...
@overload
def register(event: Literal[HookEvent.EXECUTE_STEP_END], handler: EXECUTE_STEP_END_HANDLER) -> None: ...
@overload
def register(event: Literal[HookEvent.DATA_AGENT_QUERY_START], handler: DATA_AGENT_QUERY_START_HANDLER) -> None: ...
@overload
def register(event: Literal[HookEvent.DATA_AGENT_QUERY_END], handler: DATA_AGENT_QUERY_END_HANDLER) -> None: ...
@overload
def register(event: Literal[HookEvent.MULTI_AGENT_STEP_START], handler: MULTI_AGENT_STEP_START_HANDLER) -> None: ...
@overload
def register(
    event: Literal[HookEvent.MULTI_AGENT_STEP_BATCH_START], handler: MULTI_AGENT_STEP_BATCH_START_HANDLER
) -> None: ...
@overload
def register(
    event: Literal[HookEvent.MULTI_AGENT_STEP_BATCH_END], handler: MULTI_AGENT_STEP_BATCH_END_HANDLER
) -> None: ...
@overload
def register(event: Literal[HookEvent.MULTI_AGENT_STEP_END], handler: MULTI_AGENT_STEP_END_HANDLER) -> None: ...
def register(event: HookEvent, handler: Callable[..., Any]) -> None:
    _handlers[event].append(handler)


@overload
def emit(
    event: Literal[HookEvent.FILTER_RELEVANT_AGENTS_START], *, query: str, data_agents: list[DataAgent]
) -> None: ...
@overload
def emit(
    event: Literal[HookEvent.FILTER_RELEVANT_AGENTS_END], *, filtered_relevant_agents: FilteredRelevantAgents
) -> None: ...
@overload
def emit(
    event: Literal[HookEvent.DRAFT_QUERY_PLAN_START],
    *,
    query: str,
    data_agents: list[DataAgent],
    relevant_data_agents: list[DataAgent],
) -> None: ...
@overload
def emit(event: Literal[HookEvent.DRAFT_QUERY_PLAN_END], *, drafted_query_plan: DraftedQueryPlan) -> None: ...
@overload
def emit(
    event: Literal[HookEvent.EXECUTE_QUERY_PLAN_LOOP_START],
    *,
    query_plan: QueryPlan,
    current_data_agents: list[DataAgent],
) -> None: ...
@overload
def emit(
    event: Literal[HookEvent.EXECUTE_QUERY_PLAN_LOOP_END], *, result: CollaborativeQueryAnsweringResult
) -> None: ...
@overload
def emit(
    event: Literal[HookEvent.EXECUTE_QUERY_PLAN_START],
    *,
    attempt: int,
    query_plan: QueryPlan,
    current_data_agents: list[DataAgent],
) -> None: ...
@overload
def emit(event: Literal[HookEvent.EXECUTE_QUERY_PLAN_END], *, attempt: int, result: ExecutionResult) -> None: ...
@overload
def emit(
    event: Literal[HookEvent.REPLAN_START],
    *,
    attempt: int,
    current_data_agents: list[DataAgent],
    data_agents: list[DataAgent],
    query_plan: QueryPlan,
    failed_execution: FailedExecution,
) -> None: ...
@overload
def emit(
    event: Literal[HookEvent.REPLAN_END],
    *,
    attempt: int,
    updated_data_agents: list[DataAgent],
    new_query_plan: QueryPlan,
) -> None: ...
@overload
def emit(event: Literal[HookEvent.EXECUTE_STEP_START], *, query_plan_step: QueryPlanStep) -> None: ...
@overload
def emit(
    event: Literal[HookEvent.EXECUTE_STEP_END], *, query_plan_step: QueryPlanStep, partial_answer: PartialQueryAnswer
) -> None: ...
@overload
def emit(event: Literal[HookEvent.DATA_AGENT_QUERY_START], *, data_agent: DataAgent, query: str) -> None: ...
@overload
def emit(
    event: Literal[HookEvent.DATA_AGENT_QUERY_END],
    *,
    data_agent: DataAgent,
    query: str,
    partial_answer: PartialQueryAnswer,
) -> None: ...
@overload
def emit(
    event: Literal[HookEvent.MULTI_AGENT_STEP_START],
    *,
    query_plan_step: MultiAgentQueryPlanStep,
    additional_information: list[AdditionalInformation],
    initial_ranking: list[ScoredDataAgent],
) -> None: ...
@overload
def emit(
    event: Literal[HookEvent.MULTI_AGENT_STEP_BATCH_START],
    *,
    query_plan_step: MultiAgentQueryPlanStep,
    additional_information: list[AdditionalInformation],
    iteration: int,
    batch: list[ScoredDataAgent],
    queried: list[str],
    remaining: list[ScoredDataAgent],
    result_list: AlignedResultList,
) -> None: ...
@overload
def emit(
    event: Literal[HookEvent.MULTI_AGENT_STEP_BATCH_END],
    *,
    query_plan_step: MultiAgentQueryPlanStep,
    additional_information: list[AdditionalInformation],
    iteration: int,
    batch: list[ScoredDataAgent],
    remaining: list[ScoredDataAgent],
    result_list: AlignedResultList,
    estimators: list[EstimatorStatistics | None],
    decision: EvaluatorDecision | Literal["need"],
) -> None: ...
@overload
def emit(
    event: Literal[HookEvent.MULTI_AGENT_STEP_END],
    *,
    query_plan_step: MultiAgentQueryPlanStep,
    additional_information: list[AdditionalInformation],
    result_list: AlignedResultList,
    queried: list[str],
    remaining: list[str],
    estimators: list[EstimatorStatistics | None],
) -> None: ...
def emit(event: HookEvent, **kwargs: Any) -> None:
    execution_id = get_execution_id()
    for handler in _handlers[event]:
        handler(execution_id, **kwargs)


def clear(event: HookEvent | None = None) -> None:
    if event is None:
        _handlers.clear()
    else:
        _handlers.pop(event, None)

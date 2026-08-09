from collections import defaultdict
from collections.abc import Callable
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal, Protocol, overload

if TYPE_CHECKING:
    from qcp_adaptive_termination.models import DataAgent
    from qcp_adaptive_termination.models.execution import (
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
    from qcp_adaptive_termination.stages.execution.multi_agent import MultiAgentIterationRecord

_handlers: defaultdict[str, list[Callable[..., Any]]] = defaultdict(list)


class HookEvent(StrEnum):
    FILTER_RELEVANT_AGENTS_START = "filter_relevant_agents_start"
    FILTER_RELEVANT_AGENTS_END = "filter_relevant_agents_end"
    DRAFT_QUERY_PLAN_START = "draft_query_plan_start"
    DRAFT_QUERY_PLAN_END = "draft_query_plan_end"
    EXECUTE_QUERY_PLAN_START = "execute_query_plan_start"
    EXECUTE_QUERY_PLAN_END = "execute_query_plan_end"
    REPLAN_START = "replan_start"
    REPLAN_END = "replan_end"
    EXECUTE_STEP_START = "execute_step_start"
    EXECUTE_STEP_END = "execute_step_end"
    DATA_AGENT_QUERY_START = "data_agent_query_start"
    DATA_AGENT_QUERY_END = "data_agent_query_end"
    MULTI_AGENT_BATCH_START = "multi_agent_batch_start"
    MULTI_AGENT_BATCH_END = "multi_agent_batch_end"
    ITERATION_RECORDS = "iteration_records"


class FILTER_RELEVANT_AGENTS_START_HANDLER(Protocol):
    def __call__(self, query: str, data_agents: list[DataAgent]) -> None: ...


class FILTER_RELEVANT_AGENTS_END_HANDLER(Protocol):
    def __call__(self, filtered_relevant_agents: FilteredRelevantAgents) -> None: ...


class DRAFT_QUERY_PLAN_START_HANDLER(Protocol):
    def __call__(self, query: str, data_agents: list[DataAgent], relevant_data_agents: list[DataAgent]) -> None: ...


class DRAFT_QUERY_PLAN_END_HANDLER(Protocol):
    def __call__(self, drafted_query_plan: DraftedQueryPlan) -> None: ...


class EXECUTE_QUERY_PLAN_START_HANDLER(Protocol):
    def __call__(
        self, query_plan: QueryPlan, relevant_data_agents: list[DataAgent], data_agents: list[DataAgent]
    ) -> None: ...


class EXECUTE_QUERY_PLAN_END_HANDLER(Protocol):
    def __call__(self, result: CollaborativeQueryAnsweringResult) -> None: ...


class REPLAN_START_HANDLER(Protocol):
    def __call__(self, replan_attempt: int, query_plan: QueryPlan, failed_execution: FailedExecution) -> None: ...


class REPLAN_END_HANDLER(Protocol):
    def __call__(
        self, replan_attempt: int, updated_data_agents: list[DataAgent], new_query_plan: QueryPlan
    ) -> None: ...


class EXECUTE_STEP_START_HANDLER(Protocol):
    def __call__(self, query_plan_step: QueryPlanStep) -> None: ...


class EXECUTE_STEP_END_HANDLER(Protocol):
    def __call__(self, query_plan_step: QueryPlanStep, partial_answer: PartialQueryAnswer) -> None: ...


class DATA_AGENT_QUERY_START_HANDLER(Protocol):
    def __call__(self, data_agent: DataAgent, query: str) -> None: ...


class DATA_AGENT_QUERY_END_HANDLER(Protocol):
    def __call__(self, data_agent: DataAgent, query: str, partial_answer: PartialQueryAnswer) -> None: ...


class MULTI_AGENT_BATCH_START_HANDLER(Protocol):
    def __call__(
        self,
        query_plan_step: MultiAgentQueryPlanStep,
        batch: list[str],
        agents_queried: list[str],
        agents_remaining: list[str],
    ) -> None: ...


class MULTI_AGENT_BATCH_END_HANDLER(Protocol):
    def __call__(
        self,
        query_plan_step: MultiAgentQueryPlanStep,
        batch: list[str],
        iteration_records: list[MultiAgentIterationRecord],
    ) -> None: ...


class ITERATION_RECORDS_HANDLER(Protocol):
    def __call__(self, execution_id: str, records: list[Any]) -> None: ...


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
    event: Literal[HookEvent.MULTI_AGENT_BATCH_START],
) -> Callable[[MULTI_AGENT_BATCH_START_HANDLER], MULTI_AGENT_BATCH_START_HANDLER]: ...
@overload
def on(
    event: Literal[HookEvent.MULTI_AGENT_BATCH_END],
) -> Callable[[MULTI_AGENT_BATCH_END_HANDLER], MULTI_AGENT_BATCH_END_HANDLER]: ...
@overload
def on(
    event: Literal[HookEvent.ITERATION_RECORDS],
) -> Callable[[ITERATION_RECORDS_HANDLER], ITERATION_RECORDS_HANDLER]: ...
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
def register(event: Literal[HookEvent.MULTI_AGENT_BATCH_START], handler: MULTI_AGENT_BATCH_START_HANDLER) -> None: ...
@overload
def register(event: Literal[HookEvent.MULTI_AGENT_BATCH_END], handler: MULTI_AGENT_BATCH_END_HANDLER) -> None: ...
@overload
def register(event: Literal[HookEvent.ITERATION_RECORDS], handler: ITERATION_RECORDS_HANDLER) -> None: ...
def register(event: HookEvent, handler: Callable[..., Any]) -> None:
    _handlers[event].append(handler)


@overload
def emit(event: Literal[HookEvent.FILTER_RELEVANT_AGENTS_START], query: str, data_agents: list[DataAgent]) -> None: ...
@overload
def emit(
    event: Literal[HookEvent.FILTER_RELEVANT_AGENTS_END], filtered_relevant_agents: FilteredRelevantAgents
) -> None: ...
@overload
def emit(
    event: Literal[HookEvent.DRAFT_QUERY_PLAN_START],
    query: str,
    data_agents: list[DataAgent],
    relevant_data_agents: list[DataAgent],
) -> None: ...
@overload
def emit(event: Literal[HookEvent.DRAFT_QUERY_PLAN_END], drafted_query_plan: DraftedQueryPlan) -> None: ...
@overload
def emit(
    event: Literal[HookEvent.EXECUTE_QUERY_PLAN_START],
    query_plan: QueryPlan,
    relevant_data_agents: list[DataAgent],
    data_agents: list[DataAgent],
) -> None: ...
@overload
def emit(event: Literal[HookEvent.EXECUTE_QUERY_PLAN_END], result: CollaborativeQueryAnsweringResult) -> None: ...
@overload
def emit(
    event: Literal[HookEvent.REPLAN_START],
    replan_attempt: int,
    query_plan: QueryPlan,
    failed_execution: FailedExecution,
) -> None: ...
@overload
def emit(
    event: Literal[HookEvent.REPLAN_END],
    replan_attempt: int,
    updated_data_agents: list[DataAgent],
    new_query_plan: QueryPlan,
) -> None: ...
@overload
def emit(event: Literal[HookEvent.EXECUTE_STEP_START], query_plan_step: QueryPlanStep) -> None: ...
@overload
def emit(
    event: Literal[HookEvent.EXECUTE_STEP_END], query_plan_step: QueryPlanStep, partial_answer: PartialQueryAnswer
) -> None: ...
@overload
def emit(event: Literal[HookEvent.DATA_AGENT_QUERY_START], data_agent: DataAgent, query: str) -> None: ...
@overload
def emit(
    event: Literal[HookEvent.DATA_AGENT_QUERY_END],
    data_agent: DataAgent,
    query: str,
    partial_answer: PartialQueryAnswer,
) -> None: ...
@overload
def emit(
    event: Literal[HookEvent.MULTI_AGENT_BATCH_START],
    query_plan_step: MultiAgentQueryPlanStep,
    batch: list[str],
    agents_queried: list[str],
    agents_remaining: list[str],
) -> None: ...
@overload
def emit(
    event: Literal[HookEvent.MULTI_AGENT_BATCH_END],
    query_plan_step: MultiAgentQueryPlanStep,
    batch: list[str],
    iteration_records: list[MultiAgentIterationRecord],
) -> None: ...
@overload
def emit(event: Literal[HookEvent.ITERATION_RECORDS], execution_id: str, records: list[Any]) -> None: ...
def emit(event: HookEvent, *args: Any, **kwargs: Any) -> None:
    for handler in _handlers[event]:
        handler(*args, **kwargs)


def clear(event: HookEvent | None = None) -> None:
    if event is None:
        _handlers.clear()
    else:
        _handlers.pop(event, None)

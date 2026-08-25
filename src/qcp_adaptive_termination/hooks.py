import inspect
from collections import defaultdict
from collections.abc import Awaitable, Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, overload
from uuid import uuid4

if TYPE_CHECKING:
    from qcp_adaptive_termination.models import (
        AlignedSchemaMetadata,
        Config,
        DataAgent,
        EstimatorStatistics,
        ScoredDataAgent,
    )
    from qcp_adaptive_termination.models.estimation import AlignedResultList
    from qcp_adaptive_termination.models.execution import (
        AdditionalInformation,
        EvaluatorDecision,
        ExecutionResult,
        FailedExecution,
        FinalResultOutcome,
        PartialQueryAnswer,
        QueryMoreAgentsOutcome,
    )
    from qcp_adaptive_termination.models.query_plan import (
        MultiAgentQueryPlanStep,
        QueryPlan,
        QueryPlanStep,
    )
    from qcp_adaptive_termination.models.stages import (
        CollaborativeQueryAnswering,
        CollaborativeQueryAnsweringResult,
        DraftedQueryPlan,
        FilteredRelevantAgents,
    )

_handlers: defaultdict[str, list[Callable[..., Any]]] = defaultdict(list)
_execution_id: ContextVar[str] = ContextVar("execution_id", default="")


@contextmanager
def stage_context(execution_id: str | None = None) -> Generator[str]:
    parent = _execution_id.get()
    new_id = execution_id or uuid4().hex[:12]
    id_ = f"{parent}/{new_id}" if parent else new_id
    token = _execution_id.set(id_)
    try:
        yield id_
    finally:
        _execution_id.reset(token)


def get_execution_id() -> str:
    return _execution_id.get()


class HookEvent(StrEnum):
    LOAD_INPUT = "load_input"
    LOAD_CONFIG = "load_config"
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
    COORDINATOR_RELEVANCE_OVERRIDE = "coordinator_relevance_override"
    DATA_AGENT_RELEVANCE_OVERRIDE = "data_agent_relevance_override"
    FILTER_RELEVANT_AGENTS_OVERRIDE = "filter_relevant_agents_override"
    DRAFT_QUERY_PLAN_OVERRIDE = "draft_query_plan_override"
    ALIGN_SCHEMA_OVERRIDE = "align_schema_override"
    DATA_AGENT_QUERY_OVERRIDE = "data_agent_query_override"
    COMPLETENESS_EVALUATOR_OVERRIDE = "completeness_evaluator_override"
    REPLAN_QUERY_PLAN_OVERRIDE = "replan_query_plan_override"
    ANSWER_STEP_COORDINATOR_OVERRIDE = "answer_step_coordinator_override"
    ANSWER_STEP_DATA_AGENT_OVERRIDE = "answer_step_data_agent_override"
    EXECUTE_QUERY_PLAN_STEP_OVERRIDE = "execute_query_plan_step_override"
    ANSWER_STEP_MULTIPLE_AGENTS_OVERRIDE = "answer_step_multiple_agents_override"
    EMBEDDING_PRE_FILTER_OVERRIDE = "embedding_pre_filter_override"


class LOAD_INPUT_HANDLER(Protocol):
    def __call__(
        self,
        execution_id: str,
        input_file: Path,
        data: CollaborativeQueryAnswering
        | FilteredRelevantAgents
        | DraftedQueryPlan
        | CollaborativeQueryAnsweringResult,
        extra: dict[str, Any] | None,
    ) -> None: ...


class LOAD_CONFIG_HANDLER(Protocol):
    def __call__(self, execution_id: str, config: Config) -> None: ...


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


class COORDINATOR_RELEVANCE_OVERRIDE_HANDLER(Protocol):
    def __call__(
        self,
        execution_id: str,
        query: str,
        data_agents: list[DataAgent],
        registry_string: str,
        additional_context: str | None,
    ) -> list[tuple[DataAgent, str]]: ...


class DATA_AGENT_RELEVANCE_OVERRIDE_HANDLER(Protocol):
    def __call__(
        self,
        execution_id: str,
        query: str,
        item: tuple[DataAgent, str],
        additional_context: str | None,
    ) -> DataAgent | None: ...


class FILTER_RELEVANT_AGENTS_OVERRIDE_HANDLER(Protocol):
    def __call__(
        self,
        execution_id: str,
        query: str,
        data_agents: list[DataAgent],
    ) -> FilteredRelevantAgents: ...


class DRAFT_QUERY_PLAN_OVERRIDE_HANDLER(Protocol):
    def __call__(
        self,
        execution_id: str,
        query: str,
        data_agents: list[DataAgent],
        relevant_data_agents: list[DataAgent],
    ) -> DraftedQueryPlan: ...


class ALIGN_SCHEMA_OVERRIDE_HANDLER(Protocol):
    def __call__(
        self,
        execution_id: str,
        data_agents: list[DataAgent],
        agent_names: list[str],
        query: str,
        merge_task: str,
    ) -> AlignedResultList: ...


class DATA_AGENT_QUERY_OVERRIDE_HANDLER(Protocol):
    def __call__(
        self,
        execution_id: str,
        data_agent: DataAgent,
        query: str,
        additional_information: list[AdditionalInformation],
        schema_metadata: AlignedSchemaMetadata | None,
    ) -> PartialQueryAnswer: ...


class COMPLETENESS_EVALUATOR_OVERRIDE_HANDLER(Protocol):
    def __call__(
        self,
        execution_id: str,
        aligned_result_list: AlignedResultList,
        estimator_statistics: EstimatorStatistics,
        query: str,
        previous_estimator_statistics: list[EstimatorStatistics],
        merge_task: str,
        force_result: bool,
    ) -> QueryMoreAgentsOutcome | FinalResultOutcome: ...


class REPLAN_QUERY_PLAN_OVERRIDE_HANDLER(Protocol):
    def __call__(
        self,
        execution_id: str,
        data_agents: list[DataAgent],
        relevant_data_agents: list[DataAgent],
        query_plan: QueryPlan,
        reason: FailedExecution,
    ) -> QueryPlan: ...


class ANSWER_STEP_COORDINATOR_OVERRIDE_HANDLER(Protocol):
    def __call__(
        self,
        execution_id: str,
        query_plan_step: QueryPlanStep,
        additional_information: list[AdditionalInformation],
    ) -> PartialQueryAnswer: ...


class ANSWER_STEP_DATA_AGENT_OVERRIDE_HANDLER(Protocol):
    def __call__(
        self,
        execution_id: str,
        data_agents: list[DataAgent],
        query: str,
        to_agent: str,
        additional_information: list[AdditionalInformation],
        schema_metadata: AlignedSchemaMetadata | None,
    ) -> PartialQueryAnswer: ...


class EXECUTE_QUERY_PLAN_STEP_OVERRIDE_HANDLER(Protocol):
    def __call__(
        self,
        execution_id: str,
        data_agents: list[DataAgent],
        query_plan_step: QueryPlanStep,
        additional_information: list[AdditionalInformation],
    ) -> PartialQueryAnswer: ...


class ANSWER_STEP_MULTIPLE_AGENTS_OVERRIDE_HANDLER(Protocol):
    def __call__(
        self,
        execution_id: str,
        data_agents: list[DataAgent],
        query_plan_step: MultiAgentQueryPlanStep,
        additional_information: list[AdditionalInformation],
    ) -> PartialQueryAnswer: ...


class EMBEDDING_PRE_FILTER_OVERRIDE_HANDLER(Protocol):
    def __call__(
        self,
        execution_id: str,
        query: str,
        data_agents: list[DataAgent],
    ) -> list[DataAgent]: ...


@overload
def on(
    event: Literal[HookEvent.LOAD_INPUT],
) -> Callable[[LOAD_INPUT_HANDLER], LOAD_INPUT_HANDLER]: ...
@overload
def on(
    event: Literal[HookEvent.LOAD_CONFIG],
) -> Callable[[LOAD_CONFIG_HANDLER], LOAD_CONFIG_HANDLER]: ...
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
@overload
def on(
    event: Literal[HookEvent.COORDINATOR_RELEVANCE_OVERRIDE],
) -> Callable[[COORDINATOR_RELEVANCE_OVERRIDE_HANDLER], COORDINATOR_RELEVANCE_OVERRIDE_HANDLER]: ...
@overload
def on(
    event: Literal[HookEvent.DATA_AGENT_RELEVANCE_OVERRIDE],
) -> Callable[[DATA_AGENT_RELEVANCE_OVERRIDE_HANDLER], DATA_AGENT_RELEVANCE_OVERRIDE_HANDLER]: ...
@overload
def on(
    event: Literal[HookEvent.FILTER_RELEVANT_AGENTS_OVERRIDE],
) -> Callable[[FILTER_RELEVANT_AGENTS_OVERRIDE_HANDLER], FILTER_RELEVANT_AGENTS_OVERRIDE_HANDLER]: ...
@overload
def on(
    event: Literal[HookEvent.DRAFT_QUERY_PLAN_OVERRIDE],
) -> Callable[[DRAFT_QUERY_PLAN_OVERRIDE_HANDLER], DRAFT_QUERY_PLAN_OVERRIDE_HANDLER]: ...
@overload
def on(
    event: Literal[HookEvent.ALIGN_SCHEMA_OVERRIDE],
) -> Callable[[ALIGN_SCHEMA_OVERRIDE_HANDLER], ALIGN_SCHEMA_OVERRIDE_HANDLER]: ...
@overload
def on(
    event: Literal[HookEvent.DATA_AGENT_QUERY_OVERRIDE],
) -> Callable[[DATA_AGENT_QUERY_OVERRIDE_HANDLER], DATA_AGENT_QUERY_OVERRIDE_HANDLER]: ...
@overload
def on(
    event: Literal[HookEvent.COMPLETENESS_EVALUATOR_OVERRIDE],
) -> Callable[[COMPLETENESS_EVALUATOR_OVERRIDE_HANDLER], COMPLETENESS_EVALUATOR_OVERRIDE_HANDLER]: ...
@overload
def on(
    event: Literal[HookEvent.REPLAN_QUERY_PLAN_OVERRIDE],
) -> Callable[[REPLAN_QUERY_PLAN_OVERRIDE_HANDLER], REPLAN_QUERY_PLAN_OVERRIDE_HANDLER]: ...
@overload
def on(
    event: Literal[HookEvent.ANSWER_STEP_COORDINATOR_OVERRIDE],
) -> Callable[[ANSWER_STEP_COORDINATOR_OVERRIDE_HANDLER], ANSWER_STEP_COORDINATOR_OVERRIDE_HANDLER]: ...
@overload
def on(
    event: Literal[HookEvent.ANSWER_STEP_DATA_AGENT_OVERRIDE],
) -> Callable[[ANSWER_STEP_DATA_AGENT_OVERRIDE_HANDLER], ANSWER_STEP_DATA_AGENT_OVERRIDE_HANDLER]: ...
@overload
def on(
    event: Literal[HookEvent.EXECUTE_QUERY_PLAN_STEP_OVERRIDE],
) -> Callable[[EXECUTE_QUERY_PLAN_STEP_OVERRIDE_HANDLER], EXECUTE_QUERY_PLAN_STEP_OVERRIDE_HANDLER]: ...
@overload
def on(
    event: Literal[HookEvent.ANSWER_STEP_MULTIPLE_AGENTS_OVERRIDE],
) -> Callable[[ANSWER_STEP_MULTIPLE_AGENTS_OVERRIDE_HANDLER], ANSWER_STEP_MULTIPLE_AGENTS_OVERRIDE_HANDLER]: ...
@overload
def on(
    event: Literal[HookEvent.EMBEDDING_PRE_FILTER_OVERRIDE],
) -> Callable[[EMBEDDING_PRE_FILTER_OVERRIDE_HANDLER], EMBEDDING_PRE_FILTER_OVERRIDE_HANDLER]: ...
def on(event: HookEvent) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        _handlers[event].append(fn)
        return fn

    return decorator


@overload
def register(event: Literal[HookEvent.LOAD_INPUT], handler: LOAD_INPUT_HANDLER) -> None: ...
@overload
def register(event: Literal[HookEvent.LOAD_CONFIG], handler: LOAD_CONFIG_HANDLER) -> None: ...
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
@overload
def register(
    event: Literal[HookEvent.COORDINATOR_RELEVANCE_OVERRIDE],
    handler: COORDINATOR_RELEVANCE_OVERRIDE_HANDLER,
) -> None: ...
@overload
def register(
    event: Literal[HookEvent.DATA_AGENT_RELEVANCE_OVERRIDE],
    handler: DATA_AGENT_RELEVANCE_OVERRIDE_HANDLER,
) -> None: ...
@overload
def register(
    event: Literal[HookEvent.FILTER_RELEVANT_AGENTS_OVERRIDE],
    handler: FILTER_RELEVANT_AGENTS_OVERRIDE_HANDLER,
) -> None: ...
@overload
def register(
    event: Literal[HookEvent.DRAFT_QUERY_PLAN_OVERRIDE],
    handler: DRAFT_QUERY_PLAN_OVERRIDE_HANDLER,
) -> None: ...
@overload
def register(event: Literal[HookEvent.ALIGN_SCHEMA_OVERRIDE], handler: ALIGN_SCHEMA_OVERRIDE_HANDLER) -> None: ...
@overload
def register(
    event: Literal[HookEvent.DATA_AGENT_QUERY_OVERRIDE],
    handler: DATA_AGENT_QUERY_OVERRIDE_HANDLER,
) -> None: ...
@overload
def register(
    event: Literal[HookEvent.COMPLETENESS_EVALUATOR_OVERRIDE],
    handler: COMPLETENESS_EVALUATOR_OVERRIDE_HANDLER,
) -> None: ...
@overload
def register(
    event: Literal[HookEvent.REPLAN_QUERY_PLAN_OVERRIDE],
    handler: REPLAN_QUERY_PLAN_OVERRIDE_HANDLER,
) -> None: ...
@overload
def register(
    event: Literal[HookEvent.ANSWER_STEP_COORDINATOR_OVERRIDE],
    handler: ANSWER_STEP_COORDINATOR_OVERRIDE_HANDLER,
) -> None: ...
@overload
def register(
    event: Literal[HookEvent.ANSWER_STEP_DATA_AGENT_OVERRIDE],
    handler: ANSWER_STEP_DATA_AGENT_OVERRIDE_HANDLER,
) -> None: ...
@overload
def register(
    event: Literal[HookEvent.EXECUTE_QUERY_PLAN_STEP_OVERRIDE],
    handler: EXECUTE_QUERY_PLAN_STEP_OVERRIDE_HANDLER,
) -> None: ...
@overload
def register(
    event: Literal[HookEvent.ANSWER_STEP_MULTIPLE_AGENTS_OVERRIDE],
    handler: ANSWER_STEP_MULTIPLE_AGENTS_OVERRIDE_HANDLER,
) -> None: ...
@overload
def register(
    event: Literal[HookEvent.EMBEDDING_PRE_FILTER_OVERRIDE],
    handler: EMBEDDING_PRE_FILTER_OVERRIDE_HANDLER,
) -> None: ...
def register(event: HookEvent, handler: Callable[..., Any]) -> None:
    _handlers[event].append(handler)


@overload
def emit(
    event: Literal[HookEvent.LOAD_INPUT],
    *,
    input_file: Path,
    data: CollaborativeQueryAnswering | FilteredRelevantAgents | DraftedQueryPlan | CollaborativeQueryAnsweringResult,
    extra: dict[str, Any] | None,
) -> None: ...
@overload
def emit(event: Literal[HookEvent.LOAD_CONFIG], *, config: Config) -> None: ...
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


@overload
async def call_override(
    event: Literal[HookEvent.COORDINATOR_RELEVANCE_OVERRIDE],
    *,
    default: Callable[[], list[tuple[DataAgent, str]] | Awaitable[list[tuple[DataAgent, str]]]],
    query: str,
    data_agents: list[DataAgent],
    registry_string: str,
    additional_context: str | None,
) -> list[tuple[DataAgent, str]]: ...
@overload
async def call_override(
    event: Literal[HookEvent.DATA_AGENT_RELEVANCE_OVERRIDE],
    *,
    default: Callable[[], DataAgent | None | Awaitable[DataAgent | None]],
    query: str,
    item: tuple[DataAgent, str],
    additional_context: str | None,
) -> DataAgent | None: ...
@overload
async def call_override(
    event: Literal[HookEvent.FILTER_RELEVANT_AGENTS_OVERRIDE],
    *,
    default: Callable[[], FilteredRelevantAgents | Awaitable[FilteredRelevantAgents]],
    query: str,
    data_agents: list[DataAgent],
) -> FilteredRelevantAgents: ...
@overload
async def call_override(
    event: Literal[HookEvent.DRAFT_QUERY_PLAN_OVERRIDE],
    *,
    default: Callable[[], DraftedQueryPlan | Awaitable[DraftedQueryPlan]],
    query: str,
    data_agents: list[DataAgent],
    relevant_data_agents: list[DataAgent],
) -> DraftedQueryPlan: ...
@overload
async def call_override(
    event: Literal[HookEvent.ALIGN_SCHEMA_OVERRIDE],
    *,
    default: Callable[[], AlignedResultList | Awaitable[AlignedResultList]],
    data_agents: list[DataAgent],
    agent_names: list[str],
    query: str,
    merge_task: str,
) -> AlignedResultList: ...
@overload
async def call_override(
    event: Literal[HookEvent.DATA_AGENT_QUERY_OVERRIDE],
    *,
    default: Callable[[], PartialQueryAnswer | Awaitable[PartialQueryAnswer]],
    data_agent: DataAgent,
    query: str,
    additional_information: list[AdditionalInformation],
    schema_metadata: AlignedSchemaMetadata | None,
) -> PartialQueryAnswer: ...
@overload
async def call_override(
    event: Literal[HookEvent.COMPLETENESS_EVALUATOR_OVERRIDE],
    *,
    default: Callable[
        [], QueryMoreAgentsOutcome | FinalResultOutcome | Awaitable[QueryMoreAgentsOutcome | FinalResultOutcome]
    ],
    aligned_result_list: AlignedResultList,
    estimator_statistics: EstimatorStatistics,
    query: str,
    previous_estimator_statistics: list[EstimatorStatistics],
    merge_task: str,
    force_result: bool,
) -> QueryMoreAgentsOutcome | FinalResultOutcome: ...
@overload
async def call_override(
    event: Literal[HookEvent.REPLAN_QUERY_PLAN_OVERRIDE],
    *,
    default: Callable[[], QueryPlan | Awaitable[QueryPlan]],
    data_agents: list[DataAgent],
    relevant_data_agents: list[DataAgent],
    query_plan: QueryPlan,
    reason: FailedExecution,
) -> QueryPlan: ...
@overload
async def call_override(
    event: Literal[HookEvent.ANSWER_STEP_COORDINATOR_OVERRIDE],
    *,
    default: Callable[[], PartialQueryAnswer | Awaitable[PartialQueryAnswer]],
    query_plan_step: QueryPlanStep,
    additional_information: list[AdditionalInformation],
) -> PartialQueryAnswer: ...
@overload
async def call_override(
    event: Literal[HookEvent.ANSWER_STEP_DATA_AGENT_OVERRIDE],
    *,
    default: Callable[[], PartialQueryAnswer | Awaitable[PartialQueryAnswer]],
    data_agents: list[DataAgent],
    query: str,
    to_agent: str,
    additional_information: list[AdditionalInformation],
    schema_metadata: AlignedSchemaMetadata | None,
) -> PartialQueryAnswer: ...
@overload
async def call_override(
    event: Literal[HookEvent.EXECUTE_QUERY_PLAN_STEP_OVERRIDE],
    *,
    default: Callable[[], PartialQueryAnswer | Awaitable[PartialQueryAnswer]],
    data_agents: list[DataAgent],
    query_plan_step: QueryPlanStep,
    additional_information: list[AdditionalInformation],
) -> PartialQueryAnswer: ...
@overload
async def call_override(
    event: Literal[HookEvent.ANSWER_STEP_MULTIPLE_AGENTS_OVERRIDE],
    *,
    default: Callable[[], PartialQueryAnswer | Awaitable[PartialQueryAnswer]],
    data_agents: list[DataAgent],
    query_plan_step: MultiAgentQueryPlanStep,
    additional_information: list[AdditionalInformation],
) -> PartialQueryAnswer: ...
@overload
async def call_override(
    event: Literal[HookEvent.EMBEDDING_PRE_FILTER_OVERRIDE],
    *,
    default: Callable[[], list[DataAgent] | Awaitable[list[DataAgent]]],
    query: str,
    data_agents: list[DataAgent],
) -> list[DataAgent]: ...
async def call_override(event: HookEvent, *, default: Callable[[], Any], **kwargs: Any) -> Any:
    execution_id = get_execution_id()
    handlers = _handlers.get(str(event), [])
    if handlers:
        result = handlers[0](execution_id, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result
    result = default()
    if inspect.isawaitable(result):
        return await result
    return result


@overload
def unregister(event: Literal[HookEvent.LOAD_INPUT], handler: LOAD_INPUT_HANDLER) -> None: ...
@overload
def unregister(event: Literal[HookEvent.LOAD_CONFIG], handler: LOAD_CONFIG_HANDLER) -> None: ...
@overload
def unregister(
    event: Literal[HookEvent.FILTER_RELEVANT_AGENTS_START], handler: FILTER_RELEVANT_AGENTS_START_HANDLER
) -> None: ...
@overload
def unregister(
    event: Literal[HookEvent.FILTER_RELEVANT_AGENTS_END], handler: FILTER_RELEVANT_AGENTS_END_HANDLER
) -> None: ...
@overload
def unregister(event: Literal[HookEvent.DRAFT_QUERY_PLAN_START], handler: DRAFT_QUERY_PLAN_START_HANDLER) -> None: ...
@overload
def unregister(event: Literal[HookEvent.DRAFT_QUERY_PLAN_END], handler: DRAFT_QUERY_PLAN_END_HANDLER) -> None: ...
@overload
def unregister(
    event: Literal[HookEvent.EXECUTE_QUERY_PLAN_LOOP_START], handler: EXECUTE_QUERY_PLAN_LOOP_START_HANDLER
) -> None: ...
@overload
def unregister(
    event: Literal[HookEvent.EXECUTE_QUERY_PLAN_LOOP_END], handler: EXECUTE_QUERY_PLAN_LOOP_END_HANDLER
) -> None: ...
@overload
def unregister(
    event: Literal[HookEvent.EXECUTE_QUERY_PLAN_START], handler: EXECUTE_QUERY_PLAN_START_HANDLER
) -> None: ...
@overload
def unregister(event: Literal[HookEvent.EXECUTE_QUERY_PLAN_END], handler: EXECUTE_QUERY_PLAN_END_HANDLER) -> None: ...
@overload
def unregister(event: Literal[HookEvent.REPLAN_START], handler: REPLAN_START_HANDLER) -> None: ...
@overload
def unregister(event: Literal[HookEvent.REPLAN_END], handler: REPLAN_END_HANDLER) -> None: ...
@overload
def unregister(event: Literal[HookEvent.EXECUTE_STEP_START], handler: EXECUTE_STEP_START_HANDLER) -> None: ...
@overload
def unregister(event: Literal[HookEvent.EXECUTE_STEP_END], handler: EXECUTE_STEP_END_HANDLER) -> None: ...
@overload
def unregister(event: Literal[HookEvent.DATA_AGENT_QUERY_START], handler: DATA_AGENT_QUERY_START_HANDLER) -> None: ...
@overload
def unregister(event: Literal[HookEvent.DATA_AGENT_QUERY_END], handler: DATA_AGENT_QUERY_END_HANDLER) -> None: ...
@overload
def unregister(event: Literal[HookEvent.MULTI_AGENT_STEP_START], handler: MULTI_AGENT_STEP_START_HANDLER) -> None: ...
@overload
def unregister(
    event: Literal[HookEvent.MULTI_AGENT_STEP_BATCH_START], handler: MULTI_AGENT_STEP_BATCH_START_HANDLER
) -> None: ...
@overload
def unregister(
    event: Literal[HookEvent.MULTI_AGENT_STEP_BATCH_END], handler: MULTI_AGENT_STEP_BATCH_END_HANDLER
) -> None: ...
@overload
def unregister(event: Literal[HookEvent.MULTI_AGENT_STEP_END], handler: MULTI_AGENT_STEP_END_HANDLER) -> None: ...
@overload
def unregister(
    event: Literal[HookEvent.COORDINATOR_RELEVANCE_OVERRIDE],
    handler: COORDINATOR_RELEVANCE_OVERRIDE_HANDLER,
) -> None: ...
@overload
def unregister(
    event: Literal[HookEvent.DATA_AGENT_RELEVANCE_OVERRIDE],
    handler: DATA_AGENT_RELEVANCE_OVERRIDE_HANDLER,
) -> None: ...
@overload
def unregister(
    event: Literal[HookEvent.FILTER_RELEVANT_AGENTS_OVERRIDE],
    handler: FILTER_RELEVANT_AGENTS_OVERRIDE_HANDLER,
) -> None: ...
@overload
def unregister(
    event: Literal[HookEvent.DRAFT_QUERY_PLAN_OVERRIDE],
    handler: DRAFT_QUERY_PLAN_OVERRIDE_HANDLER,
) -> None: ...
@overload
def unregister(event: Literal[HookEvent.ALIGN_SCHEMA_OVERRIDE], handler: ALIGN_SCHEMA_OVERRIDE_HANDLER) -> None: ...
@overload
def unregister(
    event: Literal[HookEvent.DATA_AGENT_QUERY_OVERRIDE],
    handler: DATA_AGENT_QUERY_OVERRIDE_HANDLER,
) -> None: ...
@overload
def unregister(
    event: Literal[HookEvent.COMPLETENESS_EVALUATOR_OVERRIDE],
    handler: COMPLETENESS_EVALUATOR_OVERRIDE_HANDLER,
) -> None: ...
@overload
def unregister(
    event: Literal[HookEvent.REPLAN_QUERY_PLAN_OVERRIDE],
    handler: REPLAN_QUERY_PLAN_OVERRIDE_HANDLER,
) -> None: ...
@overload
def unregister(
    event: Literal[HookEvent.ANSWER_STEP_COORDINATOR_OVERRIDE],
    handler: ANSWER_STEP_COORDINATOR_OVERRIDE_HANDLER,
) -> None: ...
@overload
def unregister(
    event: Literal[HookEvent.ANSWER_STEP_DATA_AGENT_OVERRIDE],
    handler: ANSWER_STEP_DATA_AGENT_OVERRIDE_HANDLER,
) -> None: ...
@overload
def unregister(
    event: Literal[HookEvent.EXECUTE_QUERY_PLAN_STEP_OVERRIDE],
    handler: EXECUTE_QUERY_PLAN_STEP_OVERRIDE_HANDLER,
) -> None: ...
@overload
def unregister(
    event: Literal[HookEvent.ANSWER_STEP_MULTIPLE_AGENTS_OVERRIDE],
    handler: ANSWER_STEP_MULTIPLE_AGENTS_OVERRIDE_HANDLER,
) -> None: ...
@overload
def unregister(
    event: Literal[HookEvent.EMBEDDING_PRE_FILTER_OVERRIDE],
    handler: EMBEDDING_PRE_FILTER_OVERRIDE_HANDLER,
) -> None: ...
def unregister(event: HookEvent, handler: Callable[..., Any]) -> None:
    _handlers[event].remove(handler)


def clear(event: HookEvent | None = None) -> None:
    if event is None:
        _handlers.clear()
    else:
        _handlers.pop(event, None)

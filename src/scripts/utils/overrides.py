"""Shared hook overrides for evaluation workers.

Both evaluation packages stub the LLM-heavy parts of the pipeline
(``align_schema``, ``data_agent``, ``completeness``) and route them through
per-execution-id dispatch maps so multiple steps/strategies can run in
parallel within one process.
"""

from collections.abc import Callable
from typing import Any

from qcp_adaptive_termination.estimator import Estimator
from qcp_adaptive_termination.models import (
    AlignedResultList,
    AlignedSchemaMetadata,
    DataAgent,
)
from qcp_adaptive_termination.models.execution import (
    AdditionalInformation,
    PartialQueryAnswer,
    StopDecision,
)


def make_completeness_override() -> Callable[..., StopDecision | None]:
    """Completeness evaluator that never stops early (used by selection strategies).

    Always returns ``None`` so the caller never produces a final result via the
    LLM result producer — the eval only cares about the agent query order.
    """

    def _handler(
        execution_id: str,
        aligned_result_list: AlignedResultList,
        estimator: Estimator | None,
        query: str,
        merge_task: str,
        step_budget: Any,
        agents_queried: int,
        force_result: bool,
        **kwargs: Any,
    ) -> StopDecision | None:
        return None

    return _handler


def make_dispatch(
    handlers: dict[str, Callable[..., Any]],
    *,
    prefix_match: bool = False,
) -> Callable[..., Any]:
    """Build a dispatch handler that routes by execution_id to a per-step handler."""

    def _dispatch(execution_id: str, **kwargs: Any) -> Any:
        for handler_id, handler in handlers.items():
            if execution_id == handler_id or (prefix_match and execution_id.startswith(handler_id + "/")):
                return handler(execution_id, **kwargs)
        raise ValueError(f"No handler for execution_id={execution_id!r}")

    return _dispatch


def make_align_schema_dispatch(
    handlers: dict[str, Callable[..., AlignedResultList]],
    *,
    prefix_match: bool = False,
) -> Callable[..., AlignedResultList]:
    """Dispatch for ALIGN_SCHEMA hook events."""

    def _dispatch(
        execution_id: str,
        data_agents: list[DataAgent],
        agent_names: list[str],
        query: str,
        merge_task: str,
    ) -> AlignedResultList:
        for handler_id, handler in handlers.items():
            if execution_id == handler_id or (prefix_match and execution_id.startswith(handler_id + "/")):
                return handler(
                    execution_id,
                    data_agents=data_agents,
                    agent_names=agent_names,
                    query=query,
                    merge_task=merge_task,
                )
        raise ValueError(f"No align_schema handler for execution_id={execution_id!r}")

    return _dispatch


def make_data_agent_dispatch(
    handlers: dict[str, Callable[..., PartialQueryAnswer]],
    *,
    prefix_match: bool = False,
) -> Callable[..., PartialQueryAnswer]:
    """Dispatch for DATA_AGENT hook events."""

    def _dispatch(
        execution_id: str,
        data_agent: DataAgent,
        query: str,
        additional_information: list[AdditionalInformation],
        schema_metadata: AlignedSchemaMetadata | None,
    ) -> PartialQueryAnswer:
        for handler_id, handler in handlers.items():
            if execution_id == handler_id or (prefix_match and execution_id.startswith(handler_id + "/")):
                return handler(
                    execution_id,
                    data_agent=data_agent,
                    query=query,
                    additional_information=additional_information,
                    schema_metadata=schema_metadata,
                )
        raise ValueError(f"No data_agent handler for execution_id={execution_id!r}")

    return _dispatch

"""
Hook system. Every event can have pre, overwrite, and post handlers.
- **pre**: observers that run before the operation (all run, receive the call args)
- **overwrite**: a single replacement implementation (last registration wins)
- **post**: observers that run after the operation (all run, receive call args + result)
"""

import functools
import inspect
from collections import defaultdict
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from enum import StrEnum
from typing import Any
from uuid import uuid4

_pre_hooks: defaultdict[str, list[Callable[..., Any]]] = defaultdict(list)
_overwrite_hooks: dict[str, Callable[..., Any]] = {}
_post_hooks: defaultdict[str, list[Callable[..., Any]]] = defaultdict(list)
_execution_id: ContextVar[str] = ContextVar("execution_id", default="")


@contextmanager
def execution_context(execution_id: str):
    """Set a known execution id for the duration of the block (used by scripts)."""
    token = _execution_id.set(execution_id)
    try:
        yield execution_id
    finally:
        _execution_id.reset(token)


def get_execution_id() -> str:
    """Return the current execution id."""
    return _execution_id.get()


class HookEvent(StrEnum):
    """Names of the pipeline events that pre/overwrite/post handlers can register on."""

    # ── Setup ────────────────────────────────────────────────────────────────
    SETUP_LOAD_INPUT = "setup_load_input"
    SETUP_LOAD_CONFIG = "setup_load_config"

    # ── Stage 1: Discovery ───────────────────────────────────────────────────
    DISCOVERY_STAGE = "discovery_stage"
    DISCOVERY_FILTER_RELEVANT_AGENTS = "discovery_filter_relevant_agents"
    DISCOVERY_LOAD_REGISTRY_DATA = "discovery_load_registry_data"
    DISCOVERY_DATA_AGENT_SCHEMA_DESCRIPTION = "discovery_data_agent_schema_description"
    DISCOVERY_EMBEDDING_PRE_FILTER = "discovery_embedding_pre_filter"
    DISCOVERY_COORDINATOR_RELEVANCE = "discovery_coordinator_relevance"
    DISCOVERY_DATA_AGENT_RELEVANCE = "discovery_data_agent_relevance"

    # ── Stage 2: Planning ────────────────────────────────────────────────────
    PLANNING_STAGE = "planning_stage"
    PLANNING_DRAFT_QUERY_PLAN = "planning_draft_query_plan"

    # ── Stage 3: Execution ───────────────────────────────────────────────────
    EXECUTION_STAGE = "execution_stage"
    EXECUTION_EXECUTE_QUERY_PLAN = "execution_execute_query_plan"
    EXECUTION_EXECUTE_QUERY_PLAN_EXECUTE_STEP = "execution_execute_query_plan_execute_step"
    EXECUTION_EXECUTE_QUERY_PLAN_EXECUTE_STEP_ANSWER_STEP_DATA_AGENT = (
        "execution_execute_query_plan_execute_step_answer_step_data_agent"
    )
    EXECUTION_QUERY_DATA_AGENT = "execution_query_data_agent"
    EXECUTION_EXECUTE_QUERY_PLAN_EXECUTE_STEP_ANSWER_STEP_COORDINATOR = (
        "execution_execute_query_plan_execute_step_answer_step_coordinator"
    )
    EXECUTION_EXECUTE_QUERY_PLAN_EXECUTE_STEP_ANSWER_STEP_MULTIPLE_AGENTS = (
        "execution_execute_query_plan_execute_step_answer_step_multiple_agents"
    )
    EXECUTION_EXECUTE_QUERY_PLAN_EXECUTE_STEP_ANSWER_STEP_MULTIPLE_AGENTS_ALIGN_SCHEMA = (
        "execution_execute_query_plan_execute_step_answer_step_multiple_agents_align_schema"
    )
    EXECUTION_EXECUTE_QUERY_PLAN_EXECUTE_STEP_ANSWER_STEP_MULTIPLE_AGENTS_COMPLETENESS = (
        "execution_execute_query_plan_execute_step_answer_step_multiple_agents_completeness"
    )
    EXECUTION_EXECUTE_QUERY_PLAN_EXECUTE_STEP_ANSWER_STEP_MULTIPLE_AGENTS_COMPLETENESS_ESTIMATOR = (
        "execution_execute_query_plan_execute_step_answer_step_multiple_agents_completeness_estimator"
    )
    EXECUTION_EXECUTE_QUERY_PLAN_EXECUTE_STEP_ANSWER_STEP_MULTIPLE_AGENTS_COMPLETENESS_LLM = (
        "execution_execute_query_plan_execute_step_answer_step_multiple_agents_completeness_llm"
    )
    EXECUTION_EXECUTE_QUERY_PLAN_EXECUTE_STEP_ANSWER_STEP_MULTIPLE_AGENTS_RESULT_PRODUCER = (
        "execution_execute_query_plan_execute_step_answer_step_multiple_agents_result_producer"
    )
    EXECUTION_REPLAN = "execution_replan"
    EXECUTION_REPLAN_DRAFT_QUERY_PLAN = "execution_replan_draft_query_plan"

    # ── Stage 4: Synthesis ───────────────────────────────────────────────────
    SYNTHESIS_STAGE = "synthesis_stage"

    # ── General Hooks ───────────────────────────────────────────────────
    AGENT_COMPLETED = "agent_completed"
    EMBEDDING_COMPLETED = "embedding_completed"


# ── Registration ─────────────────────────────────────────────────────────────


def pre(event: HookEvent):
    """Register an observer that runs before the operation."""

    def decorator(fn):
        """Register a pre-handler and return the wrapped function."""
        register_pre(event, fn)
        return fn

    return decorator


def overwrite(event: HookEvent):
    """Register a replacement implementation (last registration wins)."""

    def decorator(fn):
        """Register an overwrite handler and return the wrapped function."""
        register_overwrite(event, fn)
        return fn

    return decorator


def post(event: HookEvent):
    """Register an observer that runs after the operation."""

    def decorator(fn):
        """Register a post-handler and return the wrapped function."""
        register_post(event, fn)
        return fn

    return decorator


def register_pre(event: HookEvent, handler: Callable[..., Any]) -> None:
    """Register a pre-handler for an event."""
    _pre_hooks[event].append(handler)


def register_overwrite(event: HookEvent, handler: Callable[..., Any]) -> None:
    """Register the replacement handler for an event."""
    _overwrite_hooks[event] = handler


def register_post(event: HookEvent, handler: Callable[..., Any]) -> None:
    """Register a post-handler for an event."""
    _post_hooks[event].append(handler)


def unregister_pre(event: HookEvent, handler: Callable[..., Any]) -> None:
    """Unregister a pre-handler for an event."""
    _pre_hooks[event].remove(handler)


def unregister_overwrite(event: HookEvent, handler: Callable[..., Any]) -> None:
    """Unregister the replacement handler for an event."""
    _overwrite_hooks.pop(event, None)


def unregister_post(event: HookEvent, handler: Callable[..., Any]) -> None:
    """Unregister a post-handler for an event."""
    _post_hooks[event].remove(handler)


def clear(event: HookEvent | None = None) -> None:
    """Clear all handlers, or only those registered for a given event."""
    if event is None:
        _pre_hooks.clear()
        _overwrite_hooks.clear()
        _post_hooks.clear()
    else:
        _pre_hooks.pop(event, None)
        _overwrite_hooks.pop(event, None)
        _post_hooks.pop(event, None)


async def fire_post(event: HookEvent, **kwargs: Any) -> None:
    """Invoke only the registered post-handlers for `event`."""
    execution_id = get_execution_id()
    for handler in _post_hooks[event]:
        result = _call(handler, execution_id, kwargs)
        if inspect.isawaitable(result):
            await result


# ── Dispatch ─────────────────────────────────────────────────────────────────


@contextmanager
def _stage_context(stage_context: bool = False):
    """Enter a fresh nested execution-id scope for the duration of the block."""
    if not stage_context:
        yield
        return

    parent = _execution_id.get()
    new_id = uuid4().hex[:12]
    id_ = f"{parent}/{new_id}" if parent else new_id
    token = _execution_id.set(id_)
    try:
        yield
    finally:
        _execution_id.reset(token)


def _call(handler: Callable[..., Any], execution_id: str, kwargs: dict[str, Any]) -> Any:
    """
    Call a handler with the kwargs it declares (plus execution_id).

    Returns the handler's result, which may be a coroutine if the handler is async.
    """
    params = inspect.signature(handler).parameters
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    call_kwargs = {"execution_id": execution_id}
    for name, value in kwargs.items():
        if name in params or has_var_kw:
            call_kwargs[name] = value
    return handler(**call_kwargs)


async def _run_async(
    event: HookEvent, default: Callable[..., Any], kwargs: dict[str, Any], stage_context: bool = False
) -> Any:
    """Run pre → overwrite-or-default → post, awaiting any async handlers"""
    with _stage_context(stage_context):
        execution_id = get_execution_id()
        for handler in _pre_hooks[event]:
            result = _call(handler, execution_id, kwargs)
            if inspect.isawaitable(result):
                await result

        result = (
            _call(_overwrite_hooks[event], execution_id, kwargs) if event in _overwrite_hooks else default(**kwargs)
        )
        if inspect.isawaitable(result):
            result = await result

        for handler in _post_hooks[event]:
            post_result = _call(handler, execution_id, {**kwargs, "result": result})
            if inspect.isawaitable(post_result):
                await post_result

        return result


def _run_sync(
    event: HookEvent, default: Callable[..., Any], kwargs: dict[str, Any], stage_context: bool = False
) -> Any:
    """Run pre → overwrite-or-default → post for a sync default"""
    with _stage_context(stage_context):
        execution_id = get_execution_id()
        for handler in _pre_hooks[event]:
            _call(handler, execution_id, kwargs)

        result = (
            _call(_overwrite_hooks[event], execution_id, kwargs) if event in _overwrite_hooks else default(**kwargs)
        )

        for handler in _post_hooks[event]:
            _call(handler, execution_id, {**kwargs, "result": result})

        return result


def hook(event: HookEvent, *, stage_context: bool = False):
    """
    Wrap a default implementation so calls route through pre/overwrite/post.

    Handlers may be sync or async; async results are awaited. The wrapped
    function may be sync or async — sync defaults are called directly and their
    result returned as-is (callers awaiting it get the value).

    When `stage_context=True`, the wrapper enters a fresh execution-id scope
    before dispatch, so pre/overwrite/post handlers and the default all share
    the same nested execution id.
    """

    def decorator(fn):
        """Wrap a default implementation and route it through event dispatch."""
        sig = inspect.signature(fn)

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs):
                """Call the async dispatch for the wrapped default."""
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                return await _run_async(event, fn, dict(bound.arguments), stage_context)

            return async_wrapper

        else:

            @functools.wraps(fn)
            def sync_wrapper(*args, **kwargs):
                """Call the sync dispatch for the wrapped default."""
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                return _run_sync(event, fn, dict(bound.arguments), stage_context)

            return sync_wrapper

    return decorator

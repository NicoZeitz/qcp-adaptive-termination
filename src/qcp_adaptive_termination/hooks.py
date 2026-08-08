from collections import defaultdict
from enum import StrEnum
from typing import Any, Callable

_handlers: defaultdict[str, list[Callable[..., Any]]] = defaultdict(list)


class HookEvent(StrEnum):
    ITERATION_RECORDS = "iteration_records"


def on(event: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        register(event, fn)
        return fn

    return decorator


def register(event: str, handler: Callable[..., Any]) -> None:
    _handlers[event].append(handler)


def emit(event: str, **kwargs: Any) -> None:
    for handler in _handlers[event]:
        handler(**kwargs)


def clear(event: str | None = None) -> None:
    if event is None:
        _handlers.clear()
    else:
        _handlers.pop(event, None)

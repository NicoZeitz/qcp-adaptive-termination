"""
Cache: storage, lookup, setup, and the `@cached` decorator.
"""

import inspect
import logging
import pickle
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, cast

import orjson
import xxhash
from pydantic import BaseModel

from qcp_adaptive_termination.models import CacheMode


def _key_default(obj: Any) -> Any:
    """orjson serialization hook for cache key hashing, handling pydantic models and enums."""
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if isinstance(obj, Enum):
        return obj.value
    raise TypeError(f"cache key contains non-serializable type: {type(obj)}")


@dataclass(slots=True)
class Cache:
    folder: Path
    mode: CacheMode = field(default=CacheMode.READ_AND_WRITE)

    def get(self, key: Any) -> tuple[Any, bool]:
        """Read a value from the cache by key.

        Args:
            key: the cache key to find a cached value for

        Returns:
            (value, found): the stored value and whether it was present.
            When `found` is False, `value` is None
        """
        if self.mode in [CacheMode.WRITE_ONLY, CacheMode.DISABLED]:
            return None, False

        cache_file = self.cache_file_for_key(key)
        try:
            with cache_file.open("rb") as f:
                value = pickle.load(f)
            return value, True
        except Exception:
            self._throw_on_miss(key)
            return None, False

    def store(self, key: Any, value: Any):
        """Write a value to the cache under the given key."""
        if self.mode in [
            CacheMode.READ_ONLY,
            CacheMode.READ_ONLY_THROW_ON_MISS,
            CacheMode.DISABLED,
        ]:
            return None

        cache_file = self.cache_file_for_key(key)
        temporary_file = cache_file.with_name(f"{cache_file.stem}_{uuid.uuid4().hex}.tmp")
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with temporary_file.open("wb") as f:
                pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
            temporary_file.replace(cache_file)
        except Exception as exc:
            logging.error("Cache: ERROR storing cache file %s: %s", cache_file, exc)
            if temporary_file.exists():
                temporary_file.unlink()

    def cache_file_for_key(self, key: Any) -> Path:
        """Return the cache file path for a key."""
        key_bytes = orjson.dumps(key, default=_key_default, option=orjson.OPT_SORT_KEYS)
        hash = xxhash.xxh3_128_hexdigest(key_bytes)
        return self.folder / f"{hash}.pkl"

    def _throw_on_miss(self, key: Any):
        if self.mode in [
            CacheMode.READ_ONLY_THROW_ON_MISS,
            CacheMode.READ_AND_WRITE_THROW_ON_MISS,
        ]:
            raise KeyError(f"Cache miss for key: {key!r}")


def load_cache_file(cache_file: Path) -> tuple[Any, dict[str, Any] | None]:
    """Read a cached value and its optional extra metadata from a cache file.

    Args:
        cache_file: path to a pickled cache file

    Returns:
        (value, extra): the stored value and optional extra dict, or None for extra
    """
    try:
        with cache_file.open("rb") as f:
            payload = pickle.load(f)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Cache file not found: {cache_file}") from exc
    return payload["value"], payload.get("extra")


def cached[**Param, RetType](
    mode: CacheMode | None = None,
    ignore_args: set[str] | list[str] | None = None,
) -> Callable[[Callable[Param, Awaitable[RetType]]], Callable[Param, Awaitable[RetType]]]:
    """Wrap an async function so its result is cached across calls.

    Args:
        mode: cache mode override; uses the active cache's mode when None
        ignore_args: argument names to exclude from the cache key

    Returns:
        A decorator wrapping the async function with cache hit/miss handling
    """

    def decorator(
        func: Callable[Param, Awaitable[RetType]],
    ) -> Callable[Param, Awaitable[RetType]]:
        from qcp_adaptive_termination.observability.context import get_cache

        sig = inspect.signature(func)

        @wraps(func)
        async def wrapper(*args: Param.args, **kwargs: Param.kwargs) -> RetType:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            arguments = {**bound.arguments}
            if ignore_args:
                for arg in ignore_args:
                    arguments.pop(arg, None)

            cache_instance = get_cache()
            if cache_instance is None:
                return await func(*args, **kwargs)

            effective_cache = Cache(folder=cache_instance.folder, mode=mode or cache_instance.mode)
            result, found = effective_cache.get(key=arguments)

            if found:
                return cast(RetType, result)

            result = await func(*args, **kwargs)
            effective_cache.store(key=arguments, value=result)
            return result

        return wrapper

    return decorator


async def get_or_compute_batch[T, R](
    items: list[T],
    batch_fn: Callable[[list[T]], Awaitable[list[R]]],
) -> list[R]:
    """Return cached results for items, computing and storing only the misses.

    Args:
        items: the items to resolve
        batch_fn: computes results for a set of missing items

    Returns:
        A list of results aligned with `items`, in the same order
    """
    from qcp_adaptive_termination.observability.context import get_cache

    cache_instance = get_cache()
    if cache_instance is None:
        return await batch_fn(items)

    results: list[R | None] = []
    miss_indices: list[int] = []
    miss_items: list[T] = []

    for i, item in enumerate(items):
        cached_value, found = cache_instance.get(key=item)
        if found:
            results.append(cached_value)
        else:
            results.append(None)
            miss_indices.append(i)
            miss_items.append(item)

    if miss_items:
        miss_results = await batch_fn(miss_items)
        for idx, item, result in zip(miss_indices, miss_items, miss_results, strict=True):
            cache_instance.store(key=item, value=result)
            results[idx] = result

    return cast(list[R], results)

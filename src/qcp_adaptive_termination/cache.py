from contextlib import contextmanager
from enum import Enum, StrEnum
from functools import wraps
import hashlib
import inspect
import json
import logging
from pathlib import Path
import uuid
from typing import Any, Awaitable, Callable, cast, ParamSpec, Sequence, TypeVar

from pydantic import BaseModel, JsonValue

from qcp_adaptive_termination.tracing.span import indented_logging, own_custom_span


class CacheMode(StrEnum):
    READ_AND_WRITE = "READ_AND_WRITE"
    READ_AND_WRITE_THROW_ON_MISS = "READ_AND_WRITE_THROW_ON_MISS"
    READ_ONLY = "READ_ONLY"
    READ_ONLY_THROW_ON_MISS = "READ_ONLY_THROW_ON_MISS"
    WRITE_ONLY = "WRITE_ONLY"
    DISABLED = "DISABLED"


class Cache:
    _instance: "Cache | None" = None
    _cache_folder: Path | None = None

    mode: CacheMode = CacheMode.READ_AND_WRITE

    @classmethod
    def instance(cls: type["Cache"]) -> "Cache":
        if not cls._instance:
            cls._instance = cls.__new__(cls)
        return cls._instance

    def set_mode(self, mode: CacheMode):
        self.mode = mode

    def set_cache_folder(self, folder: Path):
        self._cache_folder = folder
        self._cache_folder.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def cache_context(self, mode: CacheMode | None = None):
        original_mode = self.mode
        if mode:
            self.set_mode(mode)
        try:
            yield self
        finally:
            self.set_mode(original_mode)

    def get(
        self, key: JsonValue, name_prefix: str = ""
    ) -> tuple[JsonValue | None, bool]:
        if self.mode in [CacheMode.WRITE_ONLY, CacheMode.DISABLED]:
            return None, False

        cache_file = self.cache_file_for_key(key, name_prefix)
        try:
            with cache_file.open("r", encoding="utf-8") as f:
                payload = json.load(f)
                indented_logging(f"Cache: HIT ({cache_file})")
                return payload.get("value"), True
        except (FileNotFoundError, Exception):
            self._throw_on_miss(key)
            return None, False

    def store(self, key: JsonValue, value: JsonValue, name_prefix: str = ""):
        if self.mode in [
            CacheMode.READ_ONLY,
            CacheMode.READ_ONLY_THROW_ON_MISS,
            CacheMode.DISABLED,
        ]:
            return None

        cache_file = self.cache_file_for_key(key, name_prefix)

        payload = {"key": key, "value": value}

        temporary_file = cache_file.with_name(
            f"{cache_file.stem}_{uuid.uuid4().hex}.tmp"
        )
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with temporary_file.open("w", encoding="utf-8") as f:
                # keep indent for readability, but remove extra spaces for smaller file size
                json.dump(
                    payload, f, separators=(",", ":"), ensure_ascii=False, indent=2
                )
            indented_logging(f"Cache: STORE ({cache_file})")
            temporary_file.replace(cache_file)
        except Exception as exc:
            indented_logging(
                f"Cache: ERROR storing cache file {cache_file}: {exc}",
                log_level=logging.ERROR,
            )
            if temporary_file.exists():
                temporary_file.unlink()

    def cache_file_for_key(self, key: JsonValue, name_prefix: str = "") -> Path:
        if self._cache_folder is None:
            raise RuntimeError(
                "Cache folder not set. Call set_cache_folder() before using the cache."
            )
        hash = hashlib.sha256(
            json.dumps(
                key, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8")
        ).hexdigest()
        filename = f"{name_prefix}_{hash}.json" if name_prefix else f"{hash}.json"
        return self._cache_folder / filename

    def _throw_on_miss(self, key: JsonValue):
        if self.mode in [
            CacheMode.READ_ONLY_THROW_ON_MISS,
            CacheMode.READ_AND_WRITE_THROW_ON_MISS,
        ]:
            raise KeyError(f"Cache miss for key: {json.dumps(key, sort_keys=True)}")


def _serialize_value(value: Any) -> JsonValue:
    if isinstance(value, BaseModel):
        serialized_value = value.model_dump(mode="python")
        serialized_value["__type__"] = type(value).__name__
        return serialized_value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        serialized_value = {
            key: _serialize_value(value) for key, value in value.items()
        }
        serialized_value["__type__"] = "dict"
        return serialized_value
    if isinstance(value, list) or isinstance(value, set) or isinstance(value, tuple):
        return [_serialize_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    raise ValueError(f"Unsupported value type: {type(value)}")


def _deserialize_value(
    value: JsonValue, deserializable_classes: Sequence[type[BaseModel]]
) -> Any:
    if isinstance(value, dict):
        if "__type__" in value and value["__type__"] == "dict":
            return {
                key: _deserialize_value(value, deserializable_classes)
                for key, value in value.items()
                if key != "__type__"
            }

        # registered pydantic models
        if "__type__" in value:
            value_type = value["__type__"]
            value_content = {
                key: value for key, value in value.items() if key != "__type__"
            }

            for cls in deserializable_classes:
                if cls.__name__ == value_type:
                    return cls.model_validate(value_content)

        raise Exception(
            f"Cannot deserialize value: {value}, no matching type found in deserializable_classes: {deserializable_classes}"
        )
    if isinstance(value, list):
        return [_deserialize_value(item, deserializable_classes) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    raise ValueError(f"Unsupported value type: {type(value)}")


def load_cache_file(
    cache_file: Path, deserializable_classes: Sequence[type[BaseModel]] | None = None
) -> tuple[Any, dict[str, Any] | None]:
    try:
        with cache_file.open("r", encoding="utf-8") as f:
            payload = json.load(f)
            value = payload.get("value")
            extra = payload.get("extra")

            if deserializable_classes:
                return _deserialize_value(value, deserializable_classes), extra

            return value, extra
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Cache file not found: {cache_file}") from exc


def setup_cache(
    cache_dir: Path,
    mode: CacheMode = CacheMode.READ_AND_WRITE,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    Cache.instance().set_cache_folder(cache_dir)
    Cache.instance().set_mode(mode)
    if mode == CacheMode.DISABLED:
        logging.info("Cache completely disabled")


Param = ParamSpec("Param")
RetType = TypeVar("RetType")


def cached(
    name_prefix: str | Callable[Param, str] = "",
    mode: CacheMode | None = None,
    ignore_args: set[str] | list[str] | None = None,
    deserializable_classes: list[type[BaseModel]] | None = None,
) -> Callable[
    [Callable[Param, Awaitable[RetType]]], Callable[Param, Awaitable[RetType]]
]:
    def decorator(
        func: Callable[Param, Awaitable[RetType]],
    ) -> Callable[Param, Awaitable[RetType]]:
        sig = inspect.signature(func)

        deserializable_values = deserializable_classes or []
        if isinstance(sig.return_annotation, type) and issubclass(
            sig.return_annotation, BaseModel
        ):
            deserializable_values.append(sig.return_annotation)
        for param in sig.parameters.values():
            if (
                isinstance(param.annotation, type)
                and issubclass(param.annotation, BaseModel)
                and param.annotation not in deserializable_values
            ):
                deserializable_values.append(param.annotation)

        @wraps(func)
        async def wrapper(*args: Param.args, **kwargs: Param.kwargs) -> RetType:
            nonlocal name_prefix

            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            arguments = {**bound.arguments}
            if ignore_args:
                for arg in ignore_args:
                    if arg in arguments:
                        del arguments[arg]

            resolved_prefix: str = (
                cast(Callable[..., str], name_prefix)(*args, **kwargs)
                if callable(name_prefix)
                else name_prefix
            )

            with Cache.instance().cache_context(mode) as cache:
                key: JsonValue = _serialize_value(arguments)
                result, found = cache.get(key=key, name_prefix=resolved_prefix)

                if found:
                    name = resolved_prefix or getattr(func, "__name__", "")

                    if "Schema Description" not in name:
                        with own_custom_span(
                            f"Cache Hit: {name}",
                            input={"arguments": key},
                            output={"result": result},
                            print_start_end=False,
                        ):
                            pass

                    return _deserialize_value(result, deserializable_values)

            result = await func(*args, **kwargs)
            cache.store(
                key=key, value=_serialize_value(result), name_prefix=resolved_prefix
            )
            return result

        return wrapper

    return decorator

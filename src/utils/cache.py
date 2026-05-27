from contextlib import contextmanager
from enum import Enum
from functools import wraps
import hashlib
import inspect
import json
import logging
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, ParamSpec, Sequence, TypeVar

from pydantic import BaseModel, JsonValue

from utils.span import own_custom_span

CacheMode = Literal[
    "READ_AND_WRITE",
    "READ_AND_WRITE_THROW_ON_MISS",
    "READ_ONLY",
    "READ_ONLY_THROW_ON_MISS",
    "WRITE_ONLY",
    "DISABLED",
]


class Cache:
    _instance: "Cache | None" = None
    _cache_folder = Path("../.cache")

    mode: CacheMode = "READ_AND_WRITE"

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
        if self.mode in ["WRITE_ONLY", "DISABLED"]:
            return None, False

        cache_file = self.cache_file_for_key(key, name_prefix)
        try:
            with cache_file.open("r", encoding="utf-8") as f:
                payload = json.load(f)
                logging.debug(f"Cache: HIT ({cache_file})")
                return payload.get("value"), True
        except (FileNotFoundError, Exception):
            self._throw_on_miss(key)
            return None, False

    def store(self, key: JsonValue, value: JsonValue, name_prefix: str = ""):
        if self.mode in ["READ_ONLY", "READ_ONLY_THROW_ON_MISS", "DISABLED"]:
            return None

        cache_file = self.cache_file_for_key(key, name_prefix)

        payload = {"key": key, "value": value}

        temporary_file = cache_file.with_suffix(".tmp")
        with temporary_file.open("w", encoding="utf-8") as f:
            # keep indent for readability, but remove extra spaces for smaller file size
            json.dump(payload, f, separators=(",", ":"), ensure_ascii=False, indent=2)
        logging.debug(f"Cache: STORE ({cache_file})")
        temporary_file.replace(cache_file)

    def cache_file_for_key(self, key: JsonValue, name_prefix: str = "") -> Path:
        hash = hashlib.sha256(
            json.dumps(
                key, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8")
        ).hexdigest()
        filename = f"{name_prefix}_{hash}.json" if name_prefix else f"{hash}.json"
        return self._cache_folder / filename

    def _throw_on_miss(self, key: JsonValue):
        if self.mode in ["READ_ONLY_THROW_ON_MISS", "READ_AND_WRITE_THROW_ON_MISS"]:
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

            name_prefix = (
                name_prefix(*args, **kwargs) if callable(name_prefix) else name_prefix
            )

            with Cache.instance().cache_context(mode) as cache:
                key: JsonValue = _serialize_value(arguments)
                result, found = cache.get(key=key, name_prefix=name_prefix)

                if found:
                    with own_custom_span(
                        f"Cache Hit: {name_prefix or func.__name__}",
                        input={"arguments": key},
                        output={"result": result},
                        print_start_end=False,
                    ):
                        pass

                    return _deserialize_value(result, deserializable_values)

                result = await func(*args, **kwargs)
                cache.store(
                    key=key, value=_serialize_value(result), name_prefix=name_prefix
                )
                return result

        return wrapper

    return decorator

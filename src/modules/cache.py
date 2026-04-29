import os
import hashlib
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, TypeVar, cast

from agents import Agent, agent_span
import cloudpickle
from pydantic import BaseModel

from modules.model import CHAR_LIMIT_SPAN, AgentContext, own_custom_span, preview_text

CACHE_FOLDER = "../.cache"

T = TypeVar("T")

CacheMode = Literal["READ_AND_WRITE", "READ_ONLY", "WRITE_ONLY", "DISABLED"]


class Cache:
    _instance = None
    mode: CacheMode = "READ_AND_WRITE"
    disabled = False
    cache_folder: Path

    def __init__(self):
        raise RuntimeError("Call instance() instead")

    def set_mode(self, mode: CacheMode):
        self.mode = mode

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            os.makedirs(CACHE_FOLDER, exist_ok=True)
            cls._instance.cache_folder = Path(CACHE_FOLDER)

        return cls._instance

    def get(self, keys: list[Any]) -> Any | None:
        if self.mode in ["WRITE_ONLY", "DISABLED"]:
            return None

        cache_file = self._cache_file_for_keys(keys)

        if not cache_file.exists():
            return None

        with cache_file.open("rb") as handle:
            payload = handle.read()

        try:
            return cloudpickle.loads(payload)
        except Exception:
            return None

    def store(self, keys: list[Any], value: Any) -> None:
        if self.mode in ["READ_ONLY", "DISABLED"]:
            return None

        cache_file = self._cache_file_for_keys(keys)
        payload = cloudpickle.dumps(value)
        temporary_file = cache_file.with_suffix(".tmp")

        with temporary_file.open("wb") as handle:
            handle.write(payload)

        os.replace(temporary_file, cache_file)

    async def get_or_compute(
        self,
        agent: Agent[AgentContext],
        user_prompt: str,
        output_model: type | None,
        compute: Callable[[], Awaitable[T]],
    ) -> T:
        cache_keys = [
            getattr(agent, "name", None),
            getattr(agent, "model", None),
            getattr(agent, "model_settings", None),
            getattr(agent, "instructions", None),
            user_prompt,
        ]
        if output_model is not None:
            cache_keys.append(output_model)

        cached_value = self.get(cache_keys)
        if cached_value is not None:
            print(f"        Cache hit for agent: {agent.name}")
            # add a span for opentelemetry here
            with agent_span(
                agent.name,
                tools=[tool.name for tool in agent.tools or []],
                output_type=type(
                    output_model if output_model is not None else ""
                ).__name__,
            ):
                with own_custom_span(
                    "Cache Hit",
                    data={
                        "agent": agent.name,
                        "instructions": preview_text(
                            cast(str, agent.instructions), CHAR_LIMIT_SPAN
                        ),
                        "user_prompt": preview_text(
                            cast(str, user_prompt), CHAR_LIMIT_SPAN
                        ),
                        "tools": [tool.name for tool in agent.tools or []],
                        "output_model": str(output_model)
                        if not hasattr(output_model, "model_json_schema")
                        else cast(type[BaseModel], output_model).model_json_schema(),
                        "value": str(cached_value)
                        if not hasattr(cached_value, "model_dump")
                        else cached_value.model_dump(),
                    },
                ):
                    pass

            return cached_value

        value = await compute()
        if self.mode in ["READ_AND_WRITE", "WRITE_ONLY"]:
            print(f"        Caching result for agent: {agent.name}")
            self.store(cache_keys, value)
        return value

    def _cache_file_for_keys(self, keys: list[Any]) -> Path:
        normalized_keys = self._normalize(keys)
        encoded_keys = repr(normalized_keys).encode("utf-8")
        cache_key = hashlib.sha256(encoded_keys).hexdigest()
        prefix = self._extract_prefix(keys)
        return self.cache_folder / f"{prefix}_{cache_key}.pkl"

    def _extract_prefix(self, keys: list[Any]) -> str:
        if not keys:
            return "cache"

        candidate = keys[0]
        if candidate is None:
            return "cache"

        prefix = str(candidate).strip().lower()
        if not prefix:
            return "cache"

        safe_chars = []
        for char in prefix:
            if char.isalnum() or char in {"-", "_"}:
                safe_chars.append(char)
            else:
                safe_chars.append("-")

        normalized = "".join(safe_chars).strip("-_")
        if not normalized:
            return "cache"

        return normalized[:60]

    def _normalize(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): self._normalize(item) for key, item in value.items()}

        if isinstance(value, list):
            return [self._normalize(item) for item in value]

        if isinstance(value, tuple):
            return ("tuple", tuple(self._normalize(item) for item in value))

        if isinstance(value, set):
            return (
                "set",
                tuple(self._normalize(item) for item in sorted(value, key=repr)),
            )

        if isinstance(value, Path):
            return ("path", str(value))

        if isinstance(value, type):
            return ("type", f"{value.__module__}.{value.__qualname__}")

        if callable(value):
            return ("callable", getattr(value, "__qualname__", repr(value)))

        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        return repr(value)

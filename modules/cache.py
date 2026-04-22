import os
import hashlib
from pathlib import Path
from typing import Any, Awaitable, Callable, TypeVar

from agents import Agent, agent_span, custom_span
from agents.tracing import response_span
import cloudpickle
from openai import BaseModel

from modules.model import AgentContext

CACHE_FOLDER = "./.cache"

T = TypeVar("T")


class Cache:
    _instance = None
    cache_folder: Path

    def __init__(self):
        raise RuntimeError('Call instance() instead')

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            os.makedirs(CACHE_FOLDER, exist_ok=True)
            cls._instance.cache_folder = Path(CACHE_FOLDER)

        return cls._instance

    def get(self, keys: list[Any]) -> Any | None:
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
        cache_file = self._cache_file_for_keys(keys)
        payload = cloudpickle.dumps(value)
        temporary_file = cache_file.with_suffix(".tmp")

        with temporary_file.open("wb") as handle:
            handle.write(payload)

        os.replace(temporary_file, cache_file)

    async def get_or_compute(self, agent: Agent[AgentContext], user_prompt: str, output_model: BaseModel | None, compute: Callable[[], Awaitable[T]]) -> T:
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
                output_type=type(output_model if output_model is not None else "").__name__):
                with custom_span("Cache Hit", data={
                    "agent": agent.name,
                    "instructions": agent.instructions,
                    "user_prompt": user_prompt,
                    "tools": [tool.name for tool in agent.tools or []],
                    "output_model": str(output_model),
                    "value": str(cached_value) if not hasattr(cached_value, "model_dump") else cached_value.model_dump()
                }):
                    pass

            return cached_value

        value = await compute()
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
            return ("set", tuple(self._normalize(item) for item in sorted(value, key=repr)))

        if isinstance(value, Path):
            return ("path", str(value))

        if isinstance(value, type):
            return ("type", f"{value.__module__}.{value.__qualname__}")

        if callable(value):
            return ("callable", getattr(value, "__qualname__", repr(value)))

        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        return repr(value)
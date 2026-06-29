from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from pydantic import BaseModel, ConfigDict


class ExecutionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    continue_after_final_result: bool = True
    include_confidence_intervals: bool = True

    _instance: ExecutionConfig | None = None

    @classmethod
    def set(cls, **kwargs) -> ExecutionConfig:
        cls._instance = cls(**kwargs)
        return cls._instance

    @classmethod
    def get(cls) -> ExecutionConfig:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    @contextmanager
    def override(cls, **kwargs) -> Generator[ExecutionConfig, None, None]:
        previous = cls._instance
        try:
            current = cls.get()
            cls._instance = current.model_copy(update=kwargs)
            yield cls._instance
        finally:
            cls._instance = previous

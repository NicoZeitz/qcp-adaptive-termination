from contextlib import contextmanager
from contextvars import ContextVar
from typing import ClassVar, Generator

from pydantic import BaseModel, ConfigDict, Field

from qcp_adaptive_termination.model.estimation.estimator_mode import EstimatorMode
from qcp_adaptive_termination.model.selection_strategy.selection_type import (
    SelectionType,
)
from qcp_adaptive_termination.estimator import CountEstimator, Estimator, ValueEstimator


class RoleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    reasoning_effort: str = "none"
    reasoning_summary: str | None = None


class ExecutionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    _var: ClassVar[ContextVar["ExecutionConfig"]]

    data_path_prefix: str = "../data/"

    continue_after_final_result: bool = False

    selection_type: SelectionType = SelectionType.SEQUENTIAL
    align_schema_agent_limit: int = 1000

    estimator_mode: EstimatorMode = EstimatorMode.FULL
    count_estimator: CountEstimator = Estimator.DEFAULT_COUNT_ESTIMATOR
    value_estimator: ValueEstimator = Estimator.DEFAULT_VALUE_ESTIMATOR
    use_bucket: bool = True
    use_monte_carlo: bool = False
    include_confidence_intervals: bool = True

    embedding_model: str = "openai/text-embedding-3-large"
    agent_configs: dict[str, RoleConfig] = Field(default_factory=dict)

    log_dir: str | None = None
    cache_dir: str | None = None
    log_name: str | None = None

    @classmethod
    def get(cls) -> "ExecutionConfig":
        return cls._var.get()

    @classmethod
    def set(cls, **kwargs) -> "ExecutionConfig":
        cfg = cls(**kwargs)
        cls._var.set(cfg)
        return cfg

    @classmethod
    @contextmanager
    def override(cls, **kwargs) -> Generator["ExecutionConfig", None, None]:
        cfg = cls.get().model_copy(update=kwargs)
        token = cls._var.set(cfg)
        try:
            yield cfg
        finally:
            cls._var.reset(token)


ExecutionConfig._var = ContextVar("execution_config", default=ExecutionConfig())

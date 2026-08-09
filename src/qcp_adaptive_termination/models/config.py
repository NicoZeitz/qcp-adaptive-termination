from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from qcp_adaptive_termination.cache import CacheMode
from qcp_adaptive_termination.estimator import CountEstimator, Estimator, ValueEstimator
from qcp_adaptive_termination.models.estimation import EstimatorMode
from qcp_adaptive_termination.models.selection_strategy_type import (
    SelectionStrategyType,
)

_config_var: ContextVar["Config | None"] = ContextVar("config", default=None)


class RoleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] = "none"
    reasoning_summary: Literal["auto", "concise", "detailed"] | None = None


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    data_path_prefix: str = "../data/"

    continue_after_final_result: bool = False

    selection_type: SelectionStrategyType = SelectionStrategyType.SEQUENTIAL
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
    cache_mode: CacheMode = CacheMode.READ_AND_WRITE

    @classmethod
    def get(cls) -> "Config":
        cfg = _config_var.get()
        if cfg is None:
            cfg = Config()
            _config_var.set(cfg)
        return cfg

    @classmethod
    @contextmanager
    def context(cls, cfg: "Config") -> Generator["Config", None, None]:
        token = _config_var.set(cfg)
        try:
            yield cfg
        finally:
            _config_var.reset(token)

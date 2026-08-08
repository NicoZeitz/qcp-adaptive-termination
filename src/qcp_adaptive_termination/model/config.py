from contextlib import contextmanager
from contextvars import ContextVar
from typing import Generator

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


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
    throw_on_cache_miss: bool = False
    disable_cache: bool = False


config_var: ContextVar[Config] = ContextVar("config", default=Config())


@contextmanager
def config_context(cfg: Config) -> Generator[Config, None, None]:
    token = config_var.set(cfg)
    try:
        yield cfg
    finally:
        config_var.reset(token)

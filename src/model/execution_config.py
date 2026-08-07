from contextlib import contextmanager
from typing import ClassVar, Generator

from pydantic import BaseModel, ConfigDict, Field

from model.estimation.estimator_mode import EstimatorMode
from model.selection_strategy.selection_type import SelectionType
from common.unknown_unknown import CountEstimator, Estimator, ValueEstimator


class RoleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    reasoning_effort: str = "none"
    reasoning_summary: str | None = None


class ExecutionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    _instance: ClassVar["ExecutionConfig | None"] = None

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

    @classmethod
    def set(cls, **kwargs) -> "ExecutionConfig":
        cls._instance = cls(**kwargs)
        return cls._instance

    @classmethod
    def get(cls) -> "ExecutionConfig":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    @contextmanager
    def override(cls, **kwargs) -> Generator["ExecutionConfig", None, None]:
        previous = cls._instance
        try:
            current = cls.get()
            cls._instance = current.model_copy(update=kwargs)
            yield cls._instance
        finally:
            cls._instance = previous

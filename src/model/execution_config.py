import os
from contextlib import contextmanager
from typing import Generator

from pydantic import BaseModel, ConfigDict

from model.estimation.estimator_mode import EstimatorMode
from model.selection_strategy.selection_type import SelectionType
from common.unknown_unknown import CountEstimator, Estimator, ValueEstimator


class ExecutionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_path_prefix: str = os.environ.get("DATA_PATH_PREFIX", "../data/")

    continue_after_final_result: bool = False

    selection_type: SelectionType = SelectionType.SEQUENTIAL
    align_schema_agent_limit: int = 1000

    estimator_mode: EstimatorMode = EstimatorMode.FULL
    count_estimator: CountEstimator = Estimator.DEFAULT_COUNT_ESTIMATOR
    value_estimator: ValueEstimator = Estimator.DEFAULT_VALUE_ESTIMATOR
    use_bucket: bool = True
    use_monte_carlo: bool = False
    include_confidence_intervals: bool = True

    _instance: "ExecutionConfig | None" = None

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

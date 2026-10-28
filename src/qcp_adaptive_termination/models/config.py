from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from qcp_adaptive_termination.estimator import CountEstimator, Estimator, ValueEstimator
from qcp_adaptive_termination.models.cache_mode import CacheMode
from qcp_adaptive_termination.models.estimation import EstimatorMode
from qcp_adaptive_termination.models.selection_strategy_type import SelectionStrategyType


class RoleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] = "none"
    reasoning_summary: Literal["auto", "concise", "detailed"] | None = None


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    continue_after_final_result: bool = False

    discovery_filter_mode: Literal["llm_baseline", "iterative_embedding_filter"] = "iterative_embedding_filter"

    selection_type: SelectionStrategyType = SelectionStrategyType.SEQUENTIAL
    align_schema_agent_limit: int = 1000

    estimator_mode: EstimatorMode = EstimatorMode.FULL
    count_estimator: CountEstimator = Estimator.DEFAULT_COUNT_ESTIMATOR
    value_estimator: ValueEstimator = Estimator.DEFAULT_VALUE_ESTIMATOR
    use_bucket: bool = True
    use_monte_carlo: bool = False
    include_confidence_intervals: bool = True

    selection_kwargs: dict[str, Any] = Field(default_factory=dict)

    embedding_model: str = "openai/text-embedding-3-large"
    agent_configs: dict[str, RoleConfig] = Field(default_factory=dict)
    llm_timeout: float = 60.0
    log_llm_requests: bool = False

    cache_mode: CacheMode = CacheMode.READ_AND_WRITE

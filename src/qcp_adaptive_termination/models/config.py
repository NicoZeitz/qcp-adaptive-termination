from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from qcp_adaptive_termination.models.cache_mode import CacheMode
from qcp_adaptive_termination.models.estimation import (
    ConfidenceIntervalType,
    CountEstimator,
    ValueEstimator,
)
from qcp_adaptive_termination.models.selection_strategy_type import SelectionStrategyType
from qcp_adaptive_termination.models.uncertainty import ApportionmentType


class RoleConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    model: str
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] = "none"
    reasoning_summary: Literal["auto", "concise", "detailed"] | None = None


class EstimatorConfig(BaseModel):
    """Configuration for the unknown-unknowns estimator."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    count_estimator: CountEstimator = CountEstimator.CHAO92
    value_estimator: ValueEstimator = ValueEstimator.NAIVE
    use_bucket: bool = True
    use_monte_carlo: bool = False
    max_buckets: int = 100
    confidence_interval_type: ConfidenceIntervalType = ConfidenceIntervalType.CHAO84
    confidence_epsilon: float = 0.05


class Config(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    continue_after_final_result: bool = False

    selection_type: SelectionStrategyType = SelectionStrategyType.RANDOM
    align_schema_agent_limit: int = 1000

    estimator: EstimatorConfig = Field(default_factory=EstimatorConfig)

    target_confidence: float = 0.95
    """Global confidence threshold with which the final result must hold."""

    apportionment: ApportionmentType = ApportionmentType.WEIGHTED
    """How the global threshold is split across multi-agent steps."""

    warmup_agents: int = 2  # TODO: find good k value
    """Never stop deterministically before this many agents have been queried."""

    max_replan_attempts: int = 2
    """Maximum number of replan attempts before giving up on a query plan."""

    llm_code_execution_timeout: int = 60
    """Timeout (seconds) for LLM code execution tool calls."""

    llm_code_execution_preview_char_limit: int = 2000
    """Max characters of code-execution output shown to the LLM (truncated in the middle)."""

    selection_kwargs: dict[str, Any] = Field(default_factory=dict)

    embedding_model: str = "openai/text-embedding-3-large"
    agent_configs: dict[str, RoleConfig] = Field(default_factory=dict)

    cache_mode: CacheMode = CacheMode.READ_AND_WRITE

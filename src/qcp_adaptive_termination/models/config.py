"""Pydantic models for runtime configuration."""

from decimal import Decimal
from typing import Any, ClassVar, Literal

from agents.run_config import DEFAULT_MAX_TURNS
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
    """Model configuration for a single agent role."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    model: str = Field(description="Model identifier used by this agent role")
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] = Field(
        default="none", description="Reasoning effort level for the role"
    )
    reasoning_summary: Literal["auto", "concise", "detailed"] | None = Field(
        default=None, description="Reasoned summary verbosity"
    )


class ModelCost(BaseModel):
    """Per-model pricing, USD per 1M tokens. Each rate is optional; unused rates may be omitted."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    input: Decimal | None = Field(default=None, description="USD per 1M input tokens")
    cached_input: Decimal | None = Field(default=None, description="USD per 1M cached input tokens")
    cache_write: Decimal | None = Field(default=None, description="USD per 1M cache write tokens")
    output: Decimal | None = Field(default=None, description="USD per 1M output tokens")
    embedding: Decimal | None = Field(default=None, description="USD per 1M embedding tokens")


class BucketConfig(BaseModel):
    """Bucket-based estimation settings."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    enabled: bool = Field(default=True, description="Whether to bucket the data for estimation")
    max_buckets: int = Field(default=100, description="Maximum number of buckets")


class MonteCarloConfig(BaseModel):
    """Monte Carlo estimation settings."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    enabled: bool = Field(default=False, description="Whether to use Monte Carlo simulation")


class ConfidenceIntervalConfig(BaseModel):
    """Confidence interval settings."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    type: ConfidenceIntervalType = Field(
        default=ConfidenceIntervalType.CHAO84, description="Method used to compute confidence intervals"
    )
    epsilon: float = Field(default=0.05, description="Confidence error probability")


class EstimatorConfig(BaseModel):
    """Configuration for the unknown-unknowns estimator."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    enabled: bool = Field(default=True, description="Whether the estimator may decide to stop")
    start_agents: int = Field(
        default=6,
        description="Never stop deterministically before this many agents have been queried",
    )
    count_estimator: CountEstimator = Field(
        default=CountEstimator.CHAO92, description="Estimator for the total entity count"
    )
    value_estimator: ValueEstimator = Field(
        default=ValueEstimator.NAIVE, description="Estimator for missing attribute values"
    )
    bucket: BucketConfig = Field(default_factory=BucketConfig, description="Bucket-based estimation settings")
    monte_carlo: MonteCarloConfig = Field(default_factory=MonteCarloConfig, description="Monte Carlo settings")
    confidence_interval: ConfidenceIntervalConfig = Field(
        default_factory=ConfidenceIntervalConfig, description="Confidence interval settings"
    )


class LLMWolrdKnowledgeConfig(BaseModel):
    """World-knowledge LLM completeness stop settings."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    enabled: bool = Field(default=True, description="Whether the world-knowledge LLM may decide to stop")
    long_list_start: int = Field(
        default=1000,
        description="When the deduplicated result list exceeds this many rows, long-list behavior kicks in",
    )
    long_list_behavior: Literal["prevent", "estimator_reasoning"] = Field(
        default="prevent",
        description=("Behavior for long lists: prevent skips the LLM, estimator_reasoning feeds estimator stats"),
    )
    long_list_history_turns: int = Field(
        default=5,
        description="On long lists, only the last this many completeness evaluations are kept in the LLM history",
    )


class CompletenessConfig(BaseModel):
    """Termination and result-completeness settings."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    target_confidence: float = Field(
        default=0.95, description="Global confidence threshold with which the final result must hold"
    )
    target_apportionment: ApportionmentType = Field(
        default=ApportionmentType.WEIGHTED,
        description="How the global threshold is split across multi-agent steps",
    )
    continue_after_final_result: bool = Field(default=False, description="Continue after the final result is produced")
    llm_world_knowledge: LLMWolrdKnowledgeConfig = Field(
        default_factory=LLMWolrdKnowledgeConfig, description="World-knowledge LLM stop settings"
    )
    estimator: EstimatorConfig = Field(
        default_factory=EstimatorConfig, description="Config for the unknown-unknowns estimator"
    )


class ExecutionConfig(BaseModel):
    """Execution engine limits."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    replanning_max_attempts: int = Field(
        default=2, description="Maximum number of replan attempts before giving up on a query plan"
    )
    llm_max_turns: int | None = Field(
        default=DEFAULT_MAX_TURNS * 4,
        description="Maximum LLM turns per agent run; None falls back to the agents-library default",
    )
    llm_code_execution_timeout: int = Field(
        default=60, description="Timeout (seconds) for LLM code execution tool calls"
    )
    llm_code_execution_preview_char_limit: int = Field(
        default=2000,
        description="Max characters of code-execution output shown to the LLM (truncated in the middle)",
    )
    schema_alignment_agent_limit: int = Field(default=1000, description="Max agents to align schemas against")


class SelectionConfig(BaseModel):
    """Agent selection strategy settings."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    strategy: SelectionStrategyType = Field(
        default=SelectionStrategyType.POSTERIOR, description="Strategy used to select agents"
    )
    params: dict[str, Any] = Field(
        default_factory=dict, description="Extra keyword args for the agent selection strategy"
    )


class CacheConfig(BaseModel):
    """Result caching behavior."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    mode: CacheMode = Field(default=CacheMode.READ_AND_WRITE, description="How results are cached across runs")


class ModelsConfig(BaseModel):
    """Model identifiers, per-role agent configs, and pricing."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    embedding: str = Field(
        default="openai/text-embedding-3-large",
        description="The embedding model used for schema alignment",
    )
    agents: dict[str, RoleConfig] = Field(default_factory=dict, description="Per-role agent configuration")
    costs: dict[str, ModelCost] = Field(
        default_factory=dict,
        description="Per-model pricing keyed by model name exactly as configured elsewhere",
    )


class Config(BaseModel):
    """Top-level application configuration."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    completeness: CompletenessConfig = Field(default_factory=CompletenessConfig, description="Completeness settings")
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig, description="Execution engine limits")
    selection: SelectionConfig = Field(default_factory=SelectionConfig, description="Agent selection settings")
    cache: CacheConfig = Field(default_factory=CacheConfig, description="Result caching behavior")
    models: ModelsConfig = Field(default_factory=ModelsConfig, description="Model identifiers and pricing")

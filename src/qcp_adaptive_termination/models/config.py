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


class EstimatorConfig(BaseModel):
    """Configuration for the unknown-unknowns estimator."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    count_estimator: CountEstimator = Field(
        default=CountEstimator.CHAO92, description="Estimator for the total entity count"
    )
    value_estimator: ValueEstimator = Field(
        default=ValueEstimator.NAIVE, description="Estimator for missing attribute values"
    )
    use_bucket: bool = Field(default=True, description="Whether to bucket the data for estimation")
    use_monte_carlo: bool = Field(default=False, description="Whether to use Monte Carlo simulation")
    max_buckets: int = Field(default=100, description="Maximum number of buckets")
    confidence_interval_type: ConfidenceIntervalType = Field(
        default=ConfidenceIntervalType.CHAO84, description="Method used to compute confidence intervals"
    )
    confidence_epsilon: float = Field(default=0.05, description="Confidence error probability")


class Config(BaseModel):
    """Top-level application configuration."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    continue_after_final_result: bool = Field(default=False, description="Continue after the final result is produced")

    selection_type: SelectionStrategyType = Field(
        default=SelectionStrategyType.RANDOM, description="Strategy used to select agents"
    )
    align_schema_agent_limit: int = Field(default=1000, description="Max agents to align schemas against")

    estimator: EstimatorConfig = Field(
        default_factory=EstimatorConfig, description="Config for the unknown-unknowns estimator"
    )

    target_confidence: float = Field(
        default=0.95, description="Global confidence threshold with which the final result must hold"
    )

    apportionment: ApportionmentType = Field(
        default=ApportionmentType.WEIGHTED,
        description="How the global threshold is split across multi-agent steps",
    )

    warmup_agents: int = Field(
        default=2,
        description="Never stop deterministically before this many agents have been queried",
    )  # TODO: find good k value

    stop_llm_max_list_items: int = Field(
        default=1000,
        description="When the deduplicated result list exceeds this many rows, long-list behavior kicks in",
    )

    long_list_behavior: Literal["prevent", "estimator_reasoning"] = Field(
        default="prevent",
        description=("Behavior for long lists: prevent skips the LLM, estimator_reasoning feeds estimator stats"),
    )

    max_replan_attempts: int = Field(
        default=2, description="Maximum number of replan attempts before giving up on a query plan"
    )

    max_turns: int | None = Field(
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

    selection_kwargs: dict[str, Any] = Field(
        default_factory=dict, description="Extra keyword args for the agent selection strategy"
    )

    embedding_model: str = Field(
        default="openai/text-embedding-3-large",
        description="The embedding model used for schema alignment",
    )
    agent_configs: dict[str, RoleConfig] = Field(default_factory=dict, description="Per-role agent configuration")

    costs: dict[str, ModelCost] = Field(
        default_factory=dict,
        description="Per-model pricing keyed by model name exactly as configured elsewhere",
    )

    cache_mode: CacheMode = Field(default=CacheMode.READ_AND_WRITE, description="How results are cached across runs")

    # -- Completeness LLM prompt context --
    completeness_show_estimator_stats: bool = Field(
        default=False,
        description="Feed estimator unknown-unknown stats into the completeness LLM prompt",
    )

    completeness_summarize_history: bool = Field(
        default=True,
        description="Tell the completeness LLM it is a continuation and to decide on the new state",
    )

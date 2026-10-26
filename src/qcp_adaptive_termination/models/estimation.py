"""
Estimation data models
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from qcp_adaptive_termination.models.result_type import ResultType, SerializedResultType


class CountEstimator(StrEnum):
    """Strategy for estimating the total number of unique entities in the population."""

    CHAO92 = "chao92"
    """Coverage-based estimator corrected for skew in the publicity distribution via CV²."""

    CHAO84 = "chao84"
    """Estimator based on the ratio of singletons to doubletons (f1² / 2·f2)."""

    JACKKNIFE1 = "jackknife1"
    """First-order jackknife: adds the singleton count f1 as a bias correction."""

    JACKKNIFE2 = "jackknife2"
    """Second-order jackknife: adds 2·f1 and subtracts f2 for a lower-bias estimate."""

    UNSEEN = "unseen"
    """
    Linear-programming estimator that fits a non-parametric frequency distribution
    and extrapolates the number of unseen species.
    """


class ValueEstimator(StrEnum):
    """Strategy for estimating the attribute values of missing entities."""

    NAIVE = "naive"
    """
    Mean substitution: assumes missing entities have the same average value
    as all observed entities (φ_K / c).
    """

    FREQUENCY = "frequency"
    """
    Singleton-mean substitution: uses the average value of singletons (φ_f1 / f1)
    as a proxy for missing entities, which are most similar to rarely seen ones.
    """

    FREQUENCY_F12 = "frequency_f12"
    """
    Singleton+doubleton-mean substitution: averages over singletons and doubletons
    (φ_f12 / (f1 + f2)) to reduce variance compared to singletons alone.
    """

    GOOD_TURING = "good_turing"
    """
    Good-Turing frequency smoothing: derives Δ directly from φ_f1 and the ratio
    f1 / (n - f1) without a separate count estimator.
    """


class ConfidenceIntervalType(StrEnum):
    """Method used to compute confidence intervals for the count estimate."""

    GOOD_TURING = "good_turing"
    """Bounds the missing mass M0 with an additive error term and converts to N via N = c / (1 - M0)."""

    CHAO84 = "chao84"
    """
    Asymmetric log-normal confidence interval for N - c; tighter than the Good-Turing bound.
    Falls back to Good-Turing when there are no doubletons.
    """


class AggregationType(StrEnum):
    """Supported value aggregation functions."""

    COUNT = "count"
    MIN = "min"
    MAX = "max"
    SUM = "sum"
    AVG = "avg"
    Q25 = "q25"
    MEDIAN = "median"
    Q75 = "q75"


class AlignedSchemaMetadata(BaseModel):
    """Describes the schema of an aligned result list."""

    model_config = ConfigDict(extra="forbid")

    aggregation_types: list[AggregationType] = Field(
        ..., min_length=1, description="Aggregation functions applied per column"
    )
    keys: list[str] = Field(..., min_length=1, description="Columns identifying unique entities")
    value_column: str | None = Field(default=None, description="The numeric column being aggregated")
    columns: list[str] = Field(
        default_factory=list, min_length=1, description="All columns present in the aligned list"
    )
    source_agent_column: str = Field(description="Column holding the source agent id")


class AlignedResultList(BaseModel):
    """A result list aligned across agents plus its schema metadata."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    data: SerializedResultType = Field(default=ResultType(), description="The aligned data rows")
    schema_metadata: AlignedSchemaMetadata = Field(description="Schema metadata for the aligned list")

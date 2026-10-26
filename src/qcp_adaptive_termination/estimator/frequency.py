"""Observed frequency statistics for the unknown-unknowns estimator."""

import math
from dataclasses import dataclass

import numpy as np

from qcp_adaptive_termination.estimator.data_item import DataItem


@dataclass(slots=True)
class EntityRecord:
    """Per-entity observation summary."""

    abundance: int
    """Number of times this entity was observed across all sources."""

    first_value: float
    """First-observed attribute value for this entity."""


class FrequencyStats:
    """Observed (known) statistics over the integrated sample."""

    def __init__(self, source_samples: list[list[DataItem]]) -> None:
        """Compute observed statistics by integrating samples from all sources."""
        self._source_samples: list[list[DataItem]] = []  # samples grouped by source
        self._all_samples: list[DataItem] = []  # all samples across all sources
        self._entities: dict[str, EntityRecord] = {}  # entity key → observation record

        self._observed_unique_entities = 0  # c
        self._observed_total_entities = 0  # n
        self._frequencies: list[int] = []  # f_j
        self._observed_value_sum = 0.0  # φ_K
        self._observed_singleton_value_sum = 0.0  # φ_f1
        self._observed_singleton_doubleton_value_sum = 0.0  # φ_f12
        self._observed_value_min = float("inf")
        self._observed_value_max = float("-inf")

        for source in source_samples:
            self.add_source(source)

        if self._observed_total_entities == 0:
            raise ValueError("Estimator requires at least one sample")

    # ── Incremental update ───────────────────────────────────────────────────

    def add_source(self, items: list[DataItem]) -> None:
        """Add one source's observations, updating all statistics incrementally"""
        if not items:
            return

        self._source_samples.append(items)

        for item in items:
            self._all_samples.append(item)

            self._observed_total_entities += 1
            self._observed_value_min = min(self._observed_value_min, item.value)
            self._observed_value_max = max(self._observed_value_max, item.value)

            old_count = self._entities[item.name].abundance if item.name in self._entities else 0

            if old_count == 0:
                self._entities[item.name] = EntityRecord(abundance=1, first_value=float(item.value))
                self._observed_unique_entities += 1

                if not self._frequencies:
                    self._frequencies.append(0)
                self._frequencies[0] += 1

                self._observed_value_sum += item.value
                self._observed_singleton_value_sum += item.value
                self._observed_singleton_doubleton_value_sum += item.value
                continue

            record = self._entities[item.name]
            self._entities[item.name].abundance += 1

            if len(self._frequencies) <= old_count:
                self._frequencies.append(0)
            self._frequencies[old_count - 1] -= 1
            self._frequencies[old_count] += 1

            if old_count == 1:
                # singleton → doubleton
                self._observed_singleton_value_sum -= record.first_value
            elif old_count == 2:
                # doubleton → tripleton
                self._observed_singleton_doubleton_value_sum -= record.first_value

    # ── Accessors ────────────────────────────────────────────────────────────

    @property
    def observed_unique_entities(self) -> int:
        """Number of unique observed entities."""
        return self._observed_unique_entities

    @property
    def observed_total_entities(self) -> int:
        """Total observations including duplicates."""
        return self._observed_total_entities

    @property
    def frequencies(self) -> list[int]:
        """Frequency vector: freq[i] = number of entities seen exactly i+1 times."""
        return self._frequencies

    @property
    def singleton_count(self) -> int:
        """Number of entities observed exactly once (singletons)."""
        return self._frequencies[0] if self._frequencies else 0

    @property
    def doubleton_count(self) -> int:
        """Number of entities observed exactly twice (doubletons)."""
        return self._frequencies[1] if len(self._frequencies) > 1 else 0

    @property
    def observed_value_sum(self) -> float:
        """SUM of first-observed attribute values across unique entities."""
        return self._observed_value_sum

    @property
    def observed_singleton_value_sum(self) -> float:
        """SUM of attribute values of entities seen exactly once."""
        return self._observed_singleton_value_sum

    @property
    def observed_singleton_doubleton_value_sum(self) -> float:
        """SUM of attribute values of singletons and doubletons combined."""
        return self._observed_singleton_doubleton_value_sum

    @property
    def observed_min(self) -> float:
        """Minimum observed attribute value across all samples."""
        return self._observed_value_min

    @property
    def observed_max(self) -> float:
        """Maximum observed attribute value across all samples."""
        return self._observed_value_max

    @property
    def entities(self) -> dict[str, EntityRecord]:
        """name → observation record; one entry per unique entity."""
        return self._entities

    @property
    def all_samples(self) -> list[DataItem]:
        """All observations flattened across sources (including duplicates)."""
        return self._all_samples

    @property
    def source_samples(self) -> list[list[DataItem]]:
        """Observations grouped by source (one inner list per source)."""
        return self._source_samples

    # ── Observed aggregates ──────────────────────────────────────────────────

    def sample_mean(self) -> float:
        """Mean of all raw observations (counting duplicates), not the per-entity mean."""
        return sum(item.value for item in self._all_samples) / self._observed_total_entities

    def sample_std(self) -> float:
        """Standard deviation of all raw observations (counting duplicates)."""
        mean = self.sample_mean()
        sq_mean = sum(item.value**2 for item in self._all_samples) / self._observed_total_entities
        return math.sqrt(max(0.0, sq_mean - mean * mean))

    def γ_sq(self) -> float:
        """Squared coefficient of variation of the publicity distribution."""
        if self._observed_total_entities == 1:
            return 0.0

        C_hat = self.sample_coverage()
        if C_hat == 0:
            return math.inf

        moment = sum((idx + 1) * idx * f for idx, f in enumerate(self._frequencies))
        γ_sq = (self._observed_unique_entities / C_hat) * (
            moment / (self._observed_total_entities * (self._observed_total_entities - 1))
        ) - 1.0
        return max(γ_sq, 0.0)

    def sample_coverage(self) -> float:
        """Good-Turing sample coverage C = 1 - f1/n."""
        return 1.0 - self.singleton_count / self._observed_total_entities

    def sample_sum(self) -> float:
        """Sum of first-observed values across all unique observed entities."""
        return self._observed_value_sum

    def sample_count(self) -> int:
        """Number of unique observed entities."""
        return self._observed_unique_entities

    def sample_avg(self) -> float:
        """Average value across observed entities."""
        return self._observed_value_sum / self._observed_unique_entities

    def sample_min(self) -> float:
        """Minimum observed value."""
        return self._observed_value_min

    def sample_max(self) -> float:
        """Maximum observed value."""
        return self._observed_value_max

    def sample_quantile(self, q: float) -> float:
        """Quantile of observed unique entity values."""
        if not 0.0 <= q <= 1.0:
            raise ValueError(f"q must be in [0, 1], got {q}")
        values = np.array([v.first_value for v in self._entities.values()], dtype=np.float64)
        return float(np.quantile(values, q))

    def sample_median(self) -> float:
        """Median of observed unique entity values. Delegates to sample_quantile(0.5)."""
        return self.sample_quantile(0.5)

    def sample_cdf(self, x: float) -> float:
        """Fraction of observed unique entities with first-observed value at most x."""
        values = np.array([v.first_value for v in self._entities.values()], dtype=np.float64)
        return float(np.mean(values <= x))

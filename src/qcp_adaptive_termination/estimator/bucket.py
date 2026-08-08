from collections import defaultdict
from dataclasses import dataclass, field

from qcp_adaptive_termination.estimator.data_item import DataItem


@dataclass(slots=True)
class HistogramBar:
    """Per-entity observation summary within a bucket.

    Tracks how many times a single entity was observed and the value range
    across those observations. Used to build the frequency vector for a bucket.
    """

    lower_bound: float = float("inf")
    """Minimum observed value for this entity within the bucket."""

    upper_bound: float = float("-inf")
    """Maximum observed value for this entity within the bucket."""

    count: int = 0
    """Number of times this entity was observed within the bucket. >= 1 after first add."""

    def add_sample(self, value: float) -> None:
        """Record one observation of this entity at the given value."""
        self.count += 1
        self.lower_bound = min(self.lower_bound, value)
        self.upper_bound = max(self.upper_bound, value)


@dataclass(slots=True)
class Bucket:
    """A contiguous value-range partition of the sample used by the bucket estimator.

    Holds all DataItems whose attribute value falls within [lower_bound, upper_bound]
    and maintains the frequency vector and histogram needed to run an Estimator on
    this slice of the data.

    The frequency vector and derived counts are computed lazily: they are rebuilt
    only when first accessed after one or more add_sample calls.
    """

    lower_bound: float = float("-inf")
    """Inclusive lower bound of the value range covered by this bucket."""

    upper_bound: float = float("inf")
    """Inclusive upper bound of the value range covered by this bucket."""

    samples: list[DataItem] = field(default_factory=list)
    """All DataItems assigned to this bucket. len == number_of_samples."""

    histogram: dict[str, HistogramBar] = field(
        default_factory=lambda: defaultdict(HistogramBar)
    )
    """Per-entity observation summary; one HistogramBar per unique entity name."""

    f: list[int] = field(default_factory=list)
    """Frequency vector: f[i] = number of entities seen exactly i+1 times within this bucket.
    Rebuilt lazily on first access after add_sample calls."""

    number_of_samples: int = 0
    """Total number of observations in this bucket (counting duplicates). >= 0."""

    number_of_unique_samples: int = 0
    """Number of distinct entities observed in this bucket. >= 0, <= number_of_samples."""

    split_locked: bool = False
    """True when _dynamic_buckets has determined that splitting this bucket further
    does not reduce the estimated delta. No further split attempts will be made."""

    idx: int = 0
    """Optional index used for ordering or identification by the caller."""

    _dirty: bool = field(default=True, repr=False, init=False)
    """True when f and derived counts are stale and must be rebuilt before use."""

    def to_sources(self) -> list[list[DataItem]]:
        """Return samples grouped by source_id, one list per source.

        Used to construct an Estimator for this bucket slice, preserving the
        per-source structure required to detect cross-source duplicates.
        """
        per_source: dict[str | None, list[DataItem]] = {}
        for item in self.samples:
            per_source.setdefault(item.source_id, []).append(item)
        return list(per_source.values())

    def add_sample(self, sample: DataItem) -> None:
        """Add one observation to this bucket, marking derived stats as stale."""
        self.samples.append(sample)
        self.histogram[sample.name].add_sample(sample.value)
        self._dirty = True

    def rebuild_frequency_vector(self) -> None:
        """Recompute f, number_of_samples, and number_of_unique_samples from the histogram."""
        n = len(self.samples)
        f = [0] * n
        for bar in self.histogram.values():
            f[bar.count - 1] += 1
        self.f = f
        self.number_of_samples = n
        self.number_of_unique_samples = len(self.histogram)
        self._dirty = False

    def _ensure_fresh(self) -> None:
        if self._dirty:
            self.rebuild_frequency_vector()

    @property
    def unique_count(self) -> int:
        """Number of distinct entities observed in this bucket."""
        self._ensure_fresh()
        return self.number_of_unique_samples

    @property
    def f1_count(self) -> int:
        """Number of entities observed exactly once (singletons) in this bucket."""
        self._ensure_fresh()
        return self.f[0] if self.f else 0

    def sorted_samples(self) -> list[DataItem]:
        """Return all samples sorted by value ascending."""
        return sorted(self.samples, key=lambda x: x.value)

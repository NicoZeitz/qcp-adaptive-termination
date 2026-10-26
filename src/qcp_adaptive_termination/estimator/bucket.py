"""Value-bucket data structures for the bucket estimator."""

from collections import defaultdict
from dataclasses import dataclass, field

from qcp_adaptive_termination.estimator.data_item import DataItem


@dataclass(slots=True)
class HistogramBar:
    """Per-entity observation summary within a bucket."""

    lower_bound: float = float("inf")
    """Minimum observed value for this entity within the bucket."""

    upper_bound: float = float("-inf")
    """Maximum observed value for this entity within the bucket."""

    count: int = 0
    """Number of times this entity was observed within the bucket. >= 1 after first add."""

    def add_sample(self, value: float):
        """Record one observation of this entity at the given value."""
        self.count += 1
        self.lower_bound = min(self.lower_bound, value)
        self.upper_bound = max(self.upper_bound, value)


@dataclass(slots=True)
class Bucket:
    """A contiguous value-range partition of the sample for the bucket estimator."""

    lower_bound: float = float("-inf")
    """Inclusive lower bound of the value range covered by this bucket."""

    upper_bound: float = float("inf")
    """Inclusive upper bound of the value range covered by this bucket."""

    samples: list[DataItem] = field(default_factory=list)
    """All DataItems assigned to this bucket. len == number_of_samples."""

    histogram: dict[str, HistogramBar] = field(default_factory=lambda: defaultdict(HistogramBar))
    """Per-entity observation summary; one HistogramBar per unique entity name."""

    f: list[int] = field(default_factory=list)
    """Frequency vector: f[i] = number of entities seen exactly i+1 times in this bucket."""

    number_of_samples: int = 0
    """Total number of observations in this bucket (counting duplicates). >= 0."""

    number_of_unique_samples: int = 0
    """Number of distinct entities observed in this bucket. >= 0, <= number_of_samples."""

    split_locked: bool = False
    """True once splitting this bucket further no longer reduces the estimate."""

    idx: int = 0
    """Optional index used for ordering or identification by the caller."""

    _dirty: bool = field(default=True, repr=False, init=False)
    """True when f and derived counts are stale and must be rebuilt before use."""

    entity_positions: dict[str, list[int]] | None = field(default=None, repr=False, init=False)
    """Cached per-entity sorted positions for sweep-based split search."""

    samples_at_last_resplit: int = field(default=0, repr=False, init=False)
    """Number of samples at the last sort+sweep pass."""

    split_backoff: int = field(default=1, repr=False, init=False)
    """Exponential backoff multiplier for the resplit threshold."""

    def to_sources(self) -> list[list[DataItem]]:
        """Return samples grouped by source_id, one list per source."""
        per_source: dict[str | None, list[DataItem]] = {}
        for item in self.samples:
            per_source.setdefault(item.source_id, []).append(item)
        return list(per_source.values())

    def add_sample(self, sample: DataItem):
        """Add one observation to this bucket, marking derived stats as stale."""
        self.samples.append(sample)
        self.histogram[sample.name].add_sample(sample.value)
        self._dirty = True
        self.entity_positions = None

    def rebuild_frequency_vector(self):
        """Recompute f, number_of_samples, and number_of_unique_samples from the histogram."""
        n = len(self.samples)
        f = [0] * n
        for bar in self.histogram.values():
            f[bar.count - 1] += 1
        self.f = f
        self.number_of_samples = n
        self.number_of_unique_samples = len(self.histogram)
        self._dirty = False

    def _ensure_fresh(self):
        """Rebuild the frequency vector if cached stats are stale."""
        if self._dirty:
            self.rebuild_frequency_vector()

    @property
    def unique_count(self) -> int:
        """Number of distinct entities observed in this bucket."""
        self._ensure_fresh()
        return self.number_of_unique_samples

    def singleton_count(self) -> int:
        """Number of entities observed exactly once (singletons) in this bucket."""
        self._ensure_fresh()
        return self.f[0] if self.f else 0

    def sorted_samples(self) -> list[DataItem]:
        """Return all samples sorted by value ascending."""
        return sorted(self.samples, key=lambda x: x.value)

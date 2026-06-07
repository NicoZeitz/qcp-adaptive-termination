from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


@dataclass(slots=True)
class HistogramBar:
    lower_bound: float = float("inf")
    upper_bound: float = float("-inf")
    count: int = 0

    def add_sample(self, value: float) -> None:
        self.count += 1
        self.lower_bound = min(self.lower_bound, value)
        self.upper_bound = max(self.upper_bound, value)


@dataclass(slots=True)
class DataItem:
    name: str
    value: float
    source_id: Optional[str] = None


@dataclass
class Bucket:
    lower_bound: float = float("-inf")
    upper_bound: float = float("inf")
    samples: list[DataItem] = field(default_factory=list)
    histogram: dict[str, HistogramBar] = field(
        default_factory=lambda: defaultdict(HistogramBar)
    )
    f: list[int] = field(default_factory=list)
    number_of_samples: int = 0
    number_of_unique_samples: int = 0
    split_locked: bool = False
    idx: int = 0

    def add_sample(self, sample: DataItem) -> None:
        self.samples.append(sample)

        bar = self.histogram[sample.name]
        bar.add_sample(sample.value)

        self._rebuild_frequency_vector()

    def _rebuild_frequency_vector(self) -> None:
        n = len(self.samples)
        f = [0] * n
        for bar in self.histogram.values():
            # f[i] means number of classes seen exactly i+1 times.
            f[bar.count - 1] += 1
        self.f = f
        self.number_of_samples = n
        self.number_of_unique_samples = len(self.histogram)

    @property
    def unique_count(self) -> int:
        return self.number_of_unique_samples

    @property
    def f1_count(self) -> int:
        return self.f[0] if self.f else 0

    def sorted_samples(self) -> list[DataItem]:
        return sorted(self.samples, key=lambda x: x.value)

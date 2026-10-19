"""Unknown-unknowns estimator facade.

Combines observed frequency statistics, count estimators, value estimators,
adaptive bucket splitting, and Monte Carlo into a single lazy interface.
Estimated values are computed on demand — nothing is precomputed eagerly.

The estimator carries its full configuration (count/value estimator, bucket/MC
flags, max buckets) from construction, so the public methods take no estimator
parameters.
"""

import math
from dataclasses import dataclass

from scipy.stats import norm

from qcp_adaptive_termination.estimator.bucket import Bucket
from qcp_adaptive_termination.estimator.bucket_splitter import (
    raw_dynamic_buckets,
    warm_split,
)
from qcp_adaptive_termination.estimator.count_estimators import (
    estimate_count,
    estimate_count_bounds,
)
from qcp_adaptive_termination.estimator.data_item import DataItem
from qcp_adaptive_termination.estimator.frequency import FrequencyStats
from qcp_adaptive_termination.estimator.monte_carlo import mc_count
from qcp_adaptive_termination.estimator.value_estimators import estimate_delta
from qcp_adaptive_termination.models.config import EstimatorConfig
from qcp_adaptive_termination.models.estimation import (
    ConfidenceIntervalType,
    CountEstimator,
    ValueEstimator,
)


@dataclass(slots=True)
class _BucketCache:
    sample_count: int
    buckets: list[Bucket]
    sub_estimators: list[Estimator]


class Estimator:
    """
    Unknown-unknowns estimators for correcting aggregate query results.

    Accepts samples grouped by source. Each source contributes a list of DataItems;
    the estimator uses the overlap pattern across sources to infer how many entities
    are likely missing from the integrated dataset.

    All estimated values are computed lazily on first access and cached per
    (value_estimator, count_estimator, max_buckets) configuration.
    """

    DEFAULT_COUNT_ESTIMATOR: CountEstimator = CountEstimator.CHAO92
    DEFAULT_VALUE_ESTIMATOR: ValueEstimator = ValueEstimator.NAIVE
    DEFAULT_CONFIDENCE_INTERVAL_TYPE: ConfidenceIntervalType = ConfidenceIntervalType.CHAO84
    DEFAULT_MAX_BUCKET_SPLITS: int = 100

    def __init__(
        self,
        source_samples: list[list[DataItem]],
        config: EstimatorConfig | None = None,
    ) -> None:
        config = config or EstimatorConfig()
        self._stats = FrequencyStats(source_samples)
        self._bucket_cache: _BucketCache | None = None
        self._count_estimator = config.count_estimator
        self._value_estimator = config.value_estimator
        self._use_bucket = config.use_bucket
        self._use_monte_carlo = config.use_monte_carlo
        self._max_buckets = config.max_buckets
        self._confidence_interval_type = config.confidence_interval_type
        self._confidence_epsilon = config.confidence_epsilon

        # Lazy value caches, invalidated by add_source.
        self._cached_count: float | None = None
        self._cached_count_bounds: tuple[float, float] | None = None
        self._cached_delta: float | None = None
        self._cached_avg: float | None = None
        self._cached_quantiles: dict[float, float] = {}

    def add_source(self, items: list[DataItem]) -> None:
        """Add one source's observations, updating observed statistics incrementally."""
        self._stats.add_source(items)
        self._cached_count = None
        self._cached_count_bounds = None
        self._cached_delta = None
        self._cached_avg = None
        self._cached_quantiles.clear()

    # ── Observed (known) aggregates ───────────────────────────────────────────

    def sample_mean(self) -> float:
        return self._stats.sample_mean()

    def sample_std(self) -> float:
        return self._stats.sample_std()

    def γ_sq(self) -> float:
        return self._stats.γ_sq()

    def sample_coverage(self) -> float:
        return self._stats.sample_coverage()

    def sample_sum(self) -> float:
        return self._stats.sample_sum()

    def sample_count(self) -> int:
        return self._stats.sample_count()

    def sample_avg(self) -> float:
        return self._stats.sample_avg()

    def sample_min(self) -> float:
        return self._stats.sample_min()

    def sample_max(self) -> float:
        return self._stats.sample_max()

    def sample_quantile(self, q: float) -> float:
        return self._stats.sample_quantile(q)

    def sample_median(self) -> float:
        return self._stats.sample_median()

    def sample_cdf(self, x: float) -> float:
        return self._stats.sample_cdf(x)

    # ── Estimated counts ───────────────────────────────────────────

    def estimated_count(self) -> float:
        """Estimated total unique entity count (observed + missing)."""

        def _compute_count() -> float:
            if self._use_bucket:
                _, sub_estimators = self._get_sub_estimators()
                return max(self._stats.observed_unique_entities, sum(self._bucket_weights(sub_estimators)))

            if self._use_monte_carlo:
                return mc_count(self._stats, self._count_estimator)

            return estimate_count(self._stats, self._count_estimator)

        if self._cached_count is None:
            self._cached_count = _compute_count()
        return self._cached_count

    def estimated_count_lower_bound(self) -> float:
        """Lower bound for the estimated total unique entity count at the given confidence level."""
        lower, _ = self.estimated_count_bounds()
        return lower

    def estimated_count_upper_bound(self) -> float:
        """Upper bound for the estimated total unique entity count at the given confidence level."""
        _, upper = self.estimated_count_bounds()
        return upper

    def estimated_count_bounds(self) -> tuple[float, float]:
        """
        Lower and upper bounds for the estimated total unique entity count.

        Uses the estimator's configured confidence interval type and epsilon.
        """
        if self._cached_count_bounds is None:
            self._cached_count_bounds = estimate_count_bounds(
                self._stats, self._confidence_interval_type, self._confidence_epsilon
            )
        return self._cached_count_bounds

    # ── Estimated sums ───────────────────────────────────────────

    def estimated_sum(self) -> float:
        """
        Corrected SUM including the estimated contribution from missing entities.

        Returns inf if the delta is not finite.
        """
        d = self.delta()
        return math.inf if not math.isfinite(d) else self._stats.observed_value_sum + d

    def delta(self) -> float:
        """Estimated SUM contribution of missing entities (Δ = φ_D - φ_K)."""

        def _compute_delta() -> float:
            if self._use_bucket:
                _, sub_estimators = self._get_sub_estimators()
                return sum(sub.delta() for sub in sub_estimators)

            n_hat = self.estimated_count()
            return estimate_delta(self._stats, self._value_estimator, n_hat)

        if self._cached_delta is None:
            self._cached_delta = _compute_delta()
        return self._cached_delta

    def estimated_sum_lower_bound(self) -> float:
        """Lower bound for the corrected SUM at the given confidence level."""
        lower, _ = self.estimated_sum_bounds()
        return lower

    def estimated_sum_upper_bound(self) -> float:
        """Upper bound for the corrected SUM at the given confidence level."""
        _, upper = self.estimated_sum_bounds()
        return upper

    def estimated_sum_bounds(self) -> tuple[float, float]:
        """
        Lower and upper bounds for the corrected SUM.

        Computed as avg_bound * N_hat_bound. The two sides are statistically
        independent, so by Bonferroni's inequality the joint failure probability
        is at most the sum of the individual failure probabilities. Each side
        therefore receives half of the configured confidence epsilon so the
        combined bound holds with probability at least 1 - epsilon.
        """
        half_epsilon = self._confidence_epsilon / 2.0
        avg_lower, avg_upper = self.estimated_avg_bounds(half_epsilon)
        N_hat_lower, N_hat_upper = self.estimated_count_bounds()
        return (avg_lower * N_hat_lower, avg_upper * N_hat_upper)

    # ── Estimated averages ───────────────────────────────────────────

    def estimated_avg(self) -> float:
        """
        Estimated average across all entities (observed + missing).

        Without buckets, returns the observed sample average (law of large numbers).
        With buckets, returns a weighted mean of per-bucket sample averages where
        each bucket's weight is its estimated unique count (N_hat).
        """

        def _compute_avg() -> float:
            if not self._use_bucket:
                return self.sample_avg()

            _, sub_estimators = self._get_sub_estimators()
            weights = self._bucket_weights(sub_estimators)

            total_weight = sum(weights)
            weighted_sum = sum(w * sub.sample_avg() for sub, w in zip(sub_estimators, weights, strict=True))
            return weighted_sum / total_weight

        if self._cached_avg is None:
            self._cached_avg = _compute_avg()
        return self._cached_avg

    def estimated_avg_lower_bound(self) -> float:
        """Lower bound for the estimated average at the given confidence level."""
        lower, _ = self.estimated_avg_bounds()
        return lower

    def estimated_avg_upper_bound(self) -> float:
        """Upper bound for the estimated average at the given confidence level."""
        _, upper = self.estimated_avg_bounds()
        return upper

    def estimated_avg_bounds(self, confidence_epsilon: float | None = None) -> tuple[float, float]:
        """
        Lower and upper bounds for the estimated average.

        Since AVG = SUM / N and both SUM bounds and N bounds scale by N, the N
        cancels and the interval reduces to φ_K/c ± z·σ_K — a confidence interval
        on the observed mean value, independent of the count estimator.
        """

        epsilon = self._confidence_epsilon if confidence_epsilon is None else confidence_epsilon
        z = float(norm.ppf(1.0 - epsilon / 2.0))
        φ_K_over_c = self._stats.observed_value_sum / self._stats.observed_unique_entities
        σ_K = math.sqrt(
            sum((v.first_value - φ_K_over_c) ** 2 for v in self._stats.entities.values())
            / self._stats.observed_unique_entities
        ) / math.sqrt(self._stats.observed_unique_entities)

        lower = φ_K_over_c - z * σ_K
        upper = φ_K_over_c + z * σ_K
        return (lower, upper)

    # ── Estimated min/max ───────────────────────────────────────────

    def estimated_min(self) -> float | None:
        """
        Confirmed minimum value, or None if unknown unknowns may lie below it.

        Divides the value range into buckets and estimates the number of missing
        entities in the lowest-value bucket.  If that estimate is zero we have
        likely observed the true minimum and return it; otherwise we return None
        to signal that the true minimum is unknown.
        """
        _, subs = self._get_sub_estimators()
        sub = subs[0]
        if sub.estimated_count() - sub._stats.observed_unique_entities == 0:
            return self._stats.observed_min
        return None

    def estimated_max(self) -> float | None:
        """
        Confirmed maximum value, or None if unknown unknowns may lie above it.

        Divides the value range into buckets and estimates the number of missing
        entities in the highest-value bucket.  If that estimate is zero we have
        likely observed the true maximum and return it; otherwise we return None
        to signal that the true maximum is unknown.
        """
        _, subs = self._get_sub_estimators()
        sub = subs[-1]
        if sub.estimated_count() - sub._stats.observed_unique_entities == 0:
            return self._stats.observed_max
        return None

    def estimated_quantile(self, q: float) -> float:
        """
        Estimated q-th quantile (0 < q < 1) via the bucket-based CDF.

        Builds a weighted histogram where each bucket's weight is its estimated
        unique count (N_hat, including missing entities), then inverts the CDF.
        Linear interpolation within each bucket assumes uniform distribution within [lb, ub].
        """

        def _compute_quantile(q: float) -> float:
            buckets, subs = self._get_sub_estimators()
            weights = self._bucket_weights(subs)

            total = sum(weights)
            if total <= 0.0:
                return self.sample_quantile(q)

            cumulative = 0.0
            for i, (bucket, weight) in enumerate(zip(buckets, weights, strict=True)):
                prev_frac = cumulative / total
                cumulative += weight
                curr_frac = cumulative / total

                if curr_frac >= q or i == len(buckets) - 1:
                    bucket_span = curr_frac - prev_frac
                    t = (q - prev_frac) / bucket_span if bucket_span > 0.0 else 0.5
                    t = max(0.0, min(1.0, t))

                    lo = bucket.lower_bound if math.isfinite(bucket.lower_bound) else self._stats.observed_min
                    hi = bucket.upper_bound if math.isfinite(bucket.upper_bound) else self._stats.observed_max
                    # Linear interpolation inside the bucket
                    return lo + t * (hi - lo)

            return self._stats.observed_max

        if not 0.0 < q < 1.0:
            raise ValueError(f"q must be in (0, 1), got {q}")

        if q not in self._cached_quantiles:
            self._cached_quantiles[q] = _compute_quantile(q)
        return self._cached_quantiles[q]

    def estimated_median(self) -> float:
        """Estimated median (50th percentile). Delegates to estimated_quantile(0.5)."""
        return self.estimated_quantile(0.5)

    # ── Private Helpers: Buckets ───────────────────────────────────────────

    def _get_sub_estimators(self) -> tuple[list[Bucket], list[Estimator]]:
        """Return (buckets, per-bucket sub-Estimators), building both lazily and updating incrementally."""
        if self._bucket_cache is None:
            buckets = raw_dynamic_buckets(
                self._stats,
                self._value_estimator,
                self._count_estimator,
                max_buckets=self._max_buckets,
            )
            subs = [self._make_sub_estimator(b.to_sources()) for b in buckets]
            self._bucket_cache = _BucketCache(
                sample_count=len(self._stats.all_samples),
                buckets=buckets,
                sub_estimators=subs,
            )
        else:
            cache = self._bucket_cache
            new_samples = self._stats.all_samples[cache.sample_count :]
            if new_samples:
                old_id_to_sub = {id(b): sub for b, sub in zip(cache.buckets, cache.sub_estimators, strict=True)}
                new_buckets, new_items_by_id = warm_split(
                    cache.buckets,
                    new_samples,
                    self._value_estimator,
                    self._count_estimator,
                    max_buckets=self._max_buckets,
                )
                new_subs: list[Estimator] = []
                for b in new_buckets:
                    bid = id(b)
                    if bid in old_id_to_sub:
                        sub = old_id_to_sub[bid]
                        for source_items in new_items_by_id.get(bid, {}).values():
                            sub.add_source(source_items)
                        new_subs.append(sub)
                    else:
                        new_subs.append(self._make_sub_estimator(b.to_sources()))
                self._bucket_cache = _BucketCache(
                    sample_count=len(self._stats.all_samples),
                    buckets=new_buckets,
                    sub_estimators=new_subs,
                )
        cache = self._bucket_cache
        return cache.buckets, cache.sub_estimators

    def _make_sub_estimator(self, source_samples: list[list[DataItem]]) -> Estimator:
        """Create a per-bucket sub-estimator inheriting this estimator's config.

        Sub-estimators operate on a single bucket slice, so they must not
        bucket-split again (that would recurse infinitely).
        """

        return Estimator(
            source_samples,
            EstimatorConfig(
                count_estimator=self._count_estimator,
                value_estimator=self._value_estimator,
                use_bucket=False,
                use_monte_carlo=self._use_monte_carlo,
                max_buckets=self._max_buckets,
                confidence_interval_type=self._confidence_interval_type,
                confidence_epsilon=self._confidence_epsilon,
            ),
        )

    def _bucket_weights(self, subs: list[Estimator]) -> list[float]:
        """Per-bucket N_hat weights, falling back to observed count when not finite."""
        weights = []
        for sub in subs:
            n_hat = sub.estimated_count()
            weights.append(n_hat if math.isfinite(n_hat) else float(sub._stats.observed_unique_entities))
        return weights

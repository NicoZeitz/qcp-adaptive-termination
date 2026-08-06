import bisect
from functools import lru_cache
import math
import random
from collections.abc import Sequence
from enum import Enum
import sys

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.optimize import linprog
from scipy.stats import norm, poisson

from .data_item import DataItem
from .bucket import Bucket

MC_DEFAULT_NB_RUNS = 200
MC_DEFAULT_LAMBDA_GRID = tuple(round(v * 0.1, 1) for v in range(-4, 5))
MC_RANDOM_SEED = None
MC_DEFAULT_MAX_THETA_POINTS = 7
MC_LAPLACE_ALPHA = 0.1


class CountEstimator(Enum):
    """Strategy for estimating the total number of unique entities in the population."""

    CHAO92 = "chao92"
    """Coverage-based estimator corrected for skew in the publicity distribution via CV²."""

    CHAO84 = "chao84"
    """Estimator based on the ratio of singletons to doubletons (f1² / 2·f2)."""

    JACKKNIFE1 = "jackknife1"
    """First-order jackknife: adds the singleton count f1 as a bias correction."""

    JACKKNIFE2 = "jackknife2"
    """Second-order jackknife: adds 2·f1 and subtracts f2 for a lower-bias estimate."""

    ACE = "ace"
    """Abundance-based coverage estimator: splits entities into rare/abundant groups
    and applies coverage correction only to the rare group."""

    UNSEEN = "unseen"
    """Linear-programming estimator that fits a non-parametric frequency distribution
    and extrapolates the number of unseen species."""


class ValueEstimator(Enum):
    """Strategy for estimating the attribute values of missing entities."""

    NAIVE = "naive"
    """Mean substitution: assumes missing entities have the same average value
    as all observed entities (φ_K / c)."""

    FREQUENCY = "frequency"
    """Singleton-mean substitution: uses the average value of singletons (φ_f1 / f1)
    as a proxy for missing entities, which are most similar to rarely seen ones."""

    FREQUENCY_F12 = "frequency_f12"
    """Singleton+doubleton-mean substitution: averages over singletons and doubletons
    (φ_f12 / (f1 + f2)) to reduce variance compared to singletons alone."""

    GOOD_TURING = "good_turing"
    """Good-Turing frequency smoothing: derives Δ directly from φ_f1 and the ratio
    f1 / (n - f1) without a separate count estimator."""


class BoundType(Enum):
    """Method used to compute confidence intervals for the count estimate."""

    GOOD_TURING = "good_turing"
    """Bounds the missing mass M0 with an additive error term and converts to N via N = c / (1 - M0)."""

    CHAO84 = "chao84"
    """Asymmetric log-normal confidence interval for N - c; tighter than the Good-Turing bound.
    Falls back to Good-Turing when there are no doubletons."""


class Estimator:
    """Unknown-unknowns estimators for correcting aggregate query results.

    Accepts samples grouped by source. Each source contributes a list of DataItems;
    the estimator uses the overlap pattern across sources to infer how many entities
    are likely missing from the integrated dataset.

    Paper notation (Chung et al. 2018, Table 1):
        c       — number of unique observed entities (size of integrated database K)
        n       — total observations including duplicates (|S|)
        N       — unknown true population size (|D|)
        φ_K     — aggregate (SUM) over the observed database K
        φ_f1    — SUM of attribute values of singleton entities (seen exactly once)
        φ_f12   — SUM of attribute values of singleton + doubleton entities
        f_j     — number of entities observed exactly j times (_freq[j-1])
        M_0     — missing mass: total probability weight of unobserved entities
        C       — sample coverage: 1 − M_0 ≈ 1 − f_1/n
        N_hat       — estimated total unique entity count
        γ²      — squared coefficient of variation of the publicity distribution
        Δ       — estimated SUM contribution of missing entities
    """

    DEFAULT_COUNT_ESTIMATOR: CountEstimator = CountEstimator.CHAO92
    DEFAULT_VALUE_ESTIMATOR: ValueEstimator = ValueEstimator.NAIVE

    # ── Class-level attribute annotations (typed, for IDE support) ───────────

    _all_samples: list[DataItem]
    """All observations flattened across sources."""

    _entities: dict[str, list[float]]
    """name → [abundance, first_observed_value]; one entry per unique entity."""

    _c: int
    """c — number of unique observed entities. >= 1."""

    _n: int
    """n — total observations including duplicates. >= _c."""

    _φ_K: float
    """φ_K — SUM of first-observed attribute values across unique entities."""

    _observed_min: float
    """Minimum observed attribute value."""

    _observed_max: float
    """Maximum observed attribute value."""

    _freq: list[int]
    """f — frequency vector; _freq[i] = number of entities seen exactly i+1 times.
    len == _n. Each entry >= 0. sum(_freq) == _c."""

    _φ_f1: float
    """φ_{f₁} — SUM of attribute values of entities seen exactly once (singletons)."""

    _φ_f12: float
    """φ_{f₁₂} — SUM of attribute values of singletons and doubletons combined."""

    def __init__(self, source_samples: list[list[DataItem]]) -> None:
        self.source_samples = source_samples
        self._all_samples = [
            item for src in source_samples for item in src if item is not None
        ]
        if not self._all_samples:
            raise ValueError("Estimator requires at least one sample")

        self._observed_min = min(s.value for s in self._all_samples)
        self._observed_max = max(s.value for s in self._all_samples)

        # name -> [abundance, first_observed_value]
        self._entities: dict[str, list[float]] = {}
        for item in self._all_samples:
            if item.name in self._entities:
                self._entities[item.name][0] += 1.0
            else:
                self._entities[item.name] = [1.0, float(item.value)]

        self._c: int = len(self._entities)
        self._φ_K: float = sum(v[1] for v in self._entities.values())
        self._n: int = sum(int(v[0]) for v in self._entities.values())

        # Frequency vector: _freq[i] = number of entities seen exactly i+1 times.
        self._freq: list[int] = [0] * self._n
        self._φ_f1 = 0.0
        self._φ_f12 = 0.0
        for abundance, value in self._entities.values():
            cnt = int(abundance)
            self._freq[cnt - 1] += 1
            if cnt == 1:
                self._φ_f1 += value
                self._φ_f12 += value
            elif cnt == 2:
                self._φ_f12 += value

        self._mc_nb_runs: int = MC_DEFAULT_NB_RUNS
        self._mc_lambda_grid: tuple[float, ...] = MC_DEFAULT_LAMBDA_GRID
        self._mc_random_seed: int | None = MC_RANDOM_SEED
        self._mc_max_theta_points: int = MC_DEFAULT_MAX_THETA_POINTS
        self._mc_laplace_alpha: float = MC_LAPLACE_ALPHA

    def set_monte_carlo_parameters(
        self,
        nb_runs: int = MC_DEFAULT_NB_RUNS,
        lambda_grid: tuple[float, ...] = MC_DEFAULT_LAMBDA_GRID,
        random_seed: int | None = MC_RANDOM_SEED,
        max_theta_points: int = MC_DEFAULT_MAX_THETA_POINTS,
        laplace_alpha: float = MC_LAPLACE_ALPHA,
    ) -> None:
        """Set parameters for the Monte Carlo estimator."""
        self._mc_nb_runs = nb_runs
        self._mc_lambda_grid = lambda_grid
        self._mc_random_seed = random_seed
        self._mc_max_theta_points = max_theta_points
        self._mc_laplace_alpha = laplace_alpha

    # ── Public Helpers: core statistics ──────────────────────────────────────────────

    def sample_mean(self) -> float:
        """Mean of all raw observations (counting duplicates), not the per-entity mean."""
        return sum(item.value for item in self._all_samples) / self._n

    def sample_std(self) -> float:
        """Standard deviation of all raw observations (counting duplicates)."""
        mean = self.sample_mean()
        sq_mean = sum(item.value**2 for item in self._all_samples) / self._n
        return math.sqrt(max(0.0, sq_mean - mean * mean))

    def f1_count(self) -> int:
        """Number of entities observed exactly once (singletons)."""
        return self._freq[0]

    def γ_sq(self) -> float:
        """Squared coefficient of variation of the publicity distribution, estimated from f-statistics.

        Returns 0 when there is only one observation (no variance possible).
        Returns inf when sample coverage is zero (all entities are singletons).
        """
        if self._n == 1:
            return 0.0
        C_hat = self.sample_coverage()
        if C_hat == 0:
            return math.inf
        moment = sum((idx + 1) * idx * f for idx, f in enumerate(self._freq))
        γ_sq = (self._c / C_hat) * (moment / (self._n * (self._n - 1))) - 1.0
        return max(γ_sq, 0.0)

    def sample_coverage(self) -> float:
        """Good-Turing sample coverage C = 1 - f1/n.

        Measures the fraction of total probability mass covered by observed entities.
        0 when every entity is a singleton (maximum uncertainty), 1 when there are
        no singletons (complete coverage of the observed population).
        """
        return 1.0 - self.f1_count() / self._n

    # ── Observed (known) aggregates ───────────────────────────────────────────

    def sample_sum(self) -> float:
        """Sum of first-observed values across all unique observed entities."""
        return self._φ_K

    def sample_count(self) -> int:
        """Number of unique observed entities."""
        return self._c

    def sample_avg(self) -> float:
        """Average value across observed entities."""
        return self._φ_K / self._c

    def sample_min(self) -> float:
        """Minimum observed value."""
        return self._observed_min

    def sample_max(self) -> float:
        """Maximum observed value."""
        return self._observed_max

    # ── Estimated counts ───────────────────────────────────────────

    def estimated_count(
        self,
        count_estimator: CountEstimator = DEFAULT_COUNT_ESTIMATOR,
        *,
        use_bucket: bool = True,
        use_monte_carlo: bool = False,
        bucket_value_estimator: ValueEstimator = DEFAULT_VALUE_ESTIMATOR,
    ) -> float:
        """Estimated total unique entity count (observed + missing)."""
        if use_bucket:
            return max(
                self._c,
                sum(
                    Estimator(b.to_sources()).estimated_count(
                        count_estimator,
                        use_bucket=False,
                        use_monte_carlo=use_monte_carlo,
                    )
                    for b in self._dynamic_buckets(
                        bucket_value_estimator, count_estimator
                    )
                ),
            )

        if use_monte_carlo:
            return self._mc_count(count_estimator)

        if count_estimator is CountEstimator.CHAO92:
            return self.chao92()
        if count_estimator is CountEstimator.CHAO84:
            return self.chao84()
        if count_estimator is CountEstimator.JACKKNIFE1:
            return self.jackknife1()
        if count_estimator is CountEstimator.JACKKNIFE2:
            return self.jackknife2()
        if count_estimator is CountEstimator.ACE:
            return self.ace()
        if count_estimator is CountEstimator.UNSEEN:
            return self.unseen()
        raise ValueError(f"Unsupported count count_estimator: {count_estimator}")

    def chao92(self) -> float:
        """Estimate total unique entity count using sample coverage and coefficient of variation.

        Uses the Good-Turing estimate of sample coverage C = 1 - f1/n, then corrects
        for skew in the publicity distribution via the squared CV γ².
        Returns inf when coverage is zero (every entity seen exactly once).
        """
        C_hat = self.sample_coverage()
        if C_hat == 0:
            return math.inf
        γ_sq = self.γ_sq()
        if not math.isfinite(γ_sq):
            return math.inf
        N_hat = (self._c / C_hat) + (self._n * (1.0 - C_hat) / C_hat) * γ_sq
        return max(self._c, N_hat)

    def chao84(self) -> float:
        """Estimate total unique entity count using the ratio of singletons to doubletons.

        Falls back to the observed count when there are no doubletons (f2 == 0),
        since the estimator is undefined in that case.
        """
        if len(self._freq) == 1 or self._freq[1] == 0:
            return self._c
        return max(
            self._c,
            self._c + (self._freq[0] * self._freq[0]) / (2.0 * self._freq[1]),
        )

    def jackknife1(self) -> float:
        """Estimate total unique entity count using the first-order jackknife correction.

        Adds the singleton count f1 to the observed count c as a bias correction
        for the number of unseen entities.
        """
        return self._c + self._freq[0]

    def jackknife2(self) -> float:
        """Estimate total unique entity count using the second-order jackknife correction.

        Extends jackknife1 by subtracting the doubleton count f2, reducing the
        bias when many entities appear exactly twice.
        """
        f2 = self._freq[1] if len(self._freq) > 1 else 0
        return self._c + 2 * self._freq[0] - f2

    def ace(self) -> float:
        """Abundance-based coverage estimator (ACE) count estimate.

        Splits entities into rare (abundance <= 10) and abundant (abundance > 10).
        Uses sample coverage of the rare group to estimate the number of unseen
        rare entities, then adds the observed abundant count.
        Falls back to chao92 when all rare entities are singletons (zero coverage).
        """
        c_rare = sum(
            1 for abundance, _ in self._entities.values() if int(abundance) <= 10
        )
        c_abundant = self._c - c_rare
        n_rare = sum(
            int(abundance)
            for abundance, _ in self._entities.values()
            if int(abundance) <= 10
        )

        if n_rare == 0:
            return self._c

        cov_ace = 1.0 - self._freq[0] / n_rare
        if cov_ace == 0:
            return self.chao92()
        if n_rare == 1:
            cv = math.sqrt(self._n)
        else:
            sum_term = 0.0
            for i in range(min(10, len(self._freq))):
                sum_term += i * (i + 1) * self._freq[i]
            cv = max(
                (c_rare / cov_ace) * (sum_term / (n_rare * (n_rare - 1))) - 1.0,
                0.0,
            )
        est = c_abundant + c_rare / cov_ace + self._freq[0] * cv / cov_ace
        return max(self._c, est)

    def unseen(self) -> float:
        """Estimate total unique entity count using a linear-programming based approach.

        Fits a non-parametric probability distribution over entity frequencies via LP,
        then estimates the total species count from that distribution.
        Falls back to chao92 if the LP solver fails.
        """
        try:
            est = self._unseen_lp()
            return max(self._c, est)
        except Exception:
            return self.chao92()

    # ── Estimated counts (Bounds) ───────────────────────────────────────────

    def estimated_count_lower_bound(
        self,
        confidence_interval_type: BoundType = BoundType.CHAO84,
        confidence_epsilon: float = 0.01,
    ) -> float:
        """Lower bound for the estimated total unique entity count at the given confidence level."""
        lower, _ = self.estimated_count_bounds(
            confidence_interval_type, confidence_epsilon
        )
        return lower

    def estimated_count_upper_bound(
        self,
        confidence_interval_type: BoundType = BoundType.CHAO84,
        confidence_epsilon: float = 0.01,
    ) -> float:
        """Upper bound for the estimated total unique entity count at the given confidence level."""
        _, upper = self.estimated_count_bounds(
            confidence_interval_type, confidence_epsilon
        )
        return upper

    def estimated_count_bounds(
        self,
        confidence_interval_type: BoundType = BoundType.CHAO84,
        confidence_epsilon: float = 0.01,
    ) -> tuple[float, float]:
        """Lower and upper bounds for the estimated total unique entity count.

        confidence_epsilon is the total error probability; e.g. 0.01 gives 99% confidence.
        """
        if confidence_epsilon <= 0.0 or confidence_epsilon > 1.0:
            raise ValueError("confidence_epsilon must be in (0, 1]")

        if confidence_interval_type == BoundType.CHAO84:
            return self.chao84_confidence_interval(confidence_epsilon)
        if confidence_interval_type == BoundType.GOOD_TURING:
            return self.good_turing_confidence_interval(confidence_epsilon)
        raise ValueError(
            f"Unsupported confidence interval type: {confidence_interval_type}"
        )

    def good_turing_confidence_interval(
        self, confidence_epsilon: float
    ) -> tuple[float, float]:
        """Confidence interval for total unique entity count using a Good-Turing error bound.

        Bounds the missing mass M0 = f1/n with an additive error term derived from
        the sample size, then converts the M0 interval to an N interval via N = c / (1 - M0).
        """
        numeric_eps = sys.float_info.epsilon
        n = self._n
        f1 = self.f1_count()
        c = self._c

        M0_hat = f1 / n  # Good-Turing estimate of missing mass
        error = (2.0 * math.sqrt(2.0) + math.sqrt(3.0)) * math.sqrt(
            math.log(3.0 / confidence_epsilon) / n
        )

        M0_upper = min(1.0, M0_hat + error)
        M0_lower = max(0.0, M0_hat - error)

        N_hat_upper = c / max(numeric_eps, 1.0 - M0_upper)
        N_hat_lower = c / max(numeric_eps, 1.0 - M0_lower)
        return (max(self._c, N_hat_lower), max(self._c, N_hat_upper))

    def chao84_confidence_interval(
        self, confidence_epsilon: float
    ) -> tuple[float, float]:
        """Confidence interval for total unique entity count using the asymmetric Chao84 interval.

        Uses the log-normal approximation for N - c, which gives a tighter interval
        than the symmetric normal approximation. Falls back to the Good-Turing interval
        when there are no doubletons (variance formula requires f2 > 0). When variance
        is zero (f1 == 0), the CI collapses to the point estimate.
        """
        f1 = self._freq[0]
        f2 = self._freq[1] if len(self._freq) > 1 else 0
        c = self._c

        N_hat = self.chao84()
        N_hat_minus_c = N_hat - c
        if N_hat_minus_c == 0:
            return (c, c)

        if f2 == 0:
            return self.good_turing_confidence_interval(confidence_epsilon)

        z_prime = float(norm.ppf(1.0 - confidence_epsilon / 2.0))

        σ_sq = f2 * (0.5 * (f1 / f2) ** 2 + (f1 / f2) ** 3 + 0.25 * (f1 / f2) ** 4)

        T = math.exp(z_prime * math.sqrt(math.log(1.0 + σ_sq / (N_hat_minus_c**2))))

        N_hat_lower = c + N_hat_minus_c / T
        N_hat_upper = c + N_hat_minus_c * T
        return (max(self._c, N_hat_lower), max(self._c, N_hat_upper))

    # ── Estimated sums ───────────────────────────────────────────

    def estimated_sum(
        self,
        value_estimator: ValueEstimator = DEFAULT_VALUE_ESTIMATOR,
        count_estimator: CountEstimator = DEFAULT_COUNT_ESTIMATOR,
        *,
        use_bucket: bool = True,
        use_monte_carlo: bool = False,
    ) -> float:
        """Corrected SUM including the estimated contribution from missing entities.

        Returns inf if the delta is not finite.
        """
        d = self.delta(
            value_estimator,
            count_estimator,
            use_bucket=use_bucket,
            use_monte_carlo=use_monte_carlo,
        )
        return math.inf if not math.isfinite(d) else self._φ_K + d

    def delta(
        self,
        value_estimator: ValueEstimator = DEFAULT_VALUE_ESTIMATOR,
        count_estimator: CountEstimator = DEFAULT_COUNT_ESTIMATOR,
        *,
        use_bucket: bool = True,
        use_monte_carlo: bool = False,
    ) -> float:
        """Estimated SUM contribution of missing entities (Δ = φ_D - φ_K)."""
        if use_bucket:
            return self._bucket_delta(
                value_estimator, count_estimator, use_monte_carlo=use_monte_carlo
            )

        if value_estimator is ValueEstimator.NAIVE:
            return self._naive_delta(
                count_estimator, use_bucket=False, use_monte_carlo=use_monte_carlo
            )
        if value_estimator is ValueEstimator.FREQUENCY:
            return self._frequency_delta(
                count_estimator, use_bucket=False, use_monte_carlo=use_monte_carlo
            )
        if value_estimator is ValueEstimator.GOOD_TURING:
            return self._good_turing_delta()
        if value_estimator is ValueEstimator.FREQUENCY_F12:
            return self._frequency_f12_delta(
                count_estimator, use_bucket=False, use_monte_carlo=use_monte_carlo
            )

        raise ValueError(f"Unsupported estimator: {value_estimator}")

    def _bucket_delta(
        self,
        value_estimator: ValueEstimator,
        count_estimator: CountEstimator,
        *,
        use_monte_carlo: bool = False,
    ) -> float:
        """Sum of per-bucket deltas across all dynamically generated buckets."""
        return sum(
            Estimator(b.to_sources()).delta(
                value_estimator,
                count_estimator,
                use_bucket=False,
                use_monte_carlo=use_monte_carlo,
            )
            for b in self._dynamic_buckets(value_estimator, count_estimator)
        )

    def _naive_delta(
        self,
        count_estimator: CountEstimator = DEFAULT_COUNT_ESTIMATOR,
        *,
        use_monte_carlo: bool = False,
        use_bucket: bool = True,
    ) -> float:
        """Estimated SUM of missing entities using mean substitution.

        Assumes missing entities have the same average value as the observed ones:
        Δ = (φ_K / c) * (N_hat - c).
        """
        N_hat = self.estimated_count(
            count_estimator, use_bucket=use_bucket, use_monte_carlo=use_monte_carlo
        )
        if not math.isfinite(N_hat):
            return math.inf
        return (self._φ_K / self._c) * (N_hat - self._c)

    def _frequency_delta(
        self,
        count_estimator: CountEstimator = DEFAULT_COUNT_ESTIMATOR,
        *,
        use_bucket: bool = True,
        use_monte_carlo: bool = False,
    ) -> float:
        """Estimated SUM of missing entities using the average singleton value.

        Uses φ_f1 / f1 as the value estimate instead of the overall mean,
        since singletons are the best proxy for unseen entities.
        Returns 0 when there are no singletons.
        """
        f1 = self.f1_count()
        if f1 == 0:
            return 0.0

        N_hat = self.estimated_count(
            count_estimator, use_bucket=use_bucket, use_monte_carlo=use_monte_carlo
        )
        if not math.isfinite(N_hat):
            return math.inf
        return (self._φ_f1 / f1) * (N_hat - self._c)

    def _good_turing_delta(self) -> float:
        """Estimated SUM of missing entities using Good-Turing frequency smoothing.

        Does not require a separate count estimator: derives Δ directly from
        the singleton sum and the ratio f1 / (n - f1), which is the Good-Turing
        estimate of the missing probability mass scaled to the observed aggregate.
        Returns 0 when there are no singletons or all observations are singletons.
        """
        f1 = self.f1_count()
        if f1 == 0:
            return 0.0

        denom = self._n - f1
        return self._φ_f1 * self._c / denom if denom > 0 else 0.0

    def _frequency_f12_delta(
        self,
        count_estimator: CountEstimator = DEFAULT_COUNT_ESTIMATOR,
        *,
        use_bucket: bool = True,
        use_monte_carlo: bool = False,
    ) -> float:
        """Estimated SUM of missing entities using the average singleton+doubleton value.

        Uses φ_f12 / (f1 + f2) as the value estimate, blending singletons and
        doubletons to reduce variance compared to singletons alone.
        Returns 0 when there are neither singletons nor doubletons.
        """
        f2 = self._freq[1] if len(self._freq) > 1 else 0
        denom = self.f1_count() + f2
        if denom == 0:
            return 0.0
        N_hat = self.estimated_count(
            count_estimator, use_bucket=use_bucket, use_monte_carlo=use_monte_carlo
        )
        if not math.isfinite(N_hat):
            return math.inf
        return (self._φ_f12 / denom) * (N_hat - self._c)

    # ── Estimated sums (Bounds) ───────────────────────────────────────────

    def estimated_sum_lower_bound(
        self,
        bound_type: BoundType = BoundType.CHAO84,
        confidence_epsilon: float = 0.01,
    ) -> float:
        """Lower bound for the corrected SUM at the given confidence level."""
        lower, _ = self.estimated_sum_bounds(bound_type, confidence_epsilon)
        return lower

    def estimated_sum_upper_bound(
        self,
        bound_type: BoundType = BoundType.CHAO84,
        confidence_epsilon: float = 0.01,
    ) -> float:
        """Upper bound for the corrected SUM at the given confidence level."""
        _, upper = self.estimated_sum_bounds(bound_type, confidence_epsilon)
        return upper

    def estimated_sum_bounds(
        self,
        bound_type: BoundType = BoundType.CHAO84,
        confidence_epsilon: float = 0.01,
    ) -> tuple[float, float]:
        """Lower and upper bounds for the corrected SUM.

        Computed as avg_bound * N_hat_bound. The two sides are statistically
        independent, so by Bonferroni's inequality the joint failure probability
        is at most the sum of the individual failure probabilities. Each side
        therefore receives half of confidence_epsilon so the combined bound holds
        with probability at least 1 - confidence_epsilon.
        """
        half_epsilon = confidence_epsilon / 2.0
        avg_lower, avg_upper = self.estimated_avg_bounds(half_epsilon)
        N_hat_lower, N_hat_upper = self.estimated_count_bounds(bound_type, half_epsilon)
        return (avg_lower * N_hat_lower, avg_upper * N_hat_upper)

    # ── Estimated averages ───────────────────────────────────────────

    def estimated_avg(
        self,
        use_bucket: bool = True,
        *,
        bucket_count_estimator: CountEstimator = DEFAULT_COUNT_ESTIMATOR,
        bucket_value_estimator: ValueEstimator = DEFAULT_VALUE_ESTIMATOR,
        bucket_use_monte_carlo: bool = False,
    ) -> float:
        """Estimated average across all entities (observed + missing).

        Without buckets, returns the observed sample average (law of large numbers).
        With buckets, returns a weighted mean of per-bucket sample averages where
        each bucket's weight is its estimated unique count (N_hat).
        """
        if not use_bucket:
            return self.sample_avg()

        buckets = self._dynamic_buckets(bucket_value_estimator, bucket_count_estimator)

        total_weight = 0.0
        weighted_sum = 0.0

        for b in buckets:
            sub = Estimator(b.to_sources())
            n_hat = sub.estimated_count(
                bucket_count_estimator,
                use_bucket=False,
                use_monte_carlo=bucket_use_monte_carlo,
            )
            weight = n_hat if math.isfinite(n_hat) else sub._c
            total_weight += weight
            weighted_sum += weight * sub.sample_avg()

        return weighted_sum / total_weight

    # ── Estimated averages (Bounds) ───────────────────────────────────────────

    def estimated_avg_lower_bound(
        self,
        confidence_epsilon: float = 0.01,
    ) -> float:
        """Lower bound for the estimated average at the given confidence level."""
        lower, _ = self.estimated_avg_bounds(confidence_epsilon)
        return lower

    def estimated_avg_upper_bound(
        self,
        confidence_epsilon: float = 0.01,
    ) -> float:
        """Upper bound for the estimated average at the given confidence level."""
        _, upper = self.estimated_avg_bounds(confidence_epsilon)
        return upper

    def estimated_avg_bounds(
        self,
        confidence_epsilon: float = 0.01,
    ) -> tuple[float, float]:
        """Lower and upper bounds for the estimated average.

        Since AVG = SUM / N and both SUM bounds and N bounds scale by N, the N
        cancels and the interval reduces to φ_K/c ± z·σ_K — a confidence interval
        on the observed mean value, independent of the count estimator.
        """
        z = float(norm.ppf(1.0 - confidence_epsilon / 2.0))
        φ_K_over_c = self._φ_K / self._c
        σ_K = math.sqrt(
            sum((v[1] - φ_K_over_c) ** 2 for v in self._entities.values()) / self._c
        ) / math.sqrt(self._c)

        lower = φ_K_over_c - z * σ_K
        upper = φ_K_over_c + z * σ_K
        return (lower, upper)

    # ── Estimated min/max ───────────────────────────────────────────

    def estimated_min(
        self,
        count_estimator: CountEstimator = DEFAULT_COUNT_ESTIMATOR,
        *,
        bucket_value_estimator: ValueEstimator = DEFAULT_VALUE_ESTIMATOR,
        use_monte_carlo: bool = False,
    ) -> float | None:
        """Confirmed minimum value, or None if unknown unknowns may lie below it.

        Divides the value range into buckets and estimates the number of missing
        entities in the lowest-value bucket.  If that estimate is zero we have
        likely observed the true minimum and return it; otherwise we return None
        to signal that the true minimum is unknown.
        """
        buckets = self._dynamic_buckets(bucket_value_estimator, count_estimator)
        sub = Estimator(buckets[0].to_sources())
        if (
            sub.estimated_count(
                count_estimator, use_bucket=False, use_monte_carlo=use_monte_carlo
            )
            - sub._c
            == 0
        ):
            return self._observed_min
        return None

    def estimated_max(
        self,
        count_estimator: CountEstimator = DEFAULT_COUNT_ESTIMATOR,
        *,
        bucket_value_estimator: ValueEstimator = DEFAULT_VALUE_ESTIMATOR,
        use_monte_carlo: bool = False,
    ) -> float | None:
        """Confirmed maximum value, or None if unknown unknowns may lie above it.

        Divides the value range into buckets and estimates the number of missing
        entities in the highest-value bucket.  If that estimate is zero we have
        likely observed the true maximum and return it; otherwise we return None
        to signal that the true maximum is unknown.
        """
        buckets = self._dynamic_buckets(bucket_value_estimator, count_estimator)
        sub = Estimator(buckets[-1].to_sources())
        if (
            sub.estimated_count(
                count_estimator, use_bucket=False, use_monte_carlo=use_monte_carlo
            )
            - sub._c
            == 0
        ):
            return self._observed_max
        return None

    # ── Private Helpers: Unseen ───────────────────────────────────────────

    def _unseen_lp(self) -> float:
        grid_factor = 1.1
        alpha = 0.5

        xlp_min = 1.0 / (self._n * max(10, self._n))

        sparse_species = 0.0
        sparse_mass = 0.0
        f_lp_raw = [0.0] * len(self._freq)
        for idx, fi in enumerate(self._freq):
            if fi == 0:
                continue

            i = idx + 1
            radius = int(math.ceil(math.sqrt(i)))
            lo = max(0, idx - radius)
            hi = min(len(self._freq) - 1, idx + radius)
            local_sum = sum(self._freq[j] for j in range(lo, hi + 1))

            # Match MATLAB split rule: sparse if local mass is below 2*sqrt(i).
            if local_sum < 2.0 * math.sqrt(i):
                sparse_species += fi
                sparse_mass += (i / self._n) * fi
            else:
                f_lp_raw[idx] = float(fi)

        dense_positions = [i for i, v in enumerate(f_lp_raw) if v > 0.0]
        if not dense_positions:
            return max(float(self._c), sparse_species)

        f_max = max(dense_positions) + 1
        f_lp = f_lp_raw[:f_max] + [0.0] * int(math.ceil(math.sqrt(f_max)))

        xlp_max = f_max / self._n
        if xlp_max <= xlp_min:
            return self.chao92()

        grid_size = (
            int(math.ceil(math.log(xlp_max / xlp_min) / math.log(grid_factor))) + 1
        )
        xlp = [xlp_min * (grid_factor**i) for i in range(max(grid_size, 2))]

        m = len(xlp)
        r = len(f_lp)
        coef = np.zeros((r, m), dtype=float)
        for i in range(r):
            k = i + 1
            for j, x in enumerate(xlp):
                coef[i, j] = poisson.pmf(k, self._n * x) / x

        # Variables are [y_1..y_m, s_plus_1, s_minus_1, ..., s_plus_r, s_minus_r]
        # with y_j = x_j * h_j (MATLAB-style scaling for conditioning).
        lp1_obj = np.zeros(m + 2 * r, dtype=float)
        for i, fi in enumerate(f_lp):
            w = 1.0 / math.sqrt(fi + 1.0)
            lp1_obj[m + 2 * i] = w
            lp1_obj[m + 2 * i + 1] = w

        a_ub = np.zeros((2 * r, m + 2 * r), dtype=float)
        b_ub = np.zeros(2 * r, dtype=float)
        for i in range(r):
            a_ub[2 * i, :m] = coef[i, :]
            a_ub[2 * i + 1, :m] = -coef[i, :]
            a_ub[2 * i, m + 2 * i] = -1.0
            a_ub[2 * i + 1, m + 2 * i + 1] = -1.0
            b_ub[2 * i] = f_lp[i]
            b_ub[2 * i + 1] = -f_lp[i]

        a_eq = np.zeros((1, m + 2 * r), dtype=float)
        a_eq[0, :m] = 1.0
        lp_mass = max(1e-12, 1.0 - sparse_mass)
        b_eq = np.array([lp_mass], dtype=float)

        bounds = [(0.0, None)] * (m + 2 * r)
        res1 = linprog(
            c=lp1_obj,
            A_ub=a_ub,
            b_ub=b_ub,
            A_eq=a_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )
        if not res1.success:
            return self.chao92()

        lp2_obj = np.zeros(m + 2 * r, dtype=float)
        lp2_obj[:m] = 1.0 / np.array(xlp, dtype=float)

        a2_ub = np.vstack([a_ub, lp1_obj])
        b2_ub = np.concatenate([b_ub, np.array([res1.fun + alpha], dtype=float)])

        res2 = linprog(
            c=lp2_obj,
            A_ub=a2_ub,
            b_ub=b2_ub,
            A_eq=a_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )
        if not res2.success:
            return self.chao92()

        y = res2.x[:m]
        lp_species = float(sum(y_i / x_i for y_i, x_i in zip(y, xlp)))
        return sparse_species + lp_species

    # ── Private Helpers: Buckets ───────────────────────────────────────────

    @lru_cache(maxsize=None)
    def _dynamic_buckets(
        self,
        bucket_value_estimator: ValueEstimator,
        bucket_count_estimator: CountEstimator,
    ) -> list[Bucket]:
        """Split the value range into adaptive buckets that minimize total estimated delta.

        Starts with one bucket covering the full range, then recursively tries to split
        each bucket at every observed value. A split is accepted only if it reduces the
        sum of per-bucket estimated deltas. Stops when no further improvement is possible.
        Results are cached keyed by (value_estimator, count_estimator).
        """
        init_bucket = Bucket(self._observed_min, self._observed_max)
        for s in sorted(self._all_samples, key=lambda x: x.value):
            init_bucket.add_sample(s)

        buckets: list[Bucket] = [init_bucket]
        previous_count = 0

        while len(buckets) > previous_count:
            previous_count = len(buckets)
            next_buckets: list[Bucket] = []

            for bucket in buckets:
                if bucket.split_locked:
                    next_buckets.append(bucket)
                    continue

                if bucket.f1_count >= bucket.number_of_samples:
                    next_buckets.append(bucket)
                    continue

                lb, ub = bucket.lower_bound, bucket.upper_bound

                best_left, best_right, best_score = None, None, math.inf
                for split_value in [(lb + ub) / 2.0] + [
                    item.value for item in bucket.sorted_samples()
                ]:
                    left, right = self._split_bucket(bucket, split_value)
                    if left.number_of_samples == 0 or right.number_of_samples == 0:
                        continue

                    # fmt: off
                    score = self._bucket_sum_est(left, bucket_value_estimator, bucket_count_estimator) + \
                            self._bucket_sum_est(right, bucket_value_estimator, bucket_count_estimator)
                    # fmt: on

                    if score < best_score:
                        best_left, best_right, best_score = left, right, score

                if (
                    best_left is None
                    or best_right is None
                    or best_score
                    >= self._bucket_sum_est(
                        bucket, bucket_value_estimator, bucket_count_estimator
                    )
                ):
                    bucket.split_locked = True
                    next_buckets.append(bucket)
                else:
                    next_buckets.extend([best_left, best_right])

            buckets = next_buckets

        return sorted(buckets, key=lambda x: x.lower_bound)

    def _bucket_sum_est(
        self,
        bucket: Bucket,
        bucket_value_estimator: ValueEstimator,
        bucket_count_estimator: CountEstimator,
    ) -> float:
        """Estimated SUM for a single bucket, used as the splitting score."""
        return Estimator(bucket.to_sources()).estimated_sum(
            bucket_value_estimator, bucket_count_estimator, use_bucket=False
        )

    @staticmethod
    def _split_bucket(bucket: Bucket, split_value: float) -> tuple[Bucket, Bucket]:
        """Split a bucket at split_value into left (≤ split_value) and right (> split_value).

        Assumes bucket.samples is sorted by value ascending (guaranteed by _dynamic_buckets).
        Uses bisect to find the partition point in O(log n), then bulk-loads each half and
        rebuilds the frequency vector once per child instead of once per sample.
        """
        samples = bucket.samples
        keys = [s.value for s in samples]
        cut = bisect.bisect_right(keys, split_value)

        left = Bucket(bucket.lower_bound, split_value)
        right = Bucket(split_value, bucket.upper_bound)

        for s in samples[:cut]:
            left.samples.append(s)
            left.histogram[s.name].add_sample(s.value)
        for s in samples[cut:]:
            right.samples.append(s)
            right.histogram[s.name].add_sample(s.value)

        left.rebuild_frequency_vector()
        right.rebuild_frequency_vector()
        return left, right

    # ── Private Helpers: Monte-Carlo ───────────────────────────────────────────

    def _mc_count(
        self,
        count_estimator: CountEstimator = DEFAULT_COUNT_ESTIMATOR,
    ) -> int:
        """Estimate total unique entity count via Monte Carlo simulation.

        Searches over a grid of (N, λ) parameters where N is the candidate population
        size and λ governs the skew of an exponential publicity distribution.
        For each candidate, simulates the sampling process and measures KL-divergence
        from the observed frequency statistics. Fits an RBF surface over the grid
        and returns the N at the minimum.
        """
        n_estimated = self.estimated_count(count_estimator, use_bucket=False)
        n_ub = (
            self._c * 2
            if not math.isfinite(n_estimated)
            else max(self._c, int(math.ceil(n_estimated)))
        )

        # If Chao is not finite or collapses to the observed count, there is no
        # meaningful N-grid search to perform for Monte Carlo.
        if n_ub == self._c:
            return self._c

        jump = max(1, math.ceil((n_ub - self._c) / self._mc_max_theta_points))
        i_last = (n_ub - self._c + 1) // jump
        n_grid = [self._c + i * jump for i in range(max(i_last, 1))]
        if n_grid[-1] < n_ub:
            n_grid.append(n_ub)
        elif n_grid[-1] > n_ub:
            n_grid[-1] = n_ub

        n_w = [len(src) for src in self.source_samples]
        points: list[tuple[float, float, float]] = [
            (float(tn), float(tl), self._mc_distance(tn, tl, n_w, self._mc_nb_runs))
            for tn in n_grid
            for tl in self._mc_lambda_grid
        ]
        best_n, _ = self._rbf_search_min(points, self._c, n_ub, self._mc_lambda_grid)
        return int(best_n)

    def _mc_distance(
        self,
        theta_n: int,
        theta_lambda: float,
        n_w: Sequence[int],
        nb_runs: int,
    ) -> float:
        """Average KL-divergence between observed and simulated f-statistic distributions."""
        rng = random.Random(self._mc_random_seed)
        total = 0.0

        # Observed f-statistics: fs[k] = number of entities seen exactly k times.
        fs_obs: dict[int, int] = {}
        for abundance, _ in self._entities.values():
            k = int(abundance)
            fs_obs[k] = fs_obs.get(k, 0) + 1

        for _ in range(nb_runs):
            # Simulate one integrated dataset Q under parameters (theta_n, theta_lambda).
            sim_abundance: dict[int, int] = {}  # entity_idx -> count across all workers
            for worker_n in n_w:
                local: set[int] = set()
                target = min(worker_n, theta_n)
                while len(local) < target:
                    idx = int(math.floor(rng.random() * theta_n))
                    p = math.exp(-idx * theta_lambda / max(theta_n, 1))
                    if rng.random() <= p:
                        local.add(idx)
                for idx in local:
                    sim_abundance[idx] = sim_abundance.get(idx, 0) + 1

            # Convert simulation to f-statistics.
            fs_sim: dict[int, int] = {}
            for cnt in sim_abundance.values():
                fs_sim[cnt] = fs_sim.get(cnt, 0) + 1

            # All frequency counts present in either distribution.
            all_counts = set(fs_obs.keys()) | set(fs_sim.keys())
            if not all_counts:
                continue

            alpha = self._mc_laplace_alpha
            # Smooth F_S: entities with counts only in Q get alpha mass.
            p_denom = sum(fs_obs.values()) + alpha * len(all_counts)
            q_denom = sum(fs_sim.values()) + alpha * len(all_counts)

            kl = 0.0
            for k in all_counts:
                p = (fs_obs.get(k, 0) + alpha) / p_denom
                q = (fs_sim.get(k, 0) + alpha) / q_denom
                kl += p * math.log(p / q)
            total += kl

        return total / max(nb_runs, 1)

    @staticmethod
    def _rbf_search_min(
        points: Sequence[tuple[float, float, float]],
        n_lb: int,
        n_ub: int,
        lambda_grid: Sequence[float],
    ) -> tuple[int, float]:
        """Fit a thin-plate-spline RBF surface over (N, λ, KL) points and return the minimum.

        Falls back to the raw minimum point when there are fewer than 3 points or
        when all N or λ values are identical (degenerate 2D surface).
        """
        if len(points) < 3:
            n_best, l_best, _ = min(points, key=lambda p: p[2])
            return int(n_best), float(l_best)

        n_values = {p[0] for p in points}
        lambda_values = {p[1] for p in points}
        # RBF interpolation in 2D needs non-degenerate coordinates.
        if len(n_values) < 2 or len(lambda_values) < 2:
            n_best, l_best, _ = min(points, key=lambda p: p[2])
            return int(n_best), float(l_best)

        xy = np.array([[p[0], p[1]] for p in points], dtype=float)
        z = np.array([p[2] for p in points], dtype=float)
        xy_scale = xy.std(axis=0)
        xy_scale[xy_scale == 0] = 1.0
        try:
            rbf = RBFInterpolator(xy / xy_scale, z, kernel="thin_plate_spline")
        except np.linalg.LinAlgError:
            n_best, l_best, _ = min(points, key=lambda p: p[2])
            return int(n_best), float(l_best)

        candidates = np.array(
            [[n, lam] for n in range(n_lb, n_ub + 1) for lam in lambda_grid],
            dtype=float,
        )
        scores = rbf(candidates / xy_scale)
        best_idx = int(np.argmin(scores))
        return int(candidates[best_idx, 0]), float(candidates[best_idx, 1])

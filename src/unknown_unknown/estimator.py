from __future__ import annotations

import math
import random
from collections.abc import Sequence
from enum import Enum

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.optimize import linprog
from scipy.stats import poisson

from .bucket import Bucket, DataItem

MC_DEFAULT_NB_RUNS = 200
MC_DEFAULT_LAMBDA_GRID_MC = (-4.0, -2.0, 0.0, 2.0, 4.0)
MC_RANDOM_SEED = None
MC_DEFAULT_MAX_THETA_POINTS = 7
MC_LAPLACE_ALPHA = 0.1


class ValueEstimator(Enum):
    """Estimator used to extrapolate missing-entity values for aggregate corrections."""

    NAIVE = "naive"
    FREQUENCY = "frequency"
    FREQUENCY_F12 = "frequency_f12"
    BUCKET_NAIVE = "bucket_naive"
    BUCKET_FREQUENCY = "bucket_frequency"
    BUCKET_FREQUENCY_F12 = "bucket_frequency_f12"
    MONTE_CARLO = "monte_carlo"


class CountEstimator(Enum):
    """Count estimator used to infer total number of unique entities."""

    CHAO92 = "chao92"
    CHAO84 = "chao84"
    JACKKNIFE1 = "jackknife1"
    JACKKNIFE2 = "jackknife2"
    ACE = "ace"
    UNSEEN = "unseen"


class Estimator:
    """Unknown-unknowns estimators for correcting aggregate query results.

    Accepts samples grouped by source. Each source contributes a list of DataItems;
    the estimator uses the overlap pattern across sources to infer how many entities
    are likely missing from the integrated dataset.

    Usage:
        e = Estimator([source_a, source_b, source_c])
        e.sample_sum()
        e.estimated_sum(ValueEstimator.NAIVE)
        e.estimated_min()
    """

    def __init__(self, source_samples: list[list[DataItem]]) -> None:
        self.source_samples = source_samples
        self._all_samples = [
            item for src in source_samples for item in src if item is not None
        ]

        self._observed_min = min((s.value for s in self._all_samples), default=0.0)
        self._observed_max = max((s.value for s in self._all_samples), default=0.0)

        # name -> [abundance, first_observed_value]
        self._entities: dict[str, list[float]] = {}
        for item in self._all_samples:
            if item.name in self._entities:
                self._entities[item.name][0] += 1.0
            else:
                self._entities[item.name] = [1.0, float(item.value)]

        self._n_unique: int = len(self._entities)
        self._observed_sum: float = sum(v[1] for v in self._entities.values())
        self._n_obs: int = sum(int(v[0]) for v in self._entities.values())

        # Frequency vector: _freq[i] = number of entities seen exactly i+1 times.
        self._freq: list[int] = [0] * max(self._n_obs, 1)
        self._singleton_sum = 0.0
        self._f12_sum = 0.0
        self._c_rare = 0
        self._c_abun = 0
        self._n_rare = 0
        for abundance, value in self._entities.values():
            cnt = int(abundance)
            self._freq[cnt - 1] += 1
            if cnt == 1:
                self._singleton_sum += value
                self._f12_sum += value
            elif cnt == 2:
                self._f12_sum += value

            if cnt <= 10:
                self._c_rare += 1
                self._n_rare += cnt
            else:
                self._c_abun += 1

        self._mc_nb_runs: int = MC_DEFAULT_NB_RUNS
        self._mc_lambda_grid: tuple[float, ...] = MC_DEFAULT_LAMBDA_GRID_MC
        self._mc_random_seed: int | None = MC_RANDOM_SEED
        self._mc_max_theta_points: int = MC_DEFAULT_MAX_THETA_POINTS
        self._mc_laplace_alpha: float = MC_LAPLACE_ALPHA

    def set_monte_carlo_parameters(
        self,
        nb_runs: int = MC_DEFAULT_NB_RUNS,
        lambda_grid: tuple[float, ...] = MC_DEFAULT_LAMBDA_GRID_MC,
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

    # ── Observed (known) aggregates ───────────────────────────────────────────

    def sample_sum(self) -> float:
        """Sum of first-observed values across all unique observed entities."""
        return self._observed_sum

    def sample_count(self) -> int:
        """Number of unique observed entities."""
        return self._n_unique

    def sample_avg(self) -> float:
        """Average value across observed entities."""
        return self._observed_sum / self._n_unique if self._n_unique > 0 else 0.0

    def sample_min(self) -> float:
        """Minimum observed value."""
        return self._observed_min

    def sample_max(self) -> float:
        """Maximum observed value."""
        return self._observed_max

    # ── Estimated (corrected) aggregates ──────────────────────────────────────

    def estimated_count(
        self, count_estimator: CountEstimator = CountEstimator.CHAO92
    ) -> float:
        """Estimated total unique entity count (observed + missing)."""
        return self._count_estimate(count_estimator)

    def chao92(self) -> float:
        """Estimated total unique entity count N̂."""
        if self._n_obs == 0:
            return 0.0
        coverage = self.sample_coverage()
        if coverage <= 0:
            return math.inf
        gamma_sq = self._gamma_sq()
        if not math.isfinite(gamma_sq):
            return math.inf
        n_hat = (self._n_unique / coverage) + (
            self._n_obs * (1.0 - coverage) / coverage
        ) * gamma_sq
        return max(float(self._n_unique), n_hat)

    def chao84(self) -> float:
        """Chao84 count estimator."""
        if self._n_obs == 0:
            return 0.0
        if len(self._freq) < 2 or self._freq[1] == 0:
            return float(self._n_unique)
        return max(
            float(self._n_unique),
            float(self._n_unique)
            + (self._freq[0] * self._freq[0]) / (2.0 * self._freq[1]),
        )

    def jackknife1(self) -> float:
        """First-order jackknife count estimator."""
        if self._n_obs == 0:
            return 0.0
        return float(self._n_unique + self._freq[0])

    def jackknife2(self) -> float:
        """Second-order jackknife count estimator."""
        if self._n_obs == 0:
            return 0.0
        f2 = self._freq[1] if len(self._freq) > 1 else 0
        return float(self._n_unique + 2 * self._freq[0] - f2)

    def ace(self) -> float:
        """Abundance-based coverage estimator (ACE) count estimate."""
        if self._n_obs == 0:
            return 0.0
        if self._n_rare == 0:
            return float(self._n_unique)

        cov_ace = 1.0 - self._freq[0] / self._n_rare
        if cov_ace <= 0:
            return self.chao92()
        if self._n_rare <= 1:
            cv = math.sqrt(self._n_obs)
        else:
            sum_term = 0.0
            for i in range(min(10, len(self._freq))):
                sum_term += i * (i + 1) * self._freq[i]
            cv = max(
                (self._c_rare / cov_ace)
                * (sum_term / (self._n_rare * (self._n_rare - 1)))
                - 1.0,
                0.0,
            )
        est = self._c_abun + self._c_rare / cov_ace + self._freq[0] * cv / cov_ace
        return max(float(self._n_unique), float(est))

    def unseen(self) -> float:
        """Valiant-style unseen count estimate."""
        if self._n_obs == 0:
            return 0.0
        try:
            est = self._unseen_lp()
            return max(float(self._n_unique), float(est))
        except Exception:
            return self.chao92()

    def estimated_count_bounds(
        self, confidence_delta: float = 0.01
    ) -> tuple[float, float]:
        """Lower/upper bounds for total unique entity count.

        Uses a Good-Turing style missing-mass concentration bound and converts it
        to count space via C / (1 - M0).
        """
        if confidence_delta <= 0.0 or confidence_delta > 1.0:
            raise ValueError("confidence_delta must be in (0, 1]")

        if self._n_obs == 0:
            return 0.0, 0.0

        coverage = self.sample_coverage()
        missing_mass = 1.0 - coverage
        error = math.sqrt(math.log(1.0 / confidence_delta) / self._n_obs)

        missing_lower = max(0.0, missing_mass - error)
        # Keep denominator strictly positive to avoid vacuous infinity at small n.
        missing_upper = min(1.0 - 1.0 / (self._n_obs + 1.0), missing_mass + error)

        count_lower = self._n_unique / max(1e-12, 1.0 - missing_lower)
        count_upper = self._n_unique / max(1e-12, 1.0 - missing_upper)
        return max(float(self._n_unique), count_lower), max(
            float(self._n_unique), count_upper
        )

    def estimated_count_lower_bound(self, confidence_delta: float = 0.01) -> float:
        """Lower bound for total unique entity count."""
        lower, _ = self.estimated_count_bounds(confidence_delta)
        return lower

    def estimated_count_upper_bound(self, confidence_delta: float = 0.01) -> float:
        """Upper bound for total unique entity count."""
        _, upper = self.estimated_count_bounds(confidence_delta)
        return upper

    def delta(
        self,
        estimator: ValueEstimator = ValueEstimator.NAIVE,
        count_estimator: CountEstimator = CountEstimator.CHAO92,
    ) -> float:
        """Estimated SUM contribution of missing entities."""
        if estimator is ValueEstimator.NAIVE:
            return self._naive_delta(count_estimator)
        if estimator is ValueEstimator.FREQUENCY:
            return self._frequency_delta(count_estimator=count_estimator)
        if estimator is ValueEstimator.FREQUENCY_F12:
            return self._frequency_f12_delta(count_estimator=count_estimator)
        if estimator is ValueEstimator.BUCKET_NAIVE:
            return self._bucket_delta(
                bucket_value_estimator=ValueEstimator.NAIVE,
                count_estimator=count_estimator,
            )
        if estimator is ValueEstimator.BUCKET_FREQUENCY:
            return self._bucket_delta(
                bucket_value_estimator=ValueEstimator.FREQUENCY,
                count_estimator=count_estimator,
            )
        if estimator is ValueEstimator.BUCKET_FREQUENCY_F12:
            return self._bucket_delta(
                bucket_value_estimator=ValueEstimator.FREQUENCY_F12,
                count_estimator=count_estimator,
            )
        if estimator is ValueEstimator.MONTE_CARLO:
            return self._mc_delta(count_estimator=count_estimator)
        raise ValueError(f"Unsupported estimator: {estimator}")

    def estimated_sum(
        self,
        estimator: ValueEstimator = ValueEstimator.NAIVE,
        count_estimator: CountEstimator = CountEstimator.CHAO92,
    ) -> float:
        """Corrected SUM including estimated contribution from missing entities."""
        d = self.delta(estimator, count_estimator)
        return math.inf if not math.isfinite(d) else self._observed_sum + d

    def estimated_sum_bounds(
        self,
        estimator: ValueEstimator = ValueEstimator.NAIVE,
        count_estimator: CountEstimator = CountEstimator.CHAO92,
        confidence_delta: float = 0.01,
    ) -> tuple[float, float]:
        """Lower/upper bounds for corrected SUM.

        Uses count bounds and a value proxy implied by the selected estimator.
        """
        count_lower, count_upper = self.estimated_count_bounds(confidence_delta)
        missing_lower = max(0.0, count_lower - self._n_unique)
        missing_upper = max(0.0, count_upper - self._n_unique)

        if estimator is ValueEstimator.NAIVE:
            missing_mean = (
                self._observed_sum / self._n_unique if self._n_unique > 0 else 0.0
            )
        elif estimator is ValueEstimator.FREQUENCY:
            f1 = self._f1_count()
            missing_mean = self._singleton_sum / f1 if f1 > 0 else 0.0
        elif estimator is ValueEstimator.FREQUENCY_F12:
            denom = self._f1_count() + (self._freq[1] if len(self._freq) > 1 else 0)
            missing_mean = self._f12_sum / denom if denom > 0 else 0.0
        elif estimator is ValueEstimator.BUCKET_NAIVE:
            est_count = self._count_estimate(count_estimator)
            missing_mean = max(
                0.0,
                self._bucket_delta(
                    bucket_value_estimator=ValueEstimator.NAIVE,
                    count_estimator=count_estimator,
                ),
            ) / max(1e-12, est_count - self._n_unique)
        elif estimator is ValueEstimator.BUCKET_FREQUENCY:
            est_count = self._count_estimate(count_estimator)
            missing_mean = max(
                0.0,
                self._bucket_delta(
                    bucket_value_estimator=ValueEstimator.FREQUENCY,
                    count_estimator=count_estimator,
                ),
            ) / max(1e-12, est_count - self._n_unique)
        elif estimator is ValueEstimator.BUCKET_FREQUENCY_F12:
            est_count = self._count_estimate(count_estimator)
            missing_mean = max(
                0.0,
                self._bucket_delta(
                    bucket_value_estimator=ValueEstimator.FREQUENCY_F12,
                    count_estimator=count_estimator,
                ),
            ) / max(1e-12, est_count - self._n_unique)
        elif estimator is ValueEstimator.MONTE_CARLO:
            missing_mean = (
                self._observed_sum / self._n_unique if self._n_unique > 0 else 0.0
            )
        else:
            raise ValueError(f"Unsupported estimator: {estimator}")

        lower = self._observed_sum + missing_mean * missing_lower
        upper = self._observed_sum + missing_mean * missing_upper
        return lower, upper

    def estimated_sum_lower_bound(
        self,
        estimator: ValueEstimator = ValueEstimator.NAIVE,
        count_estimator: CountEstimator = CountEstimator.CHAO92,
        confidence_delta: float = 0.01,
    ) -> float:
        """Lower bound for corrected SUM."""
        lower, _ = self.estimated_sum_bounds(
            estimator, count_estimator, confidence_delta
        )
        return lower

    def estimated_sum_upper_bound(
        self,
        estimator: ValueEstimator = ValueEstimator.NAIVE,
        count_estimator: CountEstimator = CountEstimator.CHAO92,
        confidence_delta: float = 0.01,
    ) -> float:
        """Upper bound for corrected SUM."""
        _, upper = self.estimated_sum_bounds(
            estimator, count_estimator, confidence_delta
        )
        return upper

    def estimated_avg(
        self,
        estimator: ValueEstimator = ValueEstimator.NAIVE,
        count_estimator: CountEstimator = CountEstimator.CHAO92,
    ) -> float:
        """Estimated average across all entities (observed + missing)."""
        est_count = self._count_estimate(count_estimator)
        if not math.isfinite(est_count) or est_count == 0:
            return self.sample_avg()
        return self.estimated_sum(estimator, count_estimator) / est_count

    def estimated_min(self) -> float:
        """Estimated minimum, extended downward when unknown entities likely exist."""
        if self.chao92() <= self._n_unique:
            return self._observed_min
        return min(self._observed_min, self._sample_mean() - 2.0 * self._sample_std())

    def estimated_max(self) -> float:
        """Estimated maximum, extended upward when unknown entities likely exist."""
        if self.chao92() <= self._n_unique:
            return self._observed_max
        return max(self._observed_max, self._sample_mean() + 2.0 * self._sample_std())

    def sample_coverage(self) -> float:
        """Good-Turing sample coverage (0..1). Higher means the sample is more complete."""
        if self._n_obs == 0:
            return 0.0
        return 1.0 - self._f1_count() / self._n_obs

    # ── Private: core statistics ──────────────────────────────────────────────

    def _f1_count(self) -> int:
        return self._freq[0] if self._freq else 0

    def _sample_mean(self) -> float:
        if self._n_obs == 0:
            return 0.0
        return sum(item.value for item in self._all_samples) / self._n_obs

    def _sample_std(self) -> float:
        if self._n_obs == 0:
            return 0.0
        mean = self._sample_mean()
        sq_mean = sum(item.value**2 for item in self._all_samples) / self._n_obs
        return math.sqrt(max(0.0, sq_mean - mean * mean))

    def _gamma_sq(self) -> float:
        """Estimated CV² via f-statistics."""
        if self._n_obs <= 1 or self._n_unique == 0:
            return 0.0
        coverage = self.sample_coverage()
        if coverage <= 0:
            return math.inf
        moment = sum(
            (idx + 1) * idx * f for idx, f in enumerate(self._freq[: self._n_unique])
        )
        gamma_sq = (self._n_unique / coverage) * (
            moment / (self._n_obs * (self._n_obs - 1))
        ) - 1.0
        return max(gamma_sq, 0.0)

    def coeff_var(self) -> float:
        """Estimated coefficient of variation (sqrt of gamma^2)."""
        return math.sqrt(max(0.0, self._gamma_sq()))

    def _count_estimate(self, estimator: CountEstimator) -> float:
        if estimator is CountEstimator.CHAO92:
            return self.chao92()
        if estimator is CountEstimator.CHAO84:
            return self.chao84()
        if estimator is CountEstimator.JACKKNIFE1:
            return self.jackknife1()
        if estimator is CountEstimator.JACKKNIFE2:
            return self.jackknife2()
        if estimator is CountEstimator.ACE:
            return self.ace()
        if estimator is CountEstimator.UNSEEN:
            return self.unseen()
        raise ValueError(f"Unsupported count estimator: {estimator}")

    def _naive_delta(
        self, count_estimator: CountEstimator = CountEstimator.CHAO92
    ) -> float:
        """Estimated SUM of missing entities using mean substitution."""
        if self._n_unique == 0:
            return 0.0
        n_hat = self._count_estimate(count_estimator)
        if not math.isfinite(n_hat):
            return math.inf
        return (self._observed_sum / self._n_unique) * max(n_hat - self._n_unique, 0.0)

    def _frequency_delta(
        self,
        simplified: bool = False,
        count_estimator: CountEstimator = CountEstimator.CHAO92,
    ) -> float:
        """Estimated SUM of missing entities using singleton values."""
        f1 = self._f1_count()
        if self._n_unique == 0 or f1 == 0:
            return 0.0
        if simplified:
            denom = self._n_obs - f1
            return self._singleton_sum * self._n_unique / denom if denom > 0 else 0.0
        n_hat = self._count_estimate(count_estimator)
        if not math.isfinite(n_hat):
            return math.inf
        return (self._singleton_sum / f1) * max(n_hat - self._n_unique, 0.0)

    def _frequency_f12_delta(
        self, count_estimator: CountEstimator = CountEstimator.CHAO92
    ) -> float:
        f2 = self._freq[1] if len(self._freq) > 1 else 0
        denom = self._f1_count() + f2
        if self._n_unique == 0 or denom == 0:
            return 0.0
        n_hat = self._count_estimate(count_estimator)
        if not math.isfinite(n_hat):
            return math.inf
        return (self._f12_sum / denom) * max(n_hat - self._n_unique, 0.0)

    # ── Private: bucket estimator ─────────────────────────────────────────────

    def _bucket_delta(
        self,
        bucket_value_estimator: ValueEstimator,
        count_estimator: CountEstimator,
    ) -> float:
        return sum(
            Estimator([b.samples])._frequency_f12_delta(count_estimator)
            if bucket_value_estimator is ValueEstimator.FREQUENCY_F12
            else Estimator([b.samples])._frequency_delta(
                count_estimator=count_estimator
            )
            if bucket_value_estimator is ValueEstimator.FREQUENCY
            else Estimator([b.samples])._naive_delta(count_estimator)
            for b in self._dynamic_buckets(bucket_value_estimator)
        )

    def _dynamic_buckets(self, bucket_value_estimator: ValueEstimator) -> list[Bucket]:
        """Splits the value range into adaptive buckets to minimise total estimated delta."""
        if not self._all_samples:
            return []

        init_bucket = Bucket(self._observed_min, self._observed_max)
        for s in sorted(self._all_samples, key=lambda x: x.value):
            init_bucket.add_sample(s)

        buckets: list[Bucket] = [init_bucket]
        previous_count = 0

        while len(buckets) > previous_count:
            previous_count = len(buckets)
            next_buckets: list[Bucket] = []

            for bucket in buckets:
                if bucket.number_of_samples == 0:
                    continue

                if (
                    bucket.f1_count < bucket.number_of_samples
                    and not bucket.split_locked
                ):
                    samples_sorted = bucket.sorted_samples()
                    lb, ub = bucket.lower_bound, bucket.upper_bound

                    best_left, best_right = self._split_bucket(bucket, (lb + ub) / 2.0)
                    best_score = self._bucket_sum_est(
                        best_left, bucket_value_estimator
                    ) + self._bucket_sum_est(best_right, bucket_value_estimator)

                    for item in samples_sorted:
                        left, right = self._split_bucket(bucket, item.value)
                        score = self._bucket_sum_est(
                            left, bucket_value_estimator
                        ) + self._bucket_sum_est(right, bucket_value_estimator)
                        if score < best_score:
                            best_left, best_right, best_score = left, right, score

                    if best_score < self._bucket_sum_est(
                        bucket, bucket_value_estimator
                    ):
                        next_buckets.extend([best_left, best_right])
                    else:
                        bucket.split_locked = True
                        next_buckets.append(bucket)
                else:
                    next_buckets.append(bucket)

            buckets = [b for b in next_buckets if b.number_of_samples > 0]

        return sorted(buckets, key=lambda x: x.lower_bound)

    def _bucket_sum_est(
        self, bucket: Bucket, bucket_value_estimator: ValueEstimator
    ) -> float:
        est = Estimator([bucket.samples])
        if bucket_value_estimator is ValueEstimator.FREQUENCY:
            delta = est._frequency_delta(count_estimator=CountEstimator.CHAO92)
        elif bucket_value_estimator is ValueEstimator.FREQUENCY_F12:
            delta = est._frequency_f12_delta(count_estimator=CountEstimator.CHAO92)
        elif bucket_value_estimator is ValueEstimator.NAIVE:
            delta = est._naive_delta(CountEstimator.CHAO92)
        else:
            raise ValueError(
                f"Unsupported bucket value estimator: {bucket_value_estimator}"
            )
        return est._observed_sum + delta

    def _unseen_lp(self) -> float:
        grid_factor = 1.05
        alpha = 0.5

        xlp_min = 1.0 / (self._n_obs * max(10, self._n_obs))
        i_min = next((i for i, v in enumerate(self._freq) if v > 0), 0)
        if i_min > 0:
            xlp_min = (i_min + 1.0) / self._n_obs

        sparse_species = 0.0
        sparse_mass = 0.0
        dense_indices: list[int] = []
        for i, fi in enumerate(self._freq):
            if fi <= 0:
                continue
            radius = int(math.ceil(math.sqrt(i)))
            lo = max(0, i - radius)
            hi = min(len(self._freq) - 1, i + radius)
            local_sum = sum(self._freq[j] for j in range(lo, hi + 1))
            if local_sum < math.sqrt(max(i, 1)):
                sparse_species += fi
                sparse_mass += ((i + 1.0) / self._n_obs) * fi
            else:
                dense_indices.append(i)

        if not dense_indices:
            return max(float(self._n_unique), sparse_species)

        lp_mass = max(1e-12, 1.0 - sparse_mass)
        f_max = max(dense_indices)
        dense_map = {i: float(self._freq[i]) for i in dense_indices}
        f_lp = [dense_map.get(i, 0.0) for i in range(f_max + 1)]
        f_lp.extend([0.0] * int(math.ceil(math.sqrt(f_max + 1))))

        xlp_max = (f_max + 1.0) / self._n_obs
        if xlp_max <= xlp_min:
            return self.chao92()

        grid_size = (
            int(math.ceil(math.log(xlp_max / xlp_min) / math.log(grid_factor))) + 1
        )
        xlp = [xlp_min * (grid_factor**i) for i in range(max(grid_size, 2))]

        m = len(xlp)
        r = len(f_lp)
        bmat = np.zeros((r, m), dtype=float)
        for i in range(r):
            k = i + 1
            for j, x in enumerate(xlp):
                bmat[i, j] = poisson.pmf(k, self._n_obs * x) / x

        # vars = [y_1..y_m, s_1..s_r], where y_j = x_j * h_j and |By-f| <= s.
        cvec = np.array(
            [alpha / x for x in xlp] + [1.0 / math.sqrt(v + 1.0) for v in f_lp],
            dtype=float,
        )

        a_ub = np.zeros((2 * r, m + r), dtype=float)
        b_ub = np.zeros(2 * r, dtype=float)
        for i in range(r):
            a_ub[i, :m] = bmat[i, :]
            a_ub[i, m + i] = -1.0
            b_ub[i] = f_lp[i]

            a_ub[r + i, :m] = -bmat[i, :]
            a_ub[r + i, m + i] = -1.0
            b_ub[r + i] = -f_lp[i]

        a_eq = np.zeros((1, m + r), dtype=float)
        a_eq[0, :m] = 1.0
        b_eq = np.array([lp_mass], dtype=float)

        bounds = [(0.0, None)] * (m + r)
        res = linprog(
            c=cvec,
            A_ub=a_ub,
            b_ub=b_ub,
            A_eq=a_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )
        if not res.success:
            return self.chao92()

        y = res.x[:m]
        lp_species = float(sum(y_i / x_i for y_i, x_i in zip(y, xlp)))
        return sparse_species + lp_species

    @staticmethod
    def _split_bucket(bucket: Bucket, split_value: float) -> tuple[Bucket, Bucket]:
        left = Bucket(bucket.lower_bound, split_value)
        right = Bucket(split_value, bucket.upper_bound)
        for s in bucket.samples:
            if bucket.lower_bound <= s.value <= split_value:
                left.add_sample(s)
            else:
                right.add_sample(s)
        return left, right

    # ── Private: Monte Carlo estimator ───────────────────────────────────────

    def _mc_delta(
        self,
        count_estimator: CountEstimator = CountEstimator.CHAO92,
        use_frequency: bool = False,
    ) -> float:
        missing = self._mc_count(count_estimator=count_estimator) - self._n_unique
        if missing <= 0:
            return 0.0
        if use_frequency:
            f1 = self._f1_count()
            return (self._singleton_sum / f1 * missing) if f1 > 0 else 0.0
        return (
            (self._observed_sum / self._n_unique * missing)
            if self._n_unique > 0
            else 0.0
        )

    def _mc_count(
        self,
        count_estimator: CountEstimator = CountEstimator.CHAO92,
    ) -> int:
        if self._n_obs == 0:
            return 0
        n_estimated = self._count_estimate(count_estimator)
        n_ub = (
            self._n_unique
            if not math.isfinite(n_estimated)
            else max(self._n_unique, int(math.ceil(n_estimated)))
        )

        # If Chao is not finite or collapses to the observed count, there is no
        # meaningful N-grid search to perform for Monte Carlo.
        if n_ub <= self._n_unique:
            return self._n_unique

        jump = max(1, math.ceil((n_ub - self._n_unique) / self._mc_max_theta_points))
        i_last = 1 if jump == 0 else (n_ub - self._n_unique + 1) // jump
        n_grid = [self._n_unique + i * jump for i in range(max(i_last, 1))]
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
        best_n, _ = self._rbf_search_min(
            points, self._n_unique, n_ub, self._mc_lambda_grid
        )
        return int(best_n)

    def _mc_distance(
        self,
        theta_n: int,
        theta_lambda: float,
        n_w: Sequence[int],
        nb_runs: int,
    ) -> float:
        if theta_n <= 0:
            return 0.0
        rng = random.Random(self._mc_random_seed)
        total = 0.0
        n_size = sum(n_w)
        p_dist = self._index_by_frequency(self._entities)

        for _ in range(nb_runs):
            simulated: dict[int, list[float]] = {}
            for worker_n in n_w:
                local: set[int] = set()
                target = min(worker_n, theta_n)
                while len(local) < target:
                    idx = int(math.floor(rng.random() * (theta_n + 1)))
                    p = (
                        1.0 - math.exp(idx * theta_lambda / theta_n)
                        if theta_lambda < 0
                        else math.exp(-idx * theta_lambda / theta_n)
                    )
                    if rng.random() <= p:
                        local.add(idx)
                for idx in local:
                    if idx in simulated:
                        simulated[idx][0] += 1.0
                    else:
                        simulated[idx] = [1.0, 0.0]

            keys = set(p_dist.keys()) | set(simulated.keys())
            if not keys:
                continue

            alpha = self._mc_laplace_alpha
            p_denom = len(self._all_samples) + alpha * len(keys)
            q_denom = n_size + alpha * len(keys)
            kl = 0.0
            for k in keys:
                p = (p_dist[k][0] + alpha) / p_denom if k in p_dist else alpha / p_denom
                q = (
                    (simulated[k][0] + alpha) / q_denom
                    if k in simulated
                    else alpha / q_denom
                )
                kl += p * math.log(p / q)
            total += kl

        return total / max(nb_runs, 1)

    @staticmethod
    def _index_by_frequency(entities: dict[str, list[float]]) -> dict[int, list[float]]:
        """Index entities by abundance rank (most frequent = index 1)."""
        sorted_entities = sorted(entities.values(), key=lambda x: x[0], reverse=True)
        return {i + 1: v for i, v in enumerate(sorted_entities)}

    @staticmethod
    def _rbf_search_min(
        points: Sequence[tuple[float, float, float]],
        n_lb: int,
        n_ub: int,
        lambda_grid: Sequence[float],
    ) -> tuple[int, float]:
        """Fit a smooth RBF surface and return the (N, lambda) pair with minimum KL."""
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

"""Adaptive bucket splitting for the bucket-based estimator.

Implements the greedy split loop that partitions the value range into buckets
so each bucket is roughly homogeneous in value, improving the accuracy of the
per-bucket count/value estimates. The NAIVE value estimator uses an O(n)
incremental sweep (numba-accelerated for CHAO92).
"""

import bisect
import math
from typing import assert_never

import numba
import numpy as np

from qcp_adaptive_termination.estimator.bucket import Bucket
from qcp_adaptive_termination.estimator.count_estimators import chao92_scalar
from qcp_adaptive_termination.estimator.data_item import DataItem
from qcp_adaptive_termination.estimator.frequency import FrequencyStats
from qcp_adaptive_termination.models.estimation import CountEstimator, ValueEstimator

# ── Resplit policy constants ──────────────────────────────────────────────────
# A bucket is re-swept when it has grown by at least _RESPLIT_FRACTION of its
# size at the previous sweep AND at least _RESPLIT_MIN_DELTA new samples.
# split_backoff on Bucket doubles (up to _RESPLIT_BACKOFF_CAP) after each
# sweep that finds no split, exponentially throttling unsplittable buckets.
_RESPLIT_MIN_DELTA: int = 100
_RESPLIT_FRACTION: float = 0.50
_RESPLIT_BACKOFF_CAP: int = 16


@numba.njit(cache=True)  # type: ignore[misc]
def _sweep_chao92(
    entity_ids: np.ndarray,
    values: np.ndarray,
    entity_position_data: np.ndarray,
    entity_position_start: np.ndarray,
    entity_observation_counts: np.ndarray,
    parent_score: float,
) -> int:
    """
    O(n) sweep to find the best split index for CHAO92/NAIVE scoring.

    Arrays are precomputed by _prepare_sweep_arrays and stored in sorted-samples order.
    entity_ids[k]                    — integer entity id for samples[k]
    values[k]                        — samples[k].value
    entity_position_data/start       — CSR: entity j's positions = data[start[j]:start[j+1]]
    entity_observation_counts[j]     — total occurrences of entity j in the bucket

    Returns the index k of the last sample to assign to the LEFT partition (split after k),
    or -1 if no split improves parent_score.
    """
    n = len(values)
    c = len(entity_observation_counts)

    # Right side starts with all n samples
    right_unique_count = c
    right_value_sum = 0.0
    right_singletons = 0
    right_doubletons = 0
    right_pair_moment = 0
    right_counts = entity_observation_counts.copy()
    right_pointers = np.zeros(c, dtype=np.int64)

    for j in range(c):
        right_value_sum += values[entity_position_data[entity_position_start[j]]]
        k = entity_observation_counts[j]
        right_pair_moment += k * (k - 1)
        if k == 1:
            right_singletons += 1
        elif k == 2:
            right_doubletons += 1

    left_unique_count = 0
    left_value_sum = 0.0
    left_singletons = 0
    left_doubletons = 0
    left_pair_moment = 0
    left_counts = np.zeros(c, dtype=np.int64)

    best_score = parent_score
    best_idx = -1

    for k in range(n - 1):
        j = entity_ids[k]
        v = values[k]

        old_right_count = right_counts[j]
        new_right_count = old_right_count - 1
        right_counts[j] = new_right_count

        right_value_sum -= v
        right_pair_moment -= 2 * (old_right_count - 1)
        if old_right_count == 1:
            right_singletons -= 1
        elif old_right_count == 2:
            right_doubletons -= 1

        if new_right_count > 0:
            if new_right_count == 1:
                right_singletons += 1
            elif new_right_count == 2:
                right_doubletons += 1
            right_pointers[j] += 1
            right_value_sum += values[entity_position_data[entity_position_start[j] + right_pointers[j]]]
        else:
            right_unique_count -= 1

        old_left_count = left_counts[j]
        new_left_count = old_left_count + 1
        left_counts[j] = new_left_count
        left_pair_moment += 2 * old_left_count

        if old_left_count == 0:
            left_unique_count += 1
            left_value_sum += v
            left_singletons += 1
        elif old_left_count == 1:
            left_singletons -= 1
            left_doubletons += 1
        elif old_left_count == 2:
            left_doubletons -= 1

        left_obs_count = k + 1
        right_obs_count = n - left_obs_count
        left_score = _chao92_naive_score(
            left_unique_count, left_obs_count, left_value_sum, left_singletons, int(left_pair_moment)
        )
        right_score = _chao92_naive_score(
            right_unique_count, right_obs_count, right_value_sum, right_singletons, int(right_pair_moment)
        )
        score = left_score + right_score
        if score < best_score:
            best_score = score
            best_idx = k

    return best_idx


@numba.njit(cache=True)
def _chao92_naive_score(
    unique_count: int,
    observation_count: int,
    representative_value_sum: float,
    singleton_count: int,
    pair_moment: int,
) -> float:  # type: ignore[return]
    """
    NAIVE estimated-sum score for a partition using the Chao92 count estimator.

    representative_value_sum — sum of representative (first-occurrence) values for the entities.
    pair_moment — sum of count*(count-1) across all entities in the partition.
    Returns inf when sample coverage is zero (all entities are singletons).
    """
    if unique_count == 0 or observation_count == 0:
        return 0.0
    estimated_count = chao92_scalar(unique_count, observation_count, singleton_count, pair_moment)
    return representative_value_sum * estimated_count / unique_count


# Pre-compile at import time to avoid first-call JIT latency (~1–3 s).
_w_eid = np.zeros(2, np.int64)
_w_vals = np.array([0.0, 1.0])
_w_ep_data = np.array([0, 1], np.int64)
_w_ep_start = np.array([0, 1, 2], np.int64)
_w_counts = np.array([1, 1], np.int64)
_sweep_chao92(_w_eid, _w_vals, _w_ep_data, _w_ep_start, _w_counts, 0.0)


def _prepare_sweep_arrays(
    samples: list[DataItem],
    entity_positions: dict[str, list[int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build numba-compatible arrays from sorted samples and per-entity position lists.

    Returns (entity_ids, values, position_data, position_start, observation_counts) where:
      entity_ids[k]                    — integer entity id for samples[k]
      values[k]                        — samples[k].value
      position_data / position_start   — CSR: entity j's positions = data[start[j]:start[j+1]]
      observation_counts[j]            — total occurrences of entity j
    """
    entity_list = list(entity_positions.keys())
    entity_count = len(entity_list)
    sample_count = len(samples)

    counts = np.array([len(entity_positions[e]) for e in entity_list], dtype=np.int64)
    position_start = np.empty(entity_count + 1, dtype=np.int64)
    position_start[0] = 0
    np.cumsum(counts, out=position_start[1:])

    position_data = np.empty(sample_count, dtype=np.int64)
    for j, e in enumerate(entity_list):
        position_data[position_start[j] : position_start[j + 1]] = entity_positions[e]

    entity_ids = np.empty(sample_count, dtype=np.int64)
    for j in range(entity_count):
        entity_ids[position_data[position_start[j] : position_start[j + 1]]] = j

    values = np.array([s.value for s in samples], dtype=np.float64)
    return entity_ids, values, position_data, position_start, counts


def bucket_sum_est(
    bucket: Bucket,
    bucket_value_estimator: ValueEstimator,
    bucket_count_estimator: CountEstimator,
) -> float:
    """
    Estimated SUM for a single bucket, used as the splitting score.

    Computed directly from Bucket.histogram and Bucket.f — no Estimator object
    created — so the split loop avoids thousands of pandas groupby calls.
    Only the CHAO92 count estimator is supported for bucket splitting.
    """
    if bucket_count_estimator is not CountEstimator.CHAO92:
        raise ValueError(f"Bucket splitting only supports CHAO92, got {bucket_count_estimator}")

    unique_count = bucket.number_of_unique_samples
    observation_count = bucket.number_of_samples
    if unique_count == 0 or observation_count == 0:
        return 0.0

    freq = bucket.f
    singleton_count = freq[0] if freq else 0
    doubleton_count = freq[1] if len(freq) > 1 else 0
    histogram = bucket.histogram

    representative_value_sum = sum(bar.lower_bound for bar in histogram.values())
    pair_moment = sum((idx + 1) * idx * fi for idx, fi in enumerate(freq) if fi)
    estimated_count = chao92_scalar(unique_count, observation_count, singleton_count, pair_moment)

    if not math.isfinite(estimated_count):
        return math.inf

    if bucket_value_estimator is ValueEstimator.NAIVE:
        delta = (representative_value_sum / unique_count) * (estimated_count - unique_count)
    elif bucket_value_estimator is ValueEstimator.FREQUENCY:
        singleton_value_sum = sum(bar.lower_bound for bar in histogram.values() if bar.count == 1)
        delta = (singleton_value_sum / singleton_count) * (estimated_count - unique_count) if singleton_count else 0.0
    elif bucket_value_estimator is ValueEstimator.GOOD_TURING:
        singleton_value_sum = sum(bar.lower_bound for bar in histogram.values() if bar.count == 1)
        denom = observation_count - singleton_count
        delta = singleton_value_sum * unique_count / denom if denom > 0 else 0.0
    elif bucket_value_estimator is ValueEstimator.FREQUENCY_F12:
        f12_value_sum = sum(bar.lower_bound for bar in histogram.values() if bar.count <= 2)
        denom = singleton_count + doubleton_count
        delta = (f12_value_sum / denom) * (estimated_count - unique_count) if denom else 0.0
    else:
        assert_never(bucket_value_estimator)

    return representative_value_sum + delta


def split_bucket(bucket: Bucket, split_value: float) -> tuple[Bucket, Bucket]:
    """
    Split a bucket at split_value into left (≤ split_value) and right (> split_value).

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


def find_best_split(
    bucket: Bucket,
    value_estimator: ValueEstimator,
    count_estimator: CountEstimator,
    parent_score: float,
) -> float | None:
    """
    Find best split value via O(n) incremental sweep for NAIVE value estimator.

    Delegates to the numba-compiled _sweep_chao92 for maximum throughput.
    Returns the split value that minimises left+right score, or None if no improvement.
    """
    if value_estimator is not ValueEstimator.NAIVE:
        return None
    if count_estimator is not CountEstimator.CHAO92:
        raise ValueError(f"Bucket splitting only supports CHAO92, got {count_estimator}")

    samples = bucket.samples
    sample_count = len(samples)
    if sample_count < 2:
        return None

    entity_positions: dict[str, list[int]] = bucket.entity_positions or {}
    if not entity_positions:
        for i, s in enumerate(samples):
            entity_positions.setdefault(s.name, []).append(i)

    entity_ids, values, pos_data, pos_start, obs_counts = _prepare_sweep_arrays(samples, entity_positions)
    best_idx = _sweep_chao92(entity_ids, values, pos_data, pos_start, obs_counts, parent_score)
    return samples[best_idx].value if best_idx >= 0 else None


def run_split_loop(
    buckets: list[Bucket],
    bucket_value_estimator: ValueEstimator,
    bucket_count_estimator: CountEstimator,
    *,
    max_buckets: int,
) -> list[Bucket]:
    """
    Run the greedy split loop until no bucket can be improved.

    Uses the O(n) sweep per bucket — no temporary Bucket copies.
    """
    previous_count = 0

    while len(buckets) > previous_count:
        previous_count = len(buckets)
        if len(buckets) >= max_buckets:
            # Bucket cap reached — stop splitting entirely.
            break
        next_buckets: list[Bucket] = []

        for bucket in buckets:
            if len(next_buckets) >= max_buckets:
                # Cap reached mid-pass — keep this bucket unsplit and lock it so
                # it is not re-swept on subsequent outer-loop passes.
                bucket.split_locked = True
                next_buckets.append(bucket)
                continue

            if bucket.split_locked:
                next_buckets.append(bucket)
                continue

            if bucket.singleton_count() >= bucket.number_of_samples:
                next_buckets.append(bucket)
                continue

            parent_score = bucket_sum_est(bucket, bucket_value_estimator, bucket_count_estimator)
            best_split_value = find_best_split(bucket, bucket_value_estimator, bucket_count_estimator, parent_score)

            if best_split_value is not None:
                # Sweep found an improvement.
                best_left, best_right = split_bucket(bucket, best_split_value)
                next_buckets.extend([best_left, best_right])
                continue

            # No improvement — back off exponentially.
            bucket.split_locked = True
            bucket.split_backoff = min(bucket.split_backoff * 2, _RESPLIT_BACKOFF_CAP)
            next_buckets.append(bucket)

        buckets = next_buckets

    return sorted(buckets, key=lambda x: x.lower_bound)


def warm_split(
    buckets: list[Bucket],
    new_samples: list[DataItem],
    bucket_value_estimator: ValueEstimator,
    bucket_count_estimator: CountEstimator,
    *,
    max_buckets: int,
) -> tuple[list[Bucket], dict[int, dict[str, list[DataItem]]]]:
    """
    Insert new samples into existing buckets and re-split only the affected ones.

    Returns (new_buckets, new_items_per_surviving_bucket) where the dict maps
    id(bucket) → {source_id → [DataItem]} for buckets that survived the split loop.
    Split children are excluded; they must be built from scratch via to_sources().
    """
    lower_bounds = [b.lower_bound for b in buckets]
    new_items_per_bucket: dict[int, dict[str, list[DataItem]]] = {}

    for item in new_samples:
        idx = max(0, bisect.bisect_right(lower_bounds, item.value) - 1)
        buckets[idx].add_sample(item)
        buckets[idx].split_locked = False
        bucket_id = id(buckets[idx])
        by_source = new_items_per_bucket.setdefault(bucket_id, {})
        by_source.setdefault(item.source_id, []).append(item)

    for bucket in buckets:
        if not bucket.split_locked:
            current_sample_count = len(bucket.samples)
            new_delta = current_sample_count - bucket.samples_at_last_resplit
            if bucket.samples_at_last_resplit > 0 and new_delta < max(
                _RESPLIT_MIN_DELTA,
                int(bucket.samples_at_last_resplit * _RESPLIT_FRACTION * bucket.split_backoff),
            ):
                # Too few new samples to justify a full sort+sweep; keep locked.
                bucket.split_locked = True
            else:
                bucket.samples.sort(key=lambda x: x.value)
                bucket.rebuild_frequency_vector()
                bucket.samples_at_last_resplit = current_sample_count
                entity_positions: dict[str, list[int]] = {}
                for i, s in enumerate(bucket.samples):
                    entity_positions.setdefault(s.name, []).append(i)
                bucket.entity_positions = entity_positions

    result_buckets = run_split_loop(
        buckets,
        bucket_value_estimator,
        bucket_count_estimator,
        max_buckets=max_buckets,
    )
    # Set samples_at_last_resplit on any child buckets produced by run_split_loop.
    # Without this, children (with default=0) would always pass the threshold check
    # and be re-swept on every subsequent warm_split call.
    for bucket in result_buckets:
        if bucket.samples_at_last_resplit == 0:
            bucket.samples_at_last_resplit = len(bucket.samples)
    surviving_ids = {id(b) for b in result_buckets}
    surviving_new_items = {bid: by_source for bid, by_source in new_items_per_bucket.items() if bid in surviving_ids}
    return result_buckets, surviving_new_items


def raw_dynamic_buckets(
    stats: FrequencyStats,
    bucket_value_estimator: ValueEstimator,
    bucket_count_estimator: CountEstimator,
    *,
    max_buckets: int,
) -> list[Bucket]:
    """Build adaptive buckets from scratch over all current samples."""
    init_bucket = Bucket(stats.observed_min, stats.observed_max)
    for s in sorted(stats.all_samples, key=lambda x: x.value):
        init_bucket.add_sample(s)
    return run_split_loop(
        [init_bucket],
        bucket_value_estimator,
        bucket_count_estimator,
        max_buckets=max_buckets,
    )

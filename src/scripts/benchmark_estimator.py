"""
Performance benchmark: 1000 agents × 1000 rows, full incremental stat loop.

Data distribution is intentionally realistic:
- Large entity pool (100K) so many entities remain unseen → N_hat >> c in sparse tiers.
- Zipfian popularity: low-value tiers (tier 0) are ~100× more frequently observed
  than high-value tiers (tier 9).  High-value entities are true unknown unknowns.
  This makes bucket splitting profitable and exercises the full estimator pipeline.
"""

import logging
import random
import time

from qcp_adaptive_termination.estimator import CountEstimator, DataItem, Estimator, ValueEstimator

AGENTS = 1000
ROWS_PER_AGENT = 1000
ENTITY_POOL_SIZE = 100_000  # large pool → genuine missing entities in high-value tiers
RANDOM_SEED = 42
VALUE_TIERS = 10
# Pareto exponent for Zipf-like entity selection.
# P(entity with index i) ∝ 1/(i+1)^(ZIPF_ALPHA+1).
# ZIPF_ALPHA=0.5 → tier-0 entities (~100× more popular than tier-9).
ZIPF_ALPHA = 0.5

COUNT_EST = CountEstimator.CHAO92
VALUE_EST = ValueEstimator.NAIVE
USE_MC = False


def _generate_source(agent_idx: int, entity_pool: list[str], rng: random.Random) -> list[DataItem]:
    items = []
    tier_size = ENTITY_POOL_SIZE // VALUE_TIERS
    for _ in range(ROWS_PER_AGENT):
        # Zipf-like sampling: low-indexed (low-value) entities are far more popular.
        # paretovariate(α) returns X ≥ 1 with P(X > x) = x^-α.
        rank = min(int(rng.paretovariate(ZIPF_ALPHA)), ENTITY_POOL_SIZE)
        entity_idx = rank - 1
        name = entity_pool[entity_idx]
        tier = entity_idx // tier_size
        value = max(0.01, rng.lognormvariate(2.0 + tier * 0.6, 0.4))
        items.append(DataItem(name=name, value=value, source_id=f"agent_{agent_idx}"))
    return items


def _compute_all_stats(estimator: Estimator) -> None:
    """Call every public stat to ensure caches are exercised."""
    estimator.sample_count()
    estimator.sample_sum()
    estimator.sample_avg()
    estimator.sample_min()
    estimator.sample_max()
    for q in (0.25, 0.5, 0.75):
        estimator.sample_quantile(q)
    estimator.estimated_count()
    estimator.estimated_sum()
    estimator.estimated_avg()
    estimator.estimated_min()
    estimator.estimated_max()
    for q in (0.25, 0.5, 0.75):
        estimator.estimated_quantile(q)


def _generate_data(rng: random.Random) -> list[list[DataItem]]:
    logging.info(
        "Generating %d agents × %d rows (pool=%d, zipf_alpha=%.1f)...",
        AGENTS,
        ROWS_PER_AGENT,
        ENTITY_POOL_SIZE,
        ZIPF_ALPHA,
    )
    entity_pool = [f"entity_{i}" for i in range(ENTITY_POOL_SIZE)]
    return [_generate_source(i, entity_pool, rng) for i in range(AGENTS)]


def _run_forced_split(sources: list[list[DataItem]]) -> None:
    """
    Build estimator from ALL sources at once and report bucket structure.

    Bypasses warm-split throttling: shows the maximum bucket granularity
    the split algorithm would produce given the complete dataset.
    """
    logging.info("=== Forced split (all %d sources) ===", len(sources))
    t0 = time.perf_counter()
    est = Estimator(sources)
    build_t = time.perf_counter() - t0

    t1 = time.perf_counter()
    buckets, subs = est._get_sub_estimators()  # type: ignore[reportPrivateUsage]
    weights = est._bucket_weights(subs)  # type: ignore[reportPrivateUsage]
    split_t = time.perf_counter() - t1

    n_hat_global = est.estimated_count()
    logging.info("Build %.2fs | Split %.2fs | Buckets: %d", build_t, split_t, len(buckets))
    logging.info(
        "Global: c=%d  N_hat=%.0f  ratio=%.2f×",
        est.sample_count(),
        n_hat_global,
        n_hat_global / max(est.sample_count(), 1),
    )
    for i, (b, w) in enumerate(zip(buckets, weights, strict=True)):
        lb = b.lower_bound if b.lower_bound != float("-inf") else 0.0
        ub = b.upper_bound if b.upper_bound != float("inf") else float("inf")
        n_hat_ratio = w / max(b.number_of_unique_samples, 1)
        logging.info(
            "  bucket %2d [%7.1f, %7.1f): n=%6d  c=%5d  N_hat=%.0f (%.2f×)",
            i,
            lb,
            ub,
            b.number_of_samples,
            b.number_of_unique_samples,
            w,
            n_hat_ratio,
        )


def _run_benchmark(sources: list[list[DataItem]]) -> list[float]:
    estimator = Estimator(sources[:2])
    step_times: list[float] = []

    for i in range(2, AGENTS):
        t0 = time.perf_counter()
        estimator.add_source(sources[i])
        _compute_all_stats(estimator)
        elapsed = time.perf_counter() - t0
        step_times.append(elapsed)

        buckets, _ = estimator._get_sub_estimators()  # type: ignore[reportPrivateUsage]
        logging.info("step %4d/%d | %.3fs | buckets=%d", i + 1, AGENTS, elapsed, len(buckets))

    return step_times


def _report(step_times: list[float], total_elapsed: float) -> None:
    avg = sum(step_times) / len(step_times)
    p50 = sorted(step_times)[len(step_times) // 2]
    p95 = sorted(step_times)[int(len(step_times) * 0.95)]
    p99 = sorted(step_times)[int(len(step_times) * 0.99)]
    max_t = max(step_times)

    logging.info("─" * 60)
    logging.info("Total wall time : %.2fs", total_elapsed)
    logging.info("Steps           : %d", len(step_times))
    logging.info("Step avg        : %.3fs", avg)
    logging.info("Step p50        : %.3fs", p50)
    logging.info("Step p95        : %.3fs", p95)
    logging.info("Step p99        : %.3fs", p99)
    logging.info("Step max        : %.3fs", max_t)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rng = random.Random(RANDOM_SEED)

    sources = _generate_data(rng)

    _run_forced_split(sources)

    t_start = time.perf_counter()
    step_times = _run_benchmark(sources)
    total_elapsed = time.perf_counter() - t_start

    _report(step_times, total_elapsed)


if __name__ == "__main__":
    main()

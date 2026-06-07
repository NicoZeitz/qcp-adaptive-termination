from __future__ import annotations

import math
import random

from unknown_unknown import DataItem, Estimator, ValueEstimator, CountEstimator


def synthetic_example(
    n: int = 20,
    source_counts: list[int] | None = None,
    lambda_: float = 1.0,
    rho: float = 0.5,
    seed: int = 42,
) -> None:
    if source_counts is None:
        source_counts = [8, 8, 6]
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    if lambda_ < 0.0:
        raise ValueError("lambda_ must be >= 0")

    # Hidden ground-truth population: positive values with a heavy tail.
    random.seed(seed)
    all_companies = {
        f"company_{i}": round(random.expovariate(1 / 500) + 50, 1)
        for i in range(1, n + 1)
    }

    # Paper-style synthetic sampling signal:
    # - lambda_ controls base publicity skew by rank (Zipf-like prior)
    # - rho controls correlation between value and publicity.
    names_by_value = sorted(
        all_companies.keys(), key=lambda k: all_companies[k], reverse=True
    )
    values = [all_companies[name] for name in names_by_value]
    v_min, v_max = min(values), max(values)
    v_span = max(v_max - v_min, 1e-12)

    rank_prior = {
        name: 1.0 / math.pow(rank + 1, lambda_)
        for rank, name in enumerate(names_by_value)
    }
    value_signal = {
        name: (all_companies[name] - v_min) / v_span for name in names_by_value
    }
    mixed = {
        name: (1.0 - rho) * rank_prior[name] + rho * value_signal[name]
        for name in names_by_value
    }
    mixed_sum = sum(mixed.values())
    publicity = {
        name: (mixed[name] / mixed_sum if mixed_sum > 0 else 1.0 / len(names_by_value))
        for name in names_by_value
    }

    def sample_source(source_id: str, n: int) -> list[DataItem]:
        # Weighted sampling without replacement to enforce unique-per-source output.
        remaining_names = list(all_companies.keys())
        target = min(n, len(remaining_names))
        selected: list[DataItem] = []

        for _ in range(target):
            remaining_weights = [publicity[name] for name in remaining_names]
            chosen_name = random.choices(
                remaining_names, weights=remaining_weights, k=1
            )[0]
            selected.append(
                DataItem(
                    name=chosen_name,
                    value=all_companies[chosen_name],
                    source_id=source_id,
                )
            )
            remaining_names.remove(chosen_name)

        return selected

    random.seed(seed)
    sources = [sample_source(f"s{i + 1}", n) for i, n in enumerate(source_counts)]
    e = Estimator(sources)

    true_sum = sum(all_companies.values())
    true_count = len(all_companies)
    true_avg = true_sum / true_count
    true_min = min(all_companies.values())
    true_max = max(all_companies.values())

    # fmt: off
    def print_estimate(count_estimator: CountEstimator, value_estimator: ValueEstimator) -> None:
        print(f"{count_estimator.name.capitalize()+',':<11s} {value_estimator.name.capitalize():<20s} : {e.estimated_count(count_estimator):>10.2f} COUNT, {e.estimated_sum(value_estimator, count_estimator):>10.2f} SUM, {e.estimated_avg(value_estimator, count_estimator):>10.2f} AVG")

    print(
        f"================== Synthetic Example (n={n}, sources={len(source_counts)}, λ={lambda_}, ρ={rho}) =================="
    )
    print(f"Coverage                         :     {e.sample_coverage():.3f}")
    print(f"True                             : {true_count:>10.2f} COUNT, {true_sum:>10.2f} SUM, {true_avg:>10.2f} AVG, {true_min:>10.2f} MIN, {true_max:>10.2f} MAX")
    print(f"Sample                           : {e.sample_count():>10.2f} COUNT, {e.sample_sum():>10.2f} SUM, {e.sample_avg():>10.2f} AVG, {e.sample_min():>10.2f} MIN, {e.sample_max():>10.2f} MAX")
    # fmt: on

    for count_estimator in CountEstimator:
        for value_estimator in ValueEstimator:
            print_estimate(count_estimator, value_estimator)


if __name__ == "__main__":
    synthetic_example(n=100, source_counts=[5] * 10, lambda_=0, rho=0)

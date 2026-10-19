"""Result serialization for estimator evaluation."""

import logging
from pathlib import Path

import pandas as pd

from scripts.estimator_evaluation.models import StepResult
from scripts.estimator_evaluation.plotting import save_step_plot


def save_step_results(result: StepResult, query_slug: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{query_slug}_{result.step_id}"
    gt = result.ground_truth

    summary = {
        "step_id": result.step_id,
        "agents_total": result.agents_total,
        "agents_queried_at_stop": result.agents_queried_at_stop,
        "stop_fraction": result.stop_fraction,
        "early_stop": result.early_stop,
        "true_count": gt.count,
        "true_value_sum": gt.value_sum,
        "true_value_avg": gt.value_avg,
        "true_value_min": gt.value_min,
        "true_value_max": gt.value_max,
    }
    pd.DataFrame([summary]).to_csv(output_dir / f"{prefix}_evaluator_summary.csv", index=False)

    batch_rows = [
        {
            "batch_num": b.batch_num,
            "cumulative_agents": b.cumulative_agents,
            "decision": b.decision,
            "sample_coverage": b.sample_coverage,
            "observed_count": b.observed_count,
            "estimated_count": b.estimated_count,
            "coverage_achieved": b.coverage_achieved,
            "observed_value_sum": b.observed_value_sum,
            "estimated_value_sum": b.estimated_value_sum,
            "observed_value_avg": b.observed_value_avg,
            "estimated_value_avg": b.estimated_value_avg,
        }
        for b in result.batches
    ]
    pd.DataFrame(batch_rows).to_csv(output_dir / f"{prefix}_evaluator_batches.csv", index=False)

    save_step_plot(result, gt, prefix, output_dir)
    logging.info("Saved evaluator results to %s", output_dir)

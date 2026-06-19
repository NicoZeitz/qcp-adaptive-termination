import logging

import pandas as pd

from qtypes.aligned_result_list import AlignedSchemaMetadata
from qtypes.estimator_result import EstimatorResult
from qtypes.result import ResultType
from unknown_unknown import DataItem, Estimator
from utils.span import indented_logging


def dataframe_to_data_items(
    df: pd.DataFrame, key_cols: list[str], value_col: str | None, source_id: str
) -> list[DataItem]:
    items = []
    for _, row in df.iterrows():
        name = "|".join(str(row[k]) for k in key_cols if k in row)
        value = float(row[value_col]) if value_col and value_col in row else 1.0
        items.append(DataItem(name=name, value=value, source_id=source_id))
    return items


def compute_estimator_result(
    schema: AlignedSchemaMetadata,
    per_agent_results: dict[str, ResultType],
    agents_queried: int,
    agents_total: int,
) -> EstimatorResult | None:
    """Compute the estimator from raw per-agent results to guarantee source separation.

    Using the merged aligned.data risks losing source_agent tracking when the LLM
    deduplicates or drops the column during merge_result — which makes every entity a
    singleton and produces coverage=0 / N_hat=inf.  We build source_samples directly
    from the pre-merge per-agent DataFrames instead.
    """
    if len(per_agent_results) < 2:
        indented_logging("Estimator: skipped (fewer than 2 agents with results)")
        return None

    try:
        source_samples: list[list[DataItem]] = []
        for agent_name, df in per_agent_results.items():
            items = dataframe_to_data_items(
                df, schema.keys, schema.value_column, agent_name
            )
            if items:
                source_samples.append(items)

        if len(source_samples) < 2:
            indented_logging(
                f"Estimator: skipped (only {len(source_samples)} non-empty source(s) after filtering)"
            )
            return None

        est = Estimator(source_samples)
        agg = schema.aggregation_type
        coverage = est.sample_coverage()
        f1 = est.f1_count()
        c = est.sample_count()
        n = est._n

        indented_logging(
            f"Estimator internals: c={c} unique entities, n={n} total obs, "
            f"f1={f1} singletons, coverage={coverage:.3f}, agg={agg}"
        )

        result = EstimatorResult(
            sample_coverage=coverage,
            agents_queried=agents_queried,
            agents_total=agents_total,
            aggregation_type=agg,
        )

        if agg == "count":
            result.observed_count = est.sample_count()
            result.estimated_count = est.estimated_count()
            lower, upper = est.estimated_count_bounds()
            result.estimated_count_lower = lower
            result.estimated_count_upper = upper
        elif agg == "sum":
            result.observed_sum = est.sample_sum()
            result.estimated_sum = est.estimated_sum()
            result.estimated_sum_lower = est.estimated_sum_lower_bound()
            result.estimated_sum_upper = est.estimated_sum_upper_bound()
        elif agg == "avg":
            result.observed_avg = est.sample_avg()
            result.estimated_avg = est.estimated_avg()
            result.estimated_avg_lower = est.estimated_avg_lower_bound()
            result.estimated_avg_upper = est.estimated_avg_upper_bound()
        elif agg == "min":
            result.estimated_min = est.estimated_min()
        elif agg == "max":
            result.estimated_max = est.estimated_max()

        indented_logging(f"Estimator result: {result}")
        return result
    except Exception:
        logging.exception("Estimator failed")
        return None

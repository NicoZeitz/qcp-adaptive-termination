import pandas as pd

from qcp_adaptive_termination.estimator import DataItem, Estimator
from qcp_adaptive_termination.model.estimation.aligned_result_list import (
    AlignedResultList,
)
from qcp_adaptive_termination.model.estimation.estimator_statistics import (
    EstimatorStatistics,
)
from qcp_adaptive_termination.model.config import config_var
from qcp_adaptive_termination.tracing.span import indented_logging


def dataframe_to_data_items(
    dataframe: pd.DataFrame, key_cols: list[str], value_col: str | None, source_id: str
) -> list[DataItem]:
    items = []
    for _, row in dataframe.iterrows():
        name = "|".join(str(row[k]) for k in key_cols if k in row)
        value = float(row[value_col]) if value_col and value_col in row else 1.0  # type: ignore[arg-type]
        items.append(DataItem(name=name, value=value, source_id=source_id))
    return items


def compute_estimator_statistics(
    result_list: AlignedResultList,
    agents_queried: int,
    agents_total: int,
    agents_current: list[str],
) -> EstimatorStatistics:
    config = config_var.get()
    schema = result_list.schema_metadata
    data = result_list.data
    agg = schema.aggregation_type

    source_samples: list[list[DataItem]] = []
    for agent_name in data[schema.source_agent_column].unique():
        dataframe = pd.DataFrame(data[data[schema.source_agent_column] == agent_name])
        items = dataframe_to_data_items(
            dataframe, schema.keys, schema.value_column, agent_name
        )
        if items:
            source_samples.append(items)

    has_data = len(source_samples) >= 1
    has_overlap = len(source_samples) >= 2

    if not has_data:
        indented_logging("Estimator: no data (all agents returned empty results)")
        return EstimatorStatistics(
            agents_queried=agents_queried,
            agents_total=agents_total,
            aggregation_type=agg,
            new_unique_count=0,
            observed_count=0,
        )

    if not has_overlap:
        indented_logging(
            "Estimator: no overlap data (fewer than 2 agents with results)"
        )

    key_series = data[schema.keys].astype(str).agg("|".join, axis=1)
    is_current = data[schema.source_agent_column].isin(agents_current)
    prev_keys = set(key_series[~is_current])
    current_keys = set(key_series[is_current])
    new_unique_count = len(current_keys - prev_keys)

    estimator = Estimator(source_samples)

    result = EstimatorStatistics(
        sample_coverage=estimator.sample_coverage() if has_overlap else None,
        agents_queried=agents_queried,
        agents_total=agents_total,
        aggregation_type=agg,
        new_unique_count=new_unique_count,
        observed_count=estimator.sample_count(),
    )

    if has_overlap:
        result.estimated_count = estimator.estimated_count(
            config.count_estimator,
            use_bucket=config.use_bucket,
            use_monte_carlo=config.use_monte_carlo,
        )
        count_lower, count_upper = estimator.estimated_count_bounds()
        result.estimated_count_lower = count_lower
        result.estimated_count_upper = count_upper

    if agg == "sum":
        result.observed_value = estimator.sample_sum()
        if has_overlap:
            result.estimated_value = estimator.estimated_sum(
                config.value_estimator,
                config.count_estimator,
                use_bucket=config.use_bucket,
                use_monte_carlo=config.use_monte_carlo,
            )
            lower, upper = estimator.estimated_sum_bounds()
            result.estimated_value_lower = lower
            result.estimated_value_upper = upper
    elif agg == "avg":
        result.observed_value = estimator.sample_avg()
        if has_overlap:
            result.estimated_value = estimator.estimated_avg(
                use_bucket=config.use_bucket,
                bucket_count_estimator=config.count_estimator,
                bucket_value_estimator=config.value_estimator,
                bucket_use_monte_carlo=config.use_monte_carlo,
            )
            lower, upper = estimator.estimated_avg_bounds()
            result.estimated_value_lower = lower
            result.estimated_value_upper = upper
    elif agg == "min":
        result.observed_value = estimator.sample_min()
        if has_overlap:
            result.estimated_value = estimator.estimated_min(
                config.count_estimator,
                bucket_value_estimator=config.value_estimator,
                use_monte_carlo=config.use_monte_carlo,
            )
    elif agg == "max":
        result.observed_value = estimator.sample_max()
        if has_overlap:
            result.estimated_value = estimator.estimated_max(
                config.count_estimator,
                bucket_value_estimator=config.value_estimator,
                use_monte_carlo=config.use_monte_carlo,
            )

    return result

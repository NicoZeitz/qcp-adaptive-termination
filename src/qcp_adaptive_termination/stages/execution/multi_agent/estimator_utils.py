import pandas as pd

from qcp_adaptive_termination.estimator import DataItem, Estimator
from qcp_adaptive_termination.models import (
    AggregationType,
    AlignedResultList,
    EstimatorStatistics,
)
from qcp_adaptive_termination.observability import context, tracing


def dataframe_to_data_items(
    dataframe: pd.DataFrame, key_cols: list[str], value_col: str | None, source_id: str
) -> list[DataItem]:
    valid_keys = [k for k in key_cols if k in dataframe.columns]
    if valid_keys:
        names: pd.Series = dataframe[valid_keys[0]].astype(str)  # type: ignore[assignment]
        for k in valid_keys[1:]:
            names = names + "|" + dataframe[k].astype(str)
    else:
        names = pd.Series([""] * len(dataframe), index=dataframe.index)

    if value_col and value_col in dataframe.columns:
        raw_values: pd.Series = pd.to_numeric(dataframe[value_col], errors="coerce")  # type: ignore[assignment]
        values = raw_values.fillna(1.0)
    else:
        values = pd.Series([1.0] * len(dataframe), index=dataframe.index)

    return [DataItem(name=n, value=float(v), source_id=source_id) for n, v in zip(names, values, strict=False)]


async def compute_estimator_statistics(
    result_list: AlignedResultList,
    agents_queried: int,
    agents_total: int,
    agents_current: list[str],
    *,
    previous_estimator: Estimator | None = None,
) -> tuple[EstimatorStatistics, Estimator | None]:
    config = context.get_config()
    schema = result_list.schema_metadata
    data = result_list.data
    aggregation_types = set(schema.aggregation_types)

    estimator: Estimator | None
    if previous_estimator is not None:
        for agent_name in agents_current:
            agent_data = data[data[schema.source_agent_column] == agent_name]
            if len(agent_data) > 0:
                new_items = dataframe_to_data_items(
                    pd.DataFrame(agent_data), schema.keys, schema.value_column, agent_name
                )
                if new_items:
                    previous_estimator.add_source(new_items)
        estimator = previous_estimator
        has_data = len(previous_estimator.source_samples) >= 1
        has_overlap = len(previous_estimator.source_samples) >= 2
    else:
        source_samples: list[list[DataItem]] = []
        for agent_name in data[schema.source_agent_column].unique():
            agent_df = pd.DataFrame(data[data[schema.source_agent_column] == agent_name])
            items = dataframe_to_data_items(agent_df, schema.keys, schema.value_column, agent_name)
            if items:
                source_samples.append(items)
        has_data = len(source_samples) >= 1
        has_overlap = len(source_samples) >= 2
        estimator = Estimator(source_samples) if has_data else None

    if schema.keys:
        key_series: pd.Series = data[schema.keys[0]].astype(str)  # type: ignore[assignment]
        for _k in schema.keys[1:]:
            key_series = key_series + "|" + data[_k].astype(str)
    else:
        key_series = pd.Series([""] * len(data), index=data.index)
    is_current = data[schema.source_agent_column].isin(agents_current)
    prev_keys = set(key_series[~is_current])
    current_keys = set(key_series[is_current])
    new_unique_count = len(current_keys - prev_keys)

    if not has_data:
        tracing.indented_logging("Estimator: no data (all agents returned empty results)")
        return EstimatorStatistics(
            agents_queried=agents_queried,
            agents_total=agents_total,
            aggregation_types=sorted(aggregation_types),
            new_unique_count=0,
            observed_count=0,
        ), None

    if not has_overlap:
        tracing.indented_logging("Estimator: no overlap data (fewer than 2 agents with results)")

    assert estimator is not None

    result = EstimatorStatistics(
        sample_coverage=estimator.sample_coverage() if has_overlap else None,
        agents_queried=agents_queried,
        agents_total=agents_total,
        aggregation_types=sorted(aggregation_types),
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

    if AggregationType.SUM in aggregation_types:
        result.observed_value_sum = estimator.sample_sum()
        if has_overlap:
            result.estimated_value_sum = estimator.estimated_sum(
                config.value_estimator,
                config.count_estimator,
                use_bucket=config.use_bucket,
                use_monte_carlo=config.use_monte_carlo,
            )
            lower, upper = estimator.estimated_sum_bounds()
            result.estimated_value_sum_lower = lower
            result.estimated_value_sum_upper = upper
    if AggregationType.AVG in aggregation_types:
        result.observed_value_avg = estimator.sample_avg()
        if has_overlap:
            result.estimated_value_avg = estimator.estimated_avg(
                use_bucket=config.use_bucket,
                bucket_count_estimator=config.count_estimator,
                bucket_value_estimator=config.value_estimator,
                bucket_use_monte_carlo=config.use_monte_carlo,
            )
            lower, upper = estimator.estimated_avg_bounds()
            result.estimated_value_avg_lower = lower
            result.estimated_value_avg_upper = upper
    if AggregationType.MIN in aggregation_types:
        result.observed_value_min = estimator.sample_min()
        if has_overlap:
            result.estimated_value_min = estimator.estimated_min(
                config.count_estimator,
                bucket_value_estimator=config.value_estimator,
                use_monte_carlo=config.use_monte_carlo,
            )
    if AggregationType.MAX in aggregation_types:
        result.observed_value_max = estimator.sample_max()
        if has_overlap:
            result.estimated_value_max = estimator.estimated_max(
                config.count_estimator,
                bucket_value_estimator=config.value_estimator,
                use_monte_carlo=config.use_monte_carlo,
            )
    if AggregationType.Q25 in aggregation_types:
        result.observed_value_q25 = estimator.sample_quantile(0.25)
        if has_overlap:
            result.estimated_value_q25 = estimator.estimated_quantile(
                0.25,
                config.count_estimator,
                bucket_value_estimator=config.value_estimator,
                use_monte_carlo=config.use_monte_carlo,
            )
    if AggregationType.MEDIAN in aggregation_types:
        result.observed_value_median = estimator.sample_median()
        if has_overlap:
            result.estimated_value_median = estimator.estimated_median(
                config.count_estimator,
                bucket_value_estimator=config.value_estimator,
                use_monte_carlo=config.use_monte_carlo,
            )
    if AggregationType.Q75 in aggregation_types:
        result.observed_value_q75 = estimator.sample_quantile(0.75)
        if has_overlap:
            result.estimated_value_q75 = estimator.estimated_quantile(
                0.75,
                config.count_estimator,
                bucket_value_estimator=config.value_estimator,
                use_monte_carlo=config.use_monte_carlo,
            )

    return result, estimator

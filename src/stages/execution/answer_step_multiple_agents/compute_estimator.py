import pandas as pd

from qtypes.aligned_result_list import AlignedResultList
from qtypes.estimator_statistics import EstimatorStatistics
from unknown_unknown import DataItem, Estimator
from utils.span import indented_logging


def dataframe_to_data_items(
    dataframe: pd.DataFrame, key_cols: list[str], value_col: str | None, source_id: str
) -> list[DataItem]:
    items = []
    for _, row in dataframe.iterrows():
        name = "|".join(str(row[k]) for k in key_cols if k in row)
        value = float(row[value_col]) if value_col and value_col in row else 1.0
        items.append(DataItem(name=name, value=value, source_id=source_id))
    return items


def compute_estimator_statistics(
    result_list: AlignedResultList,
    agents_queried: int,
    agents_total: int,
    agent_current: str,
) -> None | tuple[EstimatorStatistics, Estimator]:
    schema = result_list.schema_metadata
    data = result_list.data

    if len(data[schema.source_agent_column].unique()) < 2:
        indented_logging("Estimator: skipped (fewer than 2 agents with results)")
        return None

    source_samples: list[list[DataItem]] = []
    for agent_name in data[schema.source_agent_column].unique():
        dataframe = data[data[schema.source_agent_column] == agent_name]
        data_items = dataframe_to_data_items(
            dataframe, schema.keys, schema.value_column, agent_name
        )
        source_samples.append(data_items)

    estimator = Estimator(source_samples)
    agg = schema.aggregation_type

    new_unique_count = (result_list.data[schema.source_agent_column] == agent_current).sum()

    result = EstimatorStatistics(
        sample_coverage=estimator.sample_coverage(),
        agents_queried=agents_queried,
        agents_total=agents_total,
        aggregation_type=agg,
        new_unique_count=new_unique_count,
    )

    if agg == "count":
        result.observed_value = estimator.sample_count()
        result.estimated_value = estimator.estimated_count()
        lower, upper = estimator.estimated_count_bounds()
        result.estimated_value_lower = lower
        result.estimated_value_upper = upper
    elif agg == "sum":
        result.observed_value = estimator.sample_sum()
        result.estimated_value = estimator.estimated_sum()
        lower, upper = estimator.estimated_sum_bounds()
        result.estimated_value_lower = lower
        result.estimated_value_upper = upper
    elif agg == "avg":
        result.observed_value = estimator.sample_avg()
        result.estimated_value = estimator.estimated_avg()
        lower, upper = estimator.estimated_avg_bounds()
        result.estimated_value_lower = lower
        result.estimated_value_upper = upper
    elif agg == "min":
        result.observed_value = estimator.sample_min()
        result.estimated_value = estimator.estimated_min()
    elif agg == "max":
        result.observed_value = estimator.sample_max()
        result.estimated_value = estimator.estimated_max()

    return result, estimator

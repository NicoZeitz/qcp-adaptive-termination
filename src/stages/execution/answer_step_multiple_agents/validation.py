from typing import Any

from pandas import DataFrame

from qtypes.aligned_result_list import AggregationType, AlignedSchemaMetadata
from qtypes.result import ResultType, try_parse_value_to_result_type

SOURCE_AGENT_COLUMN = "source_agent"


def validate_aligned_schema_metadata(
    dataframe: DataFrame, metadata: AlignedSchemaMetadata
):
    validate_aggregation_type(metadata.aggregation_type)
    validate_keys(dataframe, metadata.keys)
    validate_value_column(dataframe, metadata.value_column)


def validate_aggregation_type(aggregation_type: AggregationType):
    valid_aggregation_types = {"count", "min", "max", "sum", "avg"}
    if aggregation_type not in valid_aggregation_types:
        raise ValueError(
            f"Invalid aggregation type: {aggregation_type}. Must be one of {', '.join(valid_aggregation_types)}."
        )


def validate_keys(dataframe: DataFrame, keys: list[str]):
    if not isinstance(keys, list) or not all(isinstance(key, str) for key in keys):
        raise ValueError("Keys must be a list of strings.")
    for key in keys:
        if key not in dataframe.columns:
            raise ValueError(f"Key '{key}' not found in DataFrame columns.")


def validate_value_column(dataframe: DataFrame, value_column: str | None):
    if value_column is None:
        return

    if not isinstance(value_column, str):
        raise ValueError("Value column must be a string.")
    if value_column not in dataframe.columns:
        raise ValueError(
            f"Value column '{value_column}' not found in DataFrame columns."
        )


def validate_source_agent_column(
    current_data: DataFrame, new_data: DataFrame, new_agent: str
):
    if SOURCE_AGENT_COLUMN not in new_data.columns:
        raise ValueError(
            f"New data must contain the '{SOURCE_AGENT_COLUMN}' column to indicate source agents. Please include this column in the new data with the same source agent values as in the current data."
        )

    current_agent_counts = current_data[SOURCE_AGENT_COLUMN].value_counts()
    new_agent_counts = new_data[SOURCE_AGENT_COLUMN].value_counts()

    for agent, count in current_agent_counts.items():
        if agent not in new_agent_counts:
            raise ValueError(
                f"New data is missing source agent '{agent}'. Please keep the same source agents in the new data."
            )

        new_agent_count = new_agent_counts.get(agent, 0)
        if new_agent_count != count:
            raise ValueError(
                f"Source agent '{agent}' has {count} entries in the current data but {new_agent_count} entries in the new data. Please keep all rows for each source agent in the new data."
            )

    new_data_source_agents = set(new_data[SOURCE_AGENT_COLUMN].unique())
    allowed_agents = set(current_data[SOURCE_AGENT_COLUMN].unique()) | {new_agent}
    if not new_data_source_agents.issubset(allowed_agents):
        raise ValueError(
            f"New data contains unexpected source agents: {new_data_source_agents - allowed_agents}"
        )

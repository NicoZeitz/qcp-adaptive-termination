from pandas import DataFrame
import pandas as pd

from common.model.estimation.aligned_result_list import AlignedSchemaMetadata


def validate_agent_output(dataframe: DataFrame, metadata: AlignedSchemaMetadata):
    errors = []

    expected = set(metadata.columns)
    actual = set(dataframe.columns)

    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        parts = []
        if missing:
            parts.append(f"missing columns: {missing}")
        if extra:
            parts.append(f"unexpected columns: {extra}")

        errors.append(
            f"Output does not match the required schema — {'; '.join(parts)}. Expected exactly: {sorted(expected)}"
        )

    # if there is a value column, check that it is not empty and numeric
    if metadata.value_column is not None:
        if metadata.value_column not in dataframe.columns:
            errors.append(
                f"Column '{metadata.value_column}' is missing from the output"
            )
        else:
            if dataframe[metadata.value_column].isnull().all():
                errors.append(f"Column '{metadata.value_column}' cannot be empty")
            elif not pd.api.types.is_numeric_dtype(dataframe[metadata.value_column]):
                errors.append(f"Column '{metadata.value_column}' has to be numeric")

    if len(errors) > 0:
        raise ValueError("\n".join(errors))


def validate_schema_output(
    columns: set[str],
    keys: set[str],
    value_column: str | None,
    source_agent_column: str,
    aggregation_type: str,
):
    errors = []

    if not keys.issubset(columns):
        missing = sorted(keys - columns)
        errors.append(f"keys {missing} are not in columns")

    if value_column is not None and value_column not in columns:
        errors.append(f"value_column '{value_column}' is not in columns")

    if value_column is not None and value_column in keys:
        errors.append(f"value_column '{value_column}' must not overlap with keys")

    if aggregation_type not in {"count", "min", "max", "sum", "avg"}:
        errors.append(f"aggregation_type '{aggregation_type}' is not valid")

    if source_agent_column in columns:
        errors.append(f"{source_agent_column} is not an allowed column name")

    if len(errors) > 0:
        raise ValueError("\n".join(errors))

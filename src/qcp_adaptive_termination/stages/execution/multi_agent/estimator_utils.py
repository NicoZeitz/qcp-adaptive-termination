"""Incremental estimator maintenance for the multi-agent step loop.

The loop carries a single lazy ``Estimator`` that is updated incrementally as
each agent's result is merged. No statistics are computed eagerly — consumers
(completeness evaluator, result producer, uncertainty recording) pull only what
they need from the estimator on demand.
"""

import pandas as pd

from qcp_adaptive_termination.estimator import DataItem, Estimator
from qcp_adaptive_termination.models import AlignedResultList
from qcp_adaptive_termination.observability import context


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


def update_estimator(
    estimator: Estimator | None,
    result_list: AlignedResultList,
    agent_name: str,
) -> Estimator | None:
    """Fold one agent's merged rows into the estimator, creating it if needed.

    Returns the (possibly new) estimator, or ``None`` if the agent contributed
    no usable data.
    """
    schema = result_list.schema_metadata
    data = result_list.data
    agent_data = data[data[schema.source_agent_column] == agent_name]
    if len(agent_data) == 0:
        return estimator

    items = dataframe_to_data_items(pd.DataFrame(agent_data), schema.keys, schema.value_column, agent_name)
    if not items:
        return estimator

    if estimator is None:
        return Estimator([[item] for item in items], context.get_config().estimator)
    estimator.add_source(items)
    return estimator

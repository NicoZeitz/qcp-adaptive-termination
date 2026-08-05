from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from pandas import DataFrame, MultiIndex
from pydantic import JsonValue

if TYPE_CHECKING:
    from model.execution.additional_information import AdditionalInformation

CHAR_LIMIT_LLM = 2000
CHAR_LIMIT_LONG_LLM = 6000


def preview_dataframe(
    df: DataFrame | None,
    top_rows: int = 5,
    bottom_rows: int = 5,
    show_dtypes: bool = True,
    show_row_numbers: bool = True,
) -> str:
    if df is None:
        return "None"
    if df.empty:
        return df.__repr__()
    df_copy = df.copy()
    if show_dtypes:
        columns = [(c, df_copy[c].dtype) for c in df_copy.columns]
        df_copy.columns = MultiIndex.from_tuples(columns, names=["name", "type"])
    if not show_row_numbers:
        df_copy.index = [""] * len(df_copy)
    with pd.option_context("display.max_rows", top_rows + bottom_rows + 1):
        return df_copy.__repr__()


def preview_text(text: str, char_limit=CHAR_LIMIT_LLM):
    if len(text) <= char_limit:
        return text

    char_limit_start = char_limit // 2
    char_limit_end = char_limit - char_limit_start

    omitted_count = len(text) - (char_limit_start + char_limit_end)

    return (
        text[:char_limit_start]
        + f"\n...[{omitted_count} chars]...\n"
        + text[-char_limit_end:]
    )


def preview_additional_information(
    additional_information: list[AdditionalInformation],
) -> str | None:
    if not additional_information:
        return None
    parts = []
    for info in additional_information:
        data = info.data
        preview = preview_dataframe(data) if isinstance(data, DataFrame) else str(data)
        parts.append(
            f"step_id={info.step_id} | dependency={info.dependency}\n{preview}"
        )
    return "\n\n".join(parts)


def result_to_json_preview(result: DataFrame) -> JsonValue:
    return preview_dataframe(result)


def need_to_json_preview(need: str) -> JsonValue:
    return need

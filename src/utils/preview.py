import pandas as pd
from pandas import DataFrame, MultiIndex

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

from dataclasses import dataclass, field
import textwrap
from typing import Any

import pandas as pd
from sqlalchemy import bindparam, text

from qcp_adaptive_termination.models import DataAgent
from qcp_adaptive_termination.db.utils import get_db_engine, quote_col, quote_table

DEFAULT_HISTOGRAM_BINS = 100
DEFAULT_MAX_MCV = 100


@dataclass(slots=True)
class ColumnStatistics:
    column_name: str
    distinct_count: int
    non_null_count: int
    min_value: Any | None
    max_value: Any | None
    most_common_values: pd.DataFrame  # columns: value, frequency
    histogram: pd.DataFrame  # columns: bin_start, bin_end, count


@dataclass(slots=True)
class TableStatistics:
    table_name: str
    total_rows: int
    columns: list[ColumnStatistics] = field(default_factory=list)


@dataclass(slots=True)
class DatabaseStatistics:
    name: str
    tables: list[TableStatistics] = field(default_factory=list)


def fetch_column_statistics(
    data_agent: DataAgent,
    table_name: str,
    column_name: str,
    histogram_bins: int = DEFAULT_HISTOGRAM_BINS,
    max_mcv: int = DEFAULT_MAX_MCV,
) -> ColumnStatistics:
    return fetch_table_statistics(
        data_agent,
        table_name,
        columns=[column_name],
        histogram_bins=histogram_bins,
        max_mcv=max_mcv,
    ).columns[0]


def fetch_table_statistics(
    data_agent: DataAgent,
    table_name: str,
    *,
    columns: list[str] | None = None,
    histogram_bins: int = DEFAULT_HISTOGRAM_BINS,
    max_mcv: int = DEFAULT_MAX_MCV,
) -> TableStatistics:
    db_engine = get_db_engine(data_agent)
    with db_engine.connect() as conn:
        if columns is None:
            rows = conn.execute(
                text(f"PRAGMA table_info({quote_table(table_name)})")
            ).fetchall()
            columns = [row[1] for row in rows]

        # Base Statistics: total rows, non-null counts, distinct counts, min/max
        col_exprs = ["COUNT(*) AS total_rows"]
        for i, col in enumerate(columns):
            col_exprs += [
                f"COUNT({quote_col(col)}) AS non_null_{i}",
                f"COUNT(DISTINCT {quote_col(col)}) AS distinct_{i}",
                f"MIN({quote_col(col)}) AS min_{i}",
                f"MAX({quote_col(col)}) AS max_{i}",
            ]
        base_row = (
            conn.execute(
                text(f"SELECT {', '.join(col_exprs)} FROM {quote_table(table_name)}")
            )
            .mappings()
            .one()
        )

        total_rows = int(base_row["total_rows"])
        column_stats: list[ColumnStatistics] = []

        for i, col in enumerate(columns):
            non_null = int(base_row[f"non_null_{i}"])
            distinct = int(base_row[f"distinct_{i}"])
            min_value = base_row[f"min_{i}"]
            max_value = base_row[f"max_{i}"]

            # MCV: postgres threshold formula f > 1/K + sqrt((1 - 1/K) / n)
            if non_null > 0 and distinct > 0:
                mcv_rows = conn.execute(
                    text(
                        textwrap.dedent(f"""
                        SELECT {quote_col(col)} AS value, COUNT(*) * 1.0 / :n AS frequency
                        FROM {quote_table(table_name)}
                        WHERE {quote_col(col)} IS NOT NULL AND {quote_col(col)} <> ''
                        GROUP BY {quote_col(col)}
                        HAVING COUNT(*) * 1.0 / :n > 1.0 / :k + SQRT((1.0 - 1.0 / :k) / :n)
                        ORDER BY COUNT(*) DESC
                        LIMIT :max_mcv
                    """)
                    ),
                    {"n": non_null, "k": distinct, "max_mcv": max_mcv},
                ).fetchall()
                mcv = pd.DataFrame(mcv_rows, columns=["value", "frequency"])
            else:
                mcv = pd.DataFrame(columns=["value", "frequency"])

            mcv_values = mcv["value"].tolist()

            # Histogram over non-MCV, non-null values via NTILE bucketing
            not_in = f"AND {quote_col(col)} NOT IN :mcv_values" if mcv_values else ""
            hist_sql = textwrap.dedent(f"""
                WITH non_mcv AS (
                    SELECT {quote_col(col)} AS val
                    FROM {quote_table(table_name)}
                    WHERE {quote_col(col)} IS NOT NULL AND {quote_col(col)} <> '' {not_in}
                ),
                bucketed AS (
                    SELECT val, NTILE(:bins) OVER (ORDER BY val) AS bucket
                    FROM non_mcv
                )
                SELECT
                    MIN(val) AS bin_start,
                    MAX(val) AS bin_end,
                    COUNT(*) AS count
                FROM bucketed
                GROUP BY bucket
                ORDER BY bucket
            """)
            stmt = text(hist_sql)
            if mcv_values:
                stmt = stmt.bindparams(bindparam("mcv_values", expanding=True))
            params: dict = {"bins": histogram_bins}
            if mcv_values:
                params["mcv_values"] = mcv_values
            hist_rows = conn.execute(stmt, params).fetchall()

            histogram = (
                pd.DataFrame(hist_rows, columns=["bin_start", "bin_end", "count"])
                if hist_rows
                else pd.DataFrame(columns=["bin_start", "bin_end", "count"])
            )

            column_stats.append(
                ColumnStatistics(
                    column_name=col,
                    min_value=min_value,
                    max_value=max_value,
                    distinct_count=distinct,
                    non_null_count=non_null,
                    most_common_values=mcv,
                    histogram=histogram,
                )
            )

    return TableStatistics(
        table_name=table_name, total_rows=total_rows, columns=column_stats
    )


def fetch_database_statistics(
    data_agent: DataAgent,
    histogram_bins: int = DEFAULT_HISTOGRAM_BINS,
    max_mcv: int = DEFAULT_MAX_MCV,
) -> DatabaseStatistics:
    db_engine = get_db_engine(data_agent)
    with db_engine.connect() as conn:
        rows = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table'")
        ).fetchall()
    tables = [row[0] for row in rows]

    table_stats = [
        fetch_table_statistics(
            data_agent, table_name, histogram_bins=histogram_bins, max_mcv=max_mcv
        )
        for table_name in tables
    ]
    return DatabaseStatistics(name=data_agent.name, tables=table_stats)

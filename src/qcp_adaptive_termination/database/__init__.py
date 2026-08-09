"""
SQLite database utilities shared across the project.

Provides engine creation, SQL quoting helpers, and table/column statistics fetching.
"""

import qcp_adaptive_termination.database.database_statistics as database_statistics
from qcp_adaptive_termination.database.database_statistics import (
    ColumnStatistics,
    DatabaseStatistics,
    TableStatistics,
)
from qcp_adaptive_termination.database.mschema import MSchema, SchemaEngine
from qcp_adaptive_termination.database.utils import (
    fetch_column_values,
    fetch_count,
    fetch_distinct_count,
    get_database_engine,
    quote_col,
    quote_table,
)

__all__ = [
    "ColumnStatistics",
    "DatabaseStatistics",
    "MSchema",
    "SchemaEngine",
    "TableStatistics",
    "database_statistics",
    "get_database_engine",
    "quote_col",
    "quote_table",
    "fetch_count",
    "fetch_distinct_count",
    "fetch_column_values",
]

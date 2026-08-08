"""
SQLite database utilities shared across the project.

Provides engine creation, SQL quoting helpers, and table/column statistics fetching.
"""

from qcp_adaptive_termination.db.get_db_engine import get_db_engine
from qcp_adaptive_termination.db.sql_utils import quote_col, quote_table
from qcp_adaptive_termination.db.table_statistics import (
    ColumnStatistics,
    DatabaseStatistics,
    TableStatistics,
    fetch_column_statistics,
    fetch_database_statistics,
    fetch_table_statistics,
)

__all__ = [
    "get_db_engine",
    "quote_col",
    "quote_table",
    "ColumnStatistics",
    "DatabaseStatistics",
    "TableStatistics",
    "fetch_column_statistics",
    "fetch_database_statistics",
    "fetch_table_statistics",
]

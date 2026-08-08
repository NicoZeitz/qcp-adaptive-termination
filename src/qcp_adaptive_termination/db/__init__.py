"""
SQLite database utilities shared across the project.

Provides engine creation, SQL quoting helpers, and table/column statistics fetching.
"""

import qcp_adaptive_termination.db.database_statistics as database_statistics
from qcp_adaptive_termination.db.database_statistics import (
    ColumnStatistics,
    DatabaseStatistics,
    TableStatistics,
)
from qcp_adaptive_termination.db.mschema import MSchema, SchemaEngine
from qcp_adaptive_termination.db.utils import get_db_engine, quote_col, quote_table

__all__ = [
    "ColumnStatistics",
    "DatabaseStatistics",
    "MSchema",
    "SchemaEngine",
    "TableStatistics",
    "database_statistics",
    "get_db_engine",
    "quote_col",
    "quote_table",
]

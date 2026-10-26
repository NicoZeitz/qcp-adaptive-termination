"""SQLite engine helpers, identifier quoting, and small aggregate queries."""

import os

from sqlalchemy import Engine, create_engine, text

from qcp_adaptive_termination.models import DataAgent
from qcp_adaptive_termination.observability import context

_engine_cache: dict[str, Engine] = {}


def _get_db_engine_for_path(resolved_path: str) -> Engine:
    """Return a cached read-only SQLite engine for a resolved path."""
    if resolved_path not in _engine_cache:
        engine = create_engine(f"sqlite:///file:{resolved_path}?mode=ro&uri=true")
        with engine.connect() as connection:
            connection.execute(text("PRAGMA temp_store=MEMORY;"))
        _engine_cache[resolved_path] = engine
    return _engine_cache[resolved_path]


def get_database_engine(data_agent: DataAgent) -> Engine:
    """Return a read-only engine for the agent's database path."""
    data_dir = context.get_data_dir()
    if data_dir is None:
        raise RuntimeError("data_dir context not set. Use data_context() or pipeline_context(data_dir=...).")
    resolved_path = os.path.abspath(os.path.join(data_dir, data_agent.path))
    return _get_db_engine_for_path(resolved_path)


def quote_table(name: str) -> str:
    """Quote a table name, optionally schema-qualified, for SQL."""
    return ".".join(f'"{p}"' for p in name.split(".", 1))


def quote_col(name: str) -> str:
    """Quote a column name for SQL."""
    return f'"{name}"'


def fetch_count(data_agent: DataAgent, table_name: str) -> int:
    """Return the row count of a table."""
    db_engine = get_database_engine(data_agent)
    with db_engine.connect() as conn:
        result = conn.execute(text(f"SELECT COUNT(*) FROM {quote_table(table_name)}")).scalar()
    return int(result or 0)


def fetch_distinct_count(data_agent: DataAgent, table_name: str, column_name: str) -> int:
    """Return the distinct value count of a column."""
    db_engine = get_database_engine(data_agent)
    with db_engine.connect() as conn:
        result = conn.execute(
            text(f"SELECT COUNT(DISTINCT {quote_col(column_name)}) FROM {quote_table(table_name)}")
        ).scalar()
    return int(result or 0)


def fetch_column_values(data_agent: DataAgent, table_name: str, column_name: str) -> list[str]:
    """Return the non-empty values of a column."""
    db_engine = get_database_engine(data_agent)
    with db_engine.connect() as conn:
        rows = conn.execute(
            text(
                f"SELECT {quote_col(column_name)} FROM {quote_table(table_name)}"
                f" WHERE {quote_col(column_name)} IS NOT NULL AND {quote_col(column_name)} <> ''"
            )
        ).fetchall()
    return [str(row[0]) for row in rows]

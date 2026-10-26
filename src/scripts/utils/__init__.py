"""Shared utilities for scripts."""

from scripts.utils.canonical import (
    load_meta_mapping,
    make_align_schema_override,
    make_data_agent_override,
    parse_canonical_table,
    query_agent_canonical,
    slug_from_agent,
)
from scripts.utils.orchestrator import run_parallel
from scripts.utils.overrides import (
    make_align_schema_dispatch,
    make_completeness_override,
    make_data_agent_dispatch,
)
from scripts.utils.setup import ScriptContext, run_script
from scripts.utils.sql import parse_select_columns, rewrite_sql_identifiers

__all__ = [
    "ScriptContext",
    "load_meta_mapping",
    "make_align_schema_dispatch",
    "make_align_schema_override",
    "make_completeness_override",
    "make_data_agent_dispatch",
    "make_data_agent_override",
    "parse_canonical_table",
    "parse_select_columns",
    "query_agent_canonical",
    "rewrite_sql_identifiers",
    "run_parallel",
    "run_script",
    "slug_from_agent",
]

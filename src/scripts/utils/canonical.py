"""Canonical query execution against agent databases.

Rewrites canonical step SQL per agent database and executes it, returning
dataframes with canonical column names.
"""

import hashlib
import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pandas as pd
import yaml
from sqlalchemy import create_engine, text

from qcp_adaptive_termination.models import (
    AggregationType,
    AlignedResultList,
    AlignedSchemaMetadata,
    DataAgent,
)
from qcp_adaptive_termination.models.execution import (
    AdditionalInformation,
    PartialQueryAnswer,
)
from scripts.utils.sql import rewrite_sql_identifiers

_META_MAPPING_CACHE: dict[str, Any] | None = None


def load_meta_mapping(dataset_dir: Path) -> dict[str, Any]:
    """Load and cache the canonical-to-actual schema mapping."""
    global _META_MAPPING_CACHE
    if _META_MAPPING_CACHE is None:
        mapping_path = dataset_dir / "_meta" / "meta_mapping.yaml"
        with open(mapping_path) as f:
            _META_MAPPING_CACHE = cast(dict[str, Any], yaml.safe_load(f) or {})
    return _META_MAPPING_CACHE


def slug_from_agent(agent: DataAgent) -> str:
    """Directory name of the agent's database file (matches meta_mapping keys)."""
    return Path(agent.path).parent.name


def parse_canonical_table(sql: str) -> str | None:
    """Extract first table name from FROM clause."""
    match = re.search(r"\bFROM\s+(\w+)", sql, re.IGNORECASE)
    return match.group(1) if match else None


def query_agent_canonical(
    agent: DataAgent,
    canonical_sql: str,
    canonical_table: str,
    meta_mapping: dict[str, Any],
    data_dir: Path,
) -> pd.DataFrame | None:
    """Execute rewritten step SQL on agent DB, return df with canonical column names."""
    slug = slug_from_agent(agent)
    table_meta = meta_mapping.get(canonical_table, {}).get(slug)
    if table_meta is None:
        return None

    col_map: dict[str, str] = table_meta.get("columns", {})
    try:
        actual_sql = rewrite_sql_identifiers(canonical_sql, canonical_table, table_meta["name"], col_map)
    except Exception as exc:
        logging.warning("Failed to rewrite SQL for %s (%s): %s", agent.name, slug, exc)
        return None

    db_path = data_dir / agent.path
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(actual_sql), conn)
        if df.empty:
            return None
        reverse_map = {v: k for k, v in col_map.items()}
        return df.rename(columns=reverse_map)
    except Exception as exc:
        logging.warning("Failed to query %s (%s): %s", agent.name, slug, exc)
        return None
    finally:
        engine.dispose()


def make_align_schema_override(step_meta: dict[str, Any], query: str) -> Callable[..., AlignedResultList]:
    """Build an ALIGN_SCHEMA hook handler that returns an empty aligned schema."""

    def _handler(
        execution_id: str,
        data_agents: list[DataAgent],
        agent_names: list[str],
        query: str,
        merge_task: str,
    ) -> AlignedResultList:
        """Return an empty aligned schema derived from step metadata."""
        meta = step_meta["aligned_schema_metadata"]
        source_col = f"source_agent_{hashlib.sha256(query.encode()).hexdigest()[:8]}"
        agg_types = [AggregationType(a) for a in meta["aggregation_types"]]
        schema = AlignedSchemaMetadata(
            aggregation_types=agg_types,
            keys=meta["keys"],
            value_column=meta.get("value_column"),
            columns=meta["columns"],
            source_agent_column=source_col,
        )
        return AlignedResultList(
            data=pd.DataFrame(columns=meta["columns"] + [source_col]),
            schema_metadata=schema,
        )

    return _handler


def make_data_agent_override(
    canonical_sql: str,
    canonical_table: str,
    meta_mapping: dict[str, Any],
    data_cache: dict[str, pd.DataFrame | None],
    data_dir: Path,
) -> Callable[..., PartialQueryAnswer]:
    """Build a DATA_AGENT hook handler that serves cached canonical results."""

    def _handler(
        execution_id: str,
        data_agent: DataAgent,
        query: str,
        additional_information: list[AdditionalInformation],
        schema_metadata: AlignedSchemaMetadata | None,
    ) -> PartialQueryAnswer:
        """Serve the cached canonical result for the queried agent."""
        if data_agent.name not in data_cache:
            data_cache[data_agent.name] = query_agent_canonical(
                data_agent, canonical_sql, canonical_table, meta_mapping, data_dir
            )
        df = data_cache[data_agent.name]
        partial = PartialQueryAnswer()
        partial.set_answer(df if df is not None and not df.empty else pd.DataFrame())
        return partial

    return _handler

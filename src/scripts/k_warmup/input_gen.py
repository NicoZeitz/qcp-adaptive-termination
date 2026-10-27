"""Generate k-warmup input files: one single-step ``SELECT * FROM <table>`` per table.

Input generation relies on the same top-level input file's ``value``
(CollaborativeQueryAnswering with all data agents) so the study reuses the full
agent registry; each generated file's ``extra`` carries the single-step plan and
steps metadata for one table.
"""

import json
import logging
from pathlib import Path

from qcp_adaptive_termination.models import AggregationType
from scripts.k_warmup.metrics import MetaMapping
from scripts.utils.canonical import load_meta_mapping, slug_from_agent


def enumerate_tables(meta_mapping: MetaMapping) -> list[str]:
    """All canonical tables in the meta mapping."""
    return list(meta_mapping.keys())


def relevant_agents_for_table(all_agents, meta_mapping: MetaMapping, table: str) -> list:
    """Agents whose database contains the given table.

    Args:
        all_agents: all known data agents
        meta_mapping: canonical-to-actual schema mapping
        table: the canonical table name

    Returns:
        (list): the agents containing the table
    """
    slugs = set((meta_mapping.get(table) or {}).keys())
    return [a for a in all_agents if slug_from_agent(a) in slugs]


def _schema_metadata_for_table(meta_mapping: MetaMapping, table: str) -> dict[str, object]:
    """Build aligned-schema metadata for a SELECT * step from the first agent's mapping."""
    slugs = meta_mapping.get(table) or {}
    first = next(iter(slugs.values()), {}) or {}
    cols = list((first.get("columns") or {}).keys())
    return {
        "aggregation_types": [AggregationType.COUNT.value],
        "columns": cols,
        "keys": cols[:1],
        "value_column": None,
    }


def _build_step(table: str, agent_names: list[str]) -> dict[str, object]:
    """A single multi-agent step that selects all rows of the table."""
    return {
        "id": "s1",
        "query": f"Return all rows from {table}, with all columns.",
        "to_agent": list(agent_names),
        "merge_task": f"Deduplicate by the {table} id, keep all columns.",
        "dependencies": [],
        "answer": None,
    }


def generate_inputs(
    input_dir: Path,
    dataset_dir: Path,
    all_agents,
    *,
    workers: int | None = None,
) -> list[Path]:
    """Write one k-warmup input file per canonical table and return the paths.

    Args:
        input_dir: directory to write the generated input files
        dataset_dir: dataset folder with _meta/meta_mapping.yaml
        all_agents: all known data agents (from a template value)
        workers: optional parallel-ordering worker count, embedded in extra

    Returns:
        (list[Path]): the generated input file paths
    """
    meta_mapping = load_meta_mapping(dataset_dir)
    input_dir.mkdir(parents=True, exist_ok=True)

    value = {
        "__type__": "CollaborativeQueryAnswering",
        "query": "Return all rows.",
        "data_agents": [a.model_dump(mode="json") for a in all_agents],
    }

    written: list[Path] = []
    for table in enumerate_tables(meta_mapping):
        relevant = relevant_agents_for_table(all_agents, meta_mapping, table)
        if len(relevant) < 2:
            logging.info("Table %s: only %d agent(s) — skipping", table, len(relevant))
            continue
        agent_names = [a.name for a in relevant]
        step = _build_step(table, agent_names)
        schema_meta = _schema_metadata_for_table(meta_mapping, table)
        sql = f"SELECT * FROM {table}"

        extra = {
            "query_plan": {
                "query": f"Return all rows from {table}.",
                "steps": [step],
            },
            "sql_query": sql,
            "relevant_data_agents": agent_names,
            "steps_metadata": {
                "s1": {
                    "aligned_schema_metadata": schema_meta,
                    "sql_query": sql,
                }
            },
        }
        if workers is not None:
            extra["k_warmup_workers"] = workers

        path = input_dir / f"table_{table}.json"
        path.write_text(json.dumps({"value": value, "extra": extra}, indent=2), encoding="utf-8")
        written.append(path)
    return written

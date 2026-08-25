"""SQL rewriting utilities for canonical-to-actual identifier remapping."""

from typing import Any

import sqlglot
import sqlglot.expressions as exp

__all__ = [
    "parse_select_columns",
    "rewrite_sql_identifiers",
]


def parse_select_columns(sql: str) -> list[str]:
    """Return output column names from SELECT (alias > column name > output_name)."""
    tree = sqlglot.parse_one(sql, dialect="sqlite")
    if not isinstance(tree, exp.Select):
        return []
    cols = []
    for sel in tree.selects:
        if isinstance(sel, exp.Alias):
            cols.append(sel.alias)
        elif isinstance(sel, exp.Column):
            cols.append(sel.name)
        elif not isinstance(sel, exp.Star):
            name = sel.output_name
            if name:
                cols.append(name)
    return cols


def rewrite_sql_identifiers(
    sql: str,
    canonical_table: str,
    actual_table: str,
    col_map: dict[str, str],
) -> str:
    """Rewrite table and column identifiers in sql from canonical to actual names."""
    tree = sqlglot.parse_one(sql, dialect="sqlite")

    def _transformer(node: Any) -> Any:
        if isinstance(node, exp.Table) and node.name.lower() == canonical_table.lower():
            new_node = node.copy()
            new_node.set("this", exp.to_identifier(actual_table))
            return new_node
        if isinstance(node, exp.Column) and node.name in col_map:
            new_node = node.copy()
            new_node.set("this", exp.to_identifier(col_map[node.name]))
            if node.table and node.table.lower() == canonical_table.lower():
                new_node.set("table", exp.to_identifier(actual_table))
            return new_node
        return node

    return tree.transform(_transformer).sql(dialect="sqlite")

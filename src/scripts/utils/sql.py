"""SQL rewriting utilities for canonical-to-actual identifier remapping."""

from numbers import Real
from typing import Any

import sqlglot
import sqlglot.expressions as exp

__all__ = [
    "literal_key",
    "literal_predicate_sets",
    "parse_select_columns",
    "replace_literal_predicates",
    "rewrite_sql_identifiers",
]


def literal_key(value: Any) -> str:
    """Normalize a literal or data value for set comparison."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _predicate_literals(node: Any) -> tuple[exp.Column, list[exp.Literal]] | None:
    """Return (column, literals) for ``col = lit`` or ``col IN (lit, ...)`` predicates."""
    if isinstance(node, exp.EQ):
        left, right = node.this, node.expression
        if isinstance(left, exp.Column) and isinstance(right, exp.Literal):
            return left, [right]
        if isinstance(right, exp.Column) and isinstance(left, exp.Literal):
            return right, [left]
    if isinstance(node, exp.In) and isinstance(node.this, exp.Column):
        literals = node.expressions
        if literals and all(isinstance(lit, exp.Literal) for lit in literals):
            return node.this, [lit for lit in literals if isinstance(lit, exp.Literal)]
    return None


def literal_predicate_sets(sql: str) -> list[frozenset[str]]:
    """Return the literal value set of every ``col = lit`` / ``col IN (...)`` predicate in sql."""
    tree = sqlglot.parse_one(_normalize_quotes(sql), dialect="sqlite")
    sets: list[frozenset[str]] = []
    for node in tree.find_all(exp.EQ, exp.In):
        found = _predicate_literals(node)
        if found is not None:
            sets.append(frozenset(literal_key(lit.this) for lit in found[1]))
    return sets


def replace_literal_predicates(sql: str, old_values: frozenset[str], new_values: list[Any]) -> str:
    """Replace every predicate whose literal set equals old_values with ``col IN (new_values)``."""
    tree = sqlglot.parse_one(_normalize_quotes(sql), dialect="sqlite")
    literals = [
        exp.Literal.number(literal_key(v))
        if isinstance(v, Real) and not isinstance(v, bool)
        else exp.Literal.string(str(v))
        for v in new_values
    ]

    def _transformer(node: Any) -> Any:
        """Swap matching literal predicates for an IN over the new values."""
        if not isinstance(node, (exp.EQ, exp.In)):
            return node
        found = _predicate_literals(node)
        if found is None or frozenset(literal_key(lit.this) for lit in found[1]) != old_values:
            return node
        if not literals:
            return exp.false()
        return exp.In(this=found[0].copy(), expressions=[lit.copy() for lit in literals])

    return tree.transform(_transformer).sql(dialect="sqlite")


def _normalize_quotes(sql: str) -> str:
    """Normalize curly quotes to ASCII quotes."""
    return sql.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')


def parse_select_columns(sql: str) -> list[str]:
    """Return output column names from SELECT (alias > column name > output_name)."""
    tree = sqlglot.parse_one(_normalize_quotes(sql), dialect="sqlite")
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
    tree = sqlglot.parse_one(_normalize_quotes(sql), dialect="sqlite")

    def _transformer(node: Any) -> Any:
        """Rewrite matching table and column identifiers in a node."""
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

"""SQLAlchemy-backed engine that builds an `MSchema` from a live database."""

import os

import yaml
from llama_index.core import SQLDatabase
from sqlalchemy import (
    MetaData,
    text,
)
from sqlalchemy.engine import Engine

from qcp_adaptive_termination.database.mschema.m_schema import MSchema
from qcp_adaptive_termination.database.mschema.utils import examples_to_str


class SchemaEngine(SQLDatabase):
    """Database engine that builds an `MSchema` from a live SQLAlchemy connection."""

    def __init__(
        self,
        engine: Engine,
        schema: str | None = None,
        metadata: MetaData | None = None,
        ignore_tables: list[str] | None = None,
        include_tables: list[str] | None = None,
        sample_rows_in_table_info: int = 3,
        indexes_in_table_info: bool = False,
        custom_table_info: dict | None = None,
        view_support: bool = False,
        max_string_length: int = 300,
        mschema: MSchema | None = None,
        db_name: str = "",
    ):
        """Initialize the engine, discover tables, and build the MSchema."""
        super().__init__(
            engine,
            schema,
            metadata,
            ignore_tables,
            include_tables,
            sample_rows_in_table_info,
            indexes_in_table_info,
            custom_table_info,
            view_support,
            max_string_length,
        )

        self._db_name = db_name
        # Dictionary to store table names and their corresponding schema
        self._tables_schemas: dict[
            str, str
        ] = {}  # For MySQL and similar databases, if no schema is specified but db_name is provided,
        # use db_name as the schema to avoid getting tables from all databases
        if schema is None and db_name:
            if self._engine.dialect.name == "mysql":
                schema = db_name
            elif self._engine.dialect.name == "postgresql":
                # For PostgreSQL, use 'public' as default schema
                schema = "public"

        # If a schema is specified, filter by that schema and store that value for every table.
        if schema:
            self._usable_tables = [
                table_name for table_name in self._usable_tables if self._inspector.has_table(table_name, schema)
            ]
            for table_name in self._usable_tables:
                self._tables_schemas[table_name] = schema
        else:
            all_tables = []
            # Iterate through all available schemas
            for s in self.get_schema_names():
                tables = self._inspector.get_table_names(schema=s)
                all_tables.extend(tables)
                for table in tables:
                    self._tables_schemas[table] = s
            self._usable_tables = all_tables

        self._dialect = engine.dialect.name
        self._sqlite_comments = self.load_sqlite_comments()
        db_comment = self._sqlite_comments.get("__database__", "") if self._engine.dialect.name == "sqlite" else ""
        if mschema is not None:
            self._mschema = mschema
        else:
            self._mschema = MSchema(db_id=db_name, schema=schema)
            self._mschema.database_comment = db_comment
            self.init_mschema()

    @property
    def mschema(self) -> MSchema:
        """Return M-Schema"""
        return self._mschema

    def get_pk_constraint(self, table_name: str) -> list[str]:
        """Return the primary key columns for a table."""
        return self._inspector.get_pk_constraint(table_name, self._tables_schemas[table_name])["constrained_columns"]

    def load_sqlite_comments(self) -> dict:
        """Load table and field comments from a `.comments.yaml` sidecar, if present."""
        if self._engine.dialect.name != "sqlite":
            return {}
        db_path = self._engine.url.database
        if not db_path:
            return {}
        if db_path.startswith("file:"):
            db_path = db_path[len("file:") :]
        db_path = os.path.abspath(db_path)
        comments_path = os.path.splitext(db_path)[0] + ".comments.yaml"
        if not os.path.exists(comments_path):
            return {}
        try:
            with open(comments_path) as f:
                data = yaml.safe_load(f) or {}
            return data
        except Exception:
            return {}

    def get_table_comment(self, table_name: str):
        """Return the comment for a table."""
        if self._engine.dialect.name == "sqlite":
            entry = self._sqlite_comments.get(table_name)
            if isinstance(entry, dict):
                return entry.get("__table__", "")
            return entry or ""
        try:
            return self._inspector.get_table_comment(table_name, self._tables_schemas[table_name])["text"]
        except Exception:
            return ""

    def get_field_comment(self, table_name: str, field_name: str) -> str:
        """Return the comment for a field."""
        if self._engine.dialect.name == "sqlite":
            entry = self._sqlite_comments.get(table_name)
            if isinstance(entry, dict):
                return entry.get(field_name, "")
        return ""

    def default_schema_name(self) -> str | None:
        """Return the database's default schema name."""
        return self._inspector.default_schema_name

    def get_schema_names(self) -> list[str]:
        """Return all schema names in the database."""
        return self._inspector.get_schema_names()

    def get_foreign_keys(self, table_name: str):
        """Return the foreign keys of a table."""
        return self._inspector.get_foreign_keys(table_name, self._tables_schemas[table_name])

    def get_unique_constraints(self, table_name: str):
        """Return the unique constraints of a table."""
        return self._inspector.get_unique_constraints(table_name, self._tables_schemas[table_name])

    def fetch_distinct_values(self, table_name: str, column_names: list[str], max_num: int = 5) -> dict[str, list]:
        """Fetch distinct example values for all columns in one DB round-trip via CTE + UNION ALL."""
        if not column_names:
            return {}
        schema = self._tables_schemas[table_name]
        table_ref = f'"{schema}"."{table_name}"' if schema else f'"{table_name}"'

        cte_clauses = []
        select_clauses = []
        for i, col in enumerate(column_names):
            col_id = col.replace('"', '""')
            col_str = col.replace("'", "''")
            cte_clauses.append(
                f'd_{i} AS (SELECT DISTINCT "{col_id}" AS val FROM {table_ref}'
                f' WHERE "{col_id}" IS NOT NULL AND "{col_id}" != \'\' LIMIT {max_num})'
            )
            select_clauses.append(f"SELECT '{col_str}' AS col_name, val FROM d_{i}")

        sql = "WITH " + ", ".join(cte_clauses) + " " + " UNION ALL ".join(select_clauses)

        out: dict[str, list] = {col: [] for col in column_names}
        try:
            with self._engine.connect() as connection:
                for col_name, val in connection.execute(text(sql)).fetchall():
                    if col_name in out:
                        out[col_name].append(val)
        except Exception:
            pass
        return out

    def _fetch_all_tables_distinct_values(self, table_meta: dict, max_num: int = 5) -> dict[str, dict[str, list]]:
        """Fetch distinct example values for every column in every table in one DB round-trip."""
        cte_clauses: list[str] = []
        select_clauses: list[str] = []
        idx = 0
        for table_name, meta in table_meta.items():
            schema = self._tables_schemas[table_name]
            table_ref = f'"{schema}"."{table_name}"' if schema else f'"{table_name}"'
            tbl_str = table_name.replace("'", "''")
            for col in meta["column_names"]:
                col_id = col.replace('"', '""')
                col_str = col.replace("'", "''")
                cte_clauses.append(
                    f'd_{idx} AS (SELECT DISTINCT "{col_id}" AS val FROM {table_ref}'
                    f' WHERE "{col_id}" IS NOT NULL AND "{col_id}" != \'\' LIMIT {max_num})'
                )
                select_clauses.append(f"SELECT '{tbl_str}' AS tbl_name, '{col_str}' AS col_name, val FROM d_{idx}")
                idx += 1

        out: dict[str, dict[str, list]] = {
            tname: {col: [] for col in meta["column_names"]} for tname, meta in table_meta.items()
        }
        if not cte_clauses:
            return out

        sql = "WITH " + ", ".join(cte_clauses) + " " + " UNION ALL ".join(select_clauses)
        try:
            with self._engine.connect() as connection:
                for tbl_name, col_name, val in connection.execute(text(sql)).fetchall():
                    if tbl_name in out and col_name in out[tbl_name]:
                        out[tbl_name][col_name].append(val)
        except Exception:
            pass
        return out

    def init_mschema(self):
        """Build the MSchema from inspector metadata and example values."""
        # Step 1: Sequential — gather inspector metadata (Inspector is not thread-safe)
        table_meta: dict = {}
        for table_name in self._usable_tables:
            table_comment = self.get_table_comment(table_name)
            table_comment = "" if table_comment is None else table_comment.strip()
            schema_name = self._tables_schemas[table_name]
            if (
                self._engine.dialect.name == "mysql"
                and schema_name == self._db_name
                or self._engine.dialect.name == "postgresql"
                and schema_name == "public"
            ):
                table_with_schema = table_name
            else:
                table_with_schema = schema_name + "." + table_name
            pks = self.get_pk_constraint(table_name)
            fks = self.get_foreign_keys(table_name)
            fields = self._inspector.get_columns(table_name, schema=schema_name)
            column_names = [f["name"] for f in fields]
            table_meta[table_name] = {
                "table_with_schema": table_with_schema,
                "table_comment": table_comment,
                "pks": pks,
                "fks": fks,
                "fields": fields,
                "column_names": column_names,
            }

        # Step 2: Single query — fetch all distinct values across all tables at once
        examples_results = self._fetch_all_tables_distinct_values(table_meta)

        # Step 3: Sequential — assemble mschema
        for table_name in self._usable_tables:
            meta = table_meta[table_name]
            table_with_schema = meta["table_with_schema"]
            examples_map = examples_results[table_name]

            self._mschema.add_table(table_with_schema, fields={}, comment=meta["table_comment"])

            for fk in meta["fks"]:
                referred_schema = fk["referred_schema"]
                for c, r in zip(fk["constrained_columns"], fk["referred_columns"], strict=True):
                    self._mschema.add_foreign_key(table_with_schema, c, referred_schema, fk["referred_table"], r)

            for field in meta["fields"]:
                field_name = field["name"]
                field_type = f"{field['type']!s}"
                primary_key = field_name in meta["pks"]
                field_comment = (field.get("comment") or "").strip()
                yaml_field_comment = self.get_field_comment(table_name, field_name)
                if yaml_field_comment:
                    field_comment = yaml_field_comment
                autoincrement = field.get("autoincrement", False)
                default = field.get("default")
                if default is not None:
                    default = f"{default}"
                examples = examples_to_str(examples_map.get(field_name, []))

                self._mschema.add_field(
                    table_with_schema,
                    field_name,
                    field_type=field_type,
                    primary_key=primary_key,
                    nullable=field["nullable"],
                    default=default,
                    autoincrement=autoincrement,
                    comment=field_comment,
                    examples=examples,
                )

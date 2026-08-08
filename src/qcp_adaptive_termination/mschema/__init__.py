"""
M-Schema: structured database schema representation.

`MSchema` stores tables, fields, foreign keys, and natural-language comments
in a compact format used as LLM context. `SchemaEngine` builds an `MSchema`
from a live SQLAlchemy engine, optionally enriched by a `.comments.yaml`
sidecar file placed next to the `.db` file (same stem).

Example M-Schema text (output of ``MSchema.to_mschema()``)::

    【DB_ID】 company_db
    【Schema】
    # Table: companies, All companies registered in Europe.
    [(id:INTEGER, Primary Key),
    (name:TEXT, Legal company name., Examples: [Apple, Google, SAP]),
    (employees:INTEGER, Headcount as last reported., Examples: [500, 1200, 80000]),
    (country_id:INTEGER)]
    # Table: countries, ISO country reference table.
    [(id:INTEGER, Primary Key),
    (code:TEXT, ISO 3166-1 alpha-2 country code., Examples: [DE, FR, NL]),
    (name:TEXT, Examples: [Germany, France, Netherlands])]
    【Foreign keys】
    companies.country_id=countries.id

Example `mydb.comments.yaml` sidecar (all keys optional)::

    __database__: "European company registry with headcount and revenue."
    companies:
      __table__: "All companies registered in Europe."
      name: "Legal company name."
      employees: "Headcount as last reported."
    countries:
      __table__: "ISO country reference table."
      code: "ISO 3166-1 alpha-2 country code."
"""

from qcp_adaptive_termination.mschema.m_schema import MSchema
from qcp_adaptive_termination.mschema.schema_engine import SchemaEngine

__all__ = ["MSchema", "SchemaEngine"]

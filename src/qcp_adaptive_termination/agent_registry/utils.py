from collections.abc import Callable
from typing import Any, overload

from pydantic import BaseModel, Field, create_model

import qcp_adaptive_termination.database as database
from qcp_adaptive_termination.database import MSchema, SchemaEngine
from qcp_adaptive_termination.llm import runner
from qcp_adaptive_termination.models import (
    AgentRegistryData,
    AgentRole,
    ColumnDescription,
    DataAgent,
    TableDescription,
)
from qcp_adaptive_termination.observability import tracing


def _missing_comments(
    mschema: MSchema,
) -> tuple[list[str], list[tuple[str, str]], bool]:
    """Return (table_names_missing, (table,col) pairs missing, database_missing)."""
    missing_tables = []
    missing_fields = []
    for table_name, table_info in mschema.tables.items():
        if not table_info.get("comment"):
            missing_tables.append(table_name)
        for col_name, field_info in table_info["fields"].items():
            if not field_info.get("comment"):
                missing_fields.append((table_name, col_name))
    db_missing = not mschema.database_comment
    return missing_tables, missing_fields, db_missing


def _create_database_description_model(
    schema_engine: SchemaEngine,
    missing_tables: list[str],
    missing_fields: list[tuple[str, str]],
    db_missing: bool,
) -> type[BaseModel]:
    missing_fields_set = set(missing_fields)

    def create_table_column_model(table_name: str, table_info: dict[str, Any]) -> type[BaseModel] | None:
        fields = {}
        for col_name in table_info["fields"]:
            if (table_name, col_name) in missing_fields_set:
                fields[col_name] = (
                    str,
                    Field(
                        description=f"Natural language description of the column '{col_name}' in table '{table_name}'"
                    ),
                )
        if not fields:
            return None
        return create_model(f"{table_name}ColumnDescriptions", **fields)  # ty: ignore[no-matching-overload]

    schema_dict: dict[str, Any] = {}
    for table_name, table_info in schema_engine.mschema.tables.items():
        col_model = create_table_column_model(table_name, table_info)
        if col_model is not None:
            schema_dict[table_name] = (col_model, Field(...))

    if missing_tables:
        table_desc_fields = {
            table_name: (
                str,
                Field(description=f"Natural language description of the table '{table_name}'"),
            )
            for table_name in missing_tables
        }
        TableDescriptionsModel = create_model(  # ty: ignore[no-matching-overload]
            "TableDescriptionsModel",
            **table_desc_fields,  # type: ignore[reportArgumentType, reportCallIssue]
        )
        schema_dict["table_descriptions"] = (TableDescriptionsModel, Field(...))

    if db_missing:
        schema_dict["database_description"] = (
            str,
            Field(description="Natural language description of the whole database"),
        )

    return create_model("DatabaseDescriptionModel", **schema_dict)


async def load_registry_data(data_agent: DataAgent) -> AgentRegistryData:
    db_engine = database.get_database_engine(data_agent)
    schema_engine = SchemaEngine(db_engine, db_name=data_agent.name)

    missing_tables, missing_fields, db_missing = _missing_comments(schema_engine.mschema)
    needs_llm = missing_tables or missing_fields or db_missing

    if needs_llm:
        m_schema_without_comments = schema_engine.mschema.to_mschema(include_db_id=False)
        DatabaseDescriptionModel = _create_database_description_model(
            schema_engine, missing_tables, missing_fields, db_missing
        )
        result: Any = await runner.run_llm(
            role=AgentRole.DATA_AGENT_SCHEMA_DESCRIPTION,
            agent_name=data_agent.name,
            prompt_parameters={
                "m_schema_without_comments": m_schema_without_comments,
            },
            output_type=DatabaseDescriptionModel,
        )

        missing_fields_set = set(missing_fields)
        for table_name, table_info in schema_engine.mschema.tables.items():
            col_model = getattr(result, table_name, None)
            if col_model is not None:
                for col_name, field_info in table_info["fields"].items():
                    if (table_name, col_name) in missing_fields_set:
                        field_info["comment"] = getattr(col_model, col_name)

        if missing_tables:
            table_desc_result = getattr(result, "table_descriptions", None)
            if table_desc_result is not None:
                for table_name in missing_tables:
                    schema_engine.mschema.tables[table_name]["comment"] = getattr(table_desc_result, table_name)

        if db_missing:
            natural_language_description = result.database_description
            schema_engine.mschema.database_comment = natural_language_description
    else:
        with tracing.own_span(
            "No missing comments in MSchema, skipping LLM call",
            input={
                "agent_name": data_agent.name,
            },
        ):
            pass

    natural_language_description = schema_engine.mschema.database_comment

    individual_descriptions = {
        table_name: TableDescription(
            table_name=table_name,
            natural_language_description=schema_engine.mschema.tables[table_name].get("comment", ""),
            technical_description=table_mschema,
            columns={
                col_name: ColumnDescription(
                    column_name=col_name,
                    natural_language_description=schema_engine.mschema.tables[table_name]["fields"][col_name].get(
                        "comment", ""
                    ),
                    technical_description=field_mschema,
                )
                for col_name, field_mschema in schema_engine.mschema.field_mschema(table_name).items()
            },
        )
        for table_name, table_mschema in schema_engine.mschema.table_mschema().items()
    }

    return AgentRegistryData(
        natural_language_description=natural_language_description,
        technical_description=schema_engine.mschema.to_mschema(),
        query_capabilities=["Text-to-SQL"],
        individual_descriptions=individual_descriptions,
    )


@overload
def map_entries[T](data_agents: list[AgentRegistryData], mapping: Callable[[AgentRegistryData], T]) -> list[T]: ...
@overload
def map_entries[T](data_agents: AgentRegistryData, mapping: Callable[[AgentRegistryData], T]) -> T: ...
def map_entries(data_agents, mapping):
    if isinstance(data_agents, list):
        return [mapping(agent) for agent in data_agents]
    return mapping(data_agents)

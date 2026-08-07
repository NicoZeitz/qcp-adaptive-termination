from typing import Any

from pydantic import BaseModel, create_model, Field

from qcp_adaptive_termination.agent_configuration import (
    AgentRole,
    get_agent_configuration,
)
from qcp_adaptive_termination.data_agent.get_db_engine import get_db_engine
from qcp_adaptive_termination.span import own_custom_span
from qcp_adaptive_termination.model.agent.agent_registry_data import (
    AgentRegistryData,
    ColumnDescription,
    TableDescription,
)
from qcp_adaptive_termination.model.agent.data_agent import DataAgent
from qcp_adaptive_termination.mschema.schema_engine import SchemaEngine
from qcp_adaptive_termination.mschema.m_schema import MSchema
from qcp_adaptive_termination.run_agent import run_llm


def missing_comments(
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


def create_database_description_model(
    schema_engine: SchemaEngine,
    missing_tables: list[str],
    missing_fields: list[tuple[str, str]],
    db_missing: bool,
) -> type[BaseModel]:
    missing_fields_set = set(missing_fields)

    def create_table_column_model(
        table_name: str, table_info: dict[str, Any]
    ) -> type[BaseModel] | None:
        fields = {}
        for col_name in table_info["fields"].keys():
            if (table_name, col_name) in missing_fields_set:
                fields[col_name] = (
                    str,
                    Field(
                        description=f"Natural language description of the column '{col_name}' in table '{table_name}'"
                    ),
                )
        if not fields:
            return None
        return create_model(f"{table_name}ColumnDescriptions", **fields)

    schema_dict: dict[str, Any] = {}
    for table_name, table_info in schema_engine.mschema.tables.items():
        col_model = create_table_column_model(table_name, table_info)
        if col_model is not None:
            schema_dict[table_name] = (col_model, Field(...))

    if missing_tables:
        table_desc_fields = {
            table_name: (
                str,
                Field(
                    description=f"Natural language description of the table '{table_name}'"
                ),
            )
            for table_name in missing_tables
        }
        TableDescriptionsModel = create_model(
            "TableDescriptionsModel",
            **table_desc_fields,  # type: ignore[call-overload]
        )
        schema_dict["table_descriptions"] = (TableDescriptionsModel, Field(...))

    if db_missing:
        schema_dict["database_description"] = (
            str,
            Field(description="Natural language description of the whole database"),
        )

    return create_model("DatabaseDescriptionModel", **schema_dict)


async def load_agent_registry_data(data_agent: DataAgent) -> AgentRegistryData:
    db_engine = get_db_engine(data_agent)
    schema_engine = SchemaEngine(db_engine, db_name=data_agent.name)

    missing_tables, missing_fields, db_missing = missing_comments(schema_engine.mschema)
    needs_llm = missing_tables or missing_fields or db_missing

    if needs_llm:
        m_schema_without_comments = schema_engine.mschema.to_mschema(
            include_db_id=False
        )
        DatabaseDescriptionModel = create_database_description_model(
            schema_engine, missing_tables, missing_fields, db_missing
        )
        result = await run_llm(
            agent_configuration=get_agent_configuration(
                AgentRole.DATA_AGENT_SCHEMA_DESCRIPTION, data_agent.name
            ),
            prompt_name="discovery-data-agent-sql-schema-description",
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
                    schema_engine.mschema.tables[table_name]["comment"] = getattr(
                        table_desc_result, table_name
                    )

        if db_missing:
            natural_language_description = getattr(result, "database_description")
            schema_engine.mschema.database_comment = natural_language_description
    else:
        with own_custom_span(
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
            natural_language_description=schema_engine.mschema.tables[table_name].get(
                "comment", ""
            ),
            technical_description=table_mschema,
            columns={
                col_name: ColumnDescription(
                    column_name=col_name,
                    natural_language_description=schema_engine.mschema.tables[
                        table_name
                    ]["fields"][col_name].get("comment", ""),
                    technical_description=field_mschema,
                )
                for col_name, field_mschema in schema_engine.mschema.field_mschema(
                    table_name
                ).items()
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

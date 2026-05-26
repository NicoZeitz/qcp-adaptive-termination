from pydantic import BaseModel, create_model, Field

from stages.common.agent_configuration import get_data_agent_agent_configuration
from stages.common.data_agent.sql_data_agent.get_db_connection import get_db_engine
from utils.run_agent import run_agent
from qtypes.data_agent import DataAgent
from stages.common.mschema.schema_engine import SchemaEngine


def create_database_description_model(schema_engine: SchemaEngine) -> type[BaseModel]:
    schema_dict = {}
    for table_name, table_info in schema_engine.mschema.tables.items():
        fields = {}
        for col_name in table_info["fields"].keys():
            fields[col_name] = (str, Field(...))
        schema_dict[table_name] = (
            create_model(f"{table_name}ColumnDescriptions", **fields),
            Field(...),
        )
    schema_dict["database_description"] = (str, Field(...))

    return create_model("DatabaseDescriptionModel", **schema_dict)


async def sql_data_agent_query_capabilities(data_agent: DataAgent) -> list[str]:
    return ["Text-to-SQL"]


async def sql_data_and_nl_description(data_agent: DataAgent) -> tuple[str, str]:
    db_engine = get_db_engine(data_agent)
    schema_engine = SchemaEngine(db_engine, db_name=data_agent.name)
    m_schema_without_comments = schema_engine.mschema.to_mschema()

    DatabaseDescriptionModel = create_database_description_model(schema_engine)

    result = await run_agent(
        name_postfix="Schema Description",
        agent_configuration=get_data_agent_agent_configuration(data_agent),
        prompt_name="SQL Data Agent/Schema Description",
        prompt_parameters={
            "m_schema_without_comments": m_schema_without_comments,
        },
        output_type=DatabaseDescriptionModel,
    )

    # Update the schema with the generated descriptions
    for table_name, table_info in schema_engine.mschema.tables.items():
        for col_name, field_info in table_info["fields"].items():
            description = getattr(getattr(result, table_name, {}), col_name, None)
            if description:
                field_info["comment"] = description

    natural_language_description = getattr(result, "database_description")
    return schema_engine.mschema.to_mschema(), natural_language_description

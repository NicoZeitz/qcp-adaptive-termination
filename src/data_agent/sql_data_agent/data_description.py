from async_lru import alru_cache
from pydantic import BaseModel, create_model, Field
from sqlalchemy import Engine

from data_agent.data_agent import DataAgent
from data_agent.sql_data_agent.mschema.schema_engine import SchemaEngine
from utils.prompt import load_prompt
from utils.run_agent import run_agent


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
    return create_model("DatabaseDescriptionModel", **schema_dict)


@alru_cache
async def data_description(data_agent: DataAgent, db_engine: Engine) -> str:
    schema_engine = SchemaEngine(db_engine, db_name=data_agent.name)
    m_schema_without_comments = schema_engine.mschema.to_mschema()

    DatabaseDescriptionModel = create_database_description_model(schema_engine)

    result = await run_agent(
        name_postfix="Schema Description",
        agent_configuration=data_agent.agent_configuration,
        instructions=load_prompt("SQL Data Agent/Schema Description/Instructions", {}),
        prompt=load_prompt(
            "SQL Data Agent/Schema Description/Prompt",
            {"m_schema_without_comments": m_schema_without_comments},
        ),
        output_type=DatabaseDescriptionModel,
    )

    # Update the schema with the generated descriptions
    for table_name, table_info in schema_engine.mschema.tables.items():
        for col_name, field_info in table_info["fields"].items():
            description = getattr(getattr(result, table_name, {}), col_name, None)
            if description:
                field_info["comment"] = description

    return schema_engine.mschema.to_mschema()

from pydantic import BaseModel, create_model, Field

from common.agent_configuration import get_data_agent_agent_configuration
from common.data_agent.get_db_engine import get_db_engine
from common.model.agent.agent_data import AgentData
from common.model.agent.data_agent import DataAgent
from common.mschema.schema_engine import SchemaEngine
from common.run_agent import run_llm


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


async def sql_load_agent_data(data_agent: DataAgent) -> AgentData:
    db_engine = get_db_engine(data_agent)
    schema_engine = SchemaEngine(db_engine, db_name=data_agent.name)
    m_schema_without_comments = schema_engine.mschema.to_mschema()

    DatabaseDescriptionModel = create_database_description_model(schema_engine)

    result = await run_llm(
        agent_configuration=get_data_agent_agent_configuration(
            data_agent, "Schema Description"
        ),
        prompt_name="discovery-data-agent-sql-schema-description",
        prompt_parameters={
            "m_schema_without_comments": m_schema_without_comments,
        },
        output_type=DatabaseDescriptionModel,
    )

    for table_name, table_info in schema_engine.mschema.tables.items():
        for col_name, field_info in table_info["fields"].items():
            description = getattr(getattr(result, table_name, {}), col_name, None)
            if description:
                field_info["comment"] = description

    natural_language_description = getattr(result, "database_description")

    return AgentData(
        technical_description=schema_engine.mschema.to_mschema(),
        natural_language_description=natural_language_description,
        query_capabilities=["Text-to-SQL"],
        embedding_descriptions=schema_engine.mschema.table_embedding_texts(),
        table_column_names={
            table_name: {
                col_name: field_info.get("comment", "")
                for col_name, field_info in table_info["fields"].items()
            }
            for table_name, table_info in schema_engine.mschema.tables.items()
        },
    )
